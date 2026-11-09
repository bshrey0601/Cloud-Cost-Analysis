"""Unit tests that run without the warehouse (fast) plus integration tests that use it.

    pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cca import anomaly as A  # noqa: E402
from cca import forecasting as F  # noqa: E402
from cca import pricing  # noqa: E402
from cca.auditor import _parse, audit  # noqa: E402
from cca.ledger import Ledger  # noqa: E402
from cca.warehouse import MARTS_DIR, active_db_path  # noqa: E402

needs_db = pytest.mark.skipif(not (MARTS_DIR / "mart_hourly_total.parquet").exists(), reason="marts not built")


def synthetic(n_days=30, level=1000.0, noise=5.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n_days * 24)
    return level + 30 * np.sin(2 * np.pi * t / 24) + rng.normal(0, noise, len(t))


# ------------------------------------------------------------------ pricing
def test_price_model_reproduces_list_prices():
    assert pricing.linear_price(2, 8) == pytest.approx(0.096)    # D2s_v5
    assert pricing.linear_price(2, 16) == pytest.approx(0.126)   # E2s_v5
    assert pricing.linear_price(8, 32) == pytest.approx(0.384)   # D8s_v5
    assert pricing.price_for("2", "4")[1] == 0.0846               # exact F2s_v2 list price


def test_every_trace_shape_is_priced():
    assert len(pricing.shape_rows()) == len(pricing.CORE_BUCKET_VALUE) * len(pricing.MEMORY_BUCKET_VALUE)


# --------------------------------------------------------------- forecasting
def test_forecaster_tracks_a_stable_series():
    y = synthetic()
    fc = F.CloudSpendForecaster(horizon=24).fit(y[:-24]).predict(24)
    assert F.wape(y[-24:], fc) < 0.01


def test_conformal_interval_contains_point_and_is_ordered():
    fc = F.forecast_with_intervals(synthetic(), 48)
    assert np.all(fc.lower <= fc.point) and np.all(fc.point <= fc.upper)
    assert fc.total_lower <= fc.total_point <= fc.total_upper


def test_damped_trend_does_not_explode_on_a_last_hour_dip():
    """Regression test for the end-of-trace artifact: one low final hour must not drag a week forecast."""
    y = synthetic()
    y[-1] *= 0.92
    week = F.ProfileDampedTrend().fit(y).predict(168)
    assert abs(week.mean() / y[-168:].mean() - 1) < 0.1


# ------------------------------------------------------------------- anomaly
def test_detector_finds_an_injected_spike_and_stays_quiet_otherwise():
    y = synthetic()
    clean_events = A.RobustSeasonalDetector().detect(y)
    y2 = y.copy()
    y2[400:403] *= 1.5
    events = A.RobustSeasonalDetector().detect(y2)
    assert len(clean_events) == 0
    assert any(e.start_hour <= 402 and e.end_hour >= 400 and e.direction == "spike" for e in events)


def test_detector_scale_floor_handles_perfectly_flat_series():
    y = np.full(720, 50.0)
    sc = A.RobustSeasonalDetector().score(y)
    assert np.nanmax(np.abs(sc.z)) < 1e-6           # no division blow-up
    y[500] = 80.0
    assert sc.pipe(lambda _: A.RobustSeasonalDetector().score(y)).flag.iloc[500]


# ------------------------------------------------------------------- auditor
def _ledger():
    led = Ledger()
    led.post("Daily avg spend, days 1-7 (base)", 704_912.0, "usd", "x", {}, "k")
    led.post("Daily avg spend, days 8-14 (comparison)", 723_812.0, "usd", "x", {}, "k")
    led.post("Change in daily avg spend (comparison - base)", 18_900.0, "usd", "x", {}, "k")
    led.post("Change in daily avg spend, %", 0.0268, "pct", "x", {}, "k")
    return led


def test_parser_reads_money_percent_and_skips_references():
    vals = [(n.kind, round(n.value, 4)) for n in _parse("Days 18-20: SUB-0100 spent $1.2M, up 3.5% over 30 days on 24c_64g")]
    assert vals == [("usd", 1_200_000.0), ("pct", 0.035)]


def test_auditor_passes_a_correct_memo():
    memo = ("Average daily spend increased from $704.9K [F1] in days 1-7 to $723.8K [F2] in days 8-14, "
            "rising by $18.9K [F3] per day (2.7% [F4]).")
    assert audit(memo, _ledger()).passed


@pytest.mark.parametrize("bad, kind", [
    ("Spend rose by $19.9K [F3] per day.", "value_mismatch"),
    ("Spend rose by $18.9K per day.", "uncited_number"),
    ("Spend rose by $18.9M [F3] per day.", "value_mismatch"),
    ("Spend fell by $18.9K [F3] per day.", "direction"),
    ("Spend decreased from $704.9K [F1] to $723.8K [F2].", "direction"),
    ("Spend rose by $18.9K [F9] per day.", "unknown_citation"),
    ("There were 6,298 VMs and $0.00 of excess spend.", "uncited_number"),   # regression: bare numbers before words
])
def test_auditor_catches_each_fault(bad, kind):
    rep = audit(bad, _ledger())
    assert not rep.passed and kind in {i["type"] for i in rep.issues}


# --------------------------------------------------------- integration (db)
@needs_db
def test_reconciliation_table_all_pass():
    import duckdb
    con = duckdb.connect(str(active_db_path()), read_only=True)
    statuses = [r[0] for r in con.execute("SELECT status FROM dq_reconciliation").fetchall()]
    assert statuses and set(statuses) == {"PASS"}


@needs_db
def test_offline_agent_memo_passes_audit_and_reconciles():
    from cca.agent import CostAnalystAgent
    res = CostAnalystAgent(llm=None).answer("Why did spend change in days 18-20 compared with days 11-17?")
    assert res.audit.passed and res.audit.facts_reconciled > 0


@needs_db
def test_auditor_catches_ledger_tampering():
    from copy import deepcopy

    from cca.agent import CostAnalystAgent
    res = CostAnalystAgent(llm=None).answer("Where can we save money?")
    led = deepcopy(res.ledger)
    led.facts["F2"].value *= 1.1
    memo = res.memo.replace(f"{res.ledger.facts['F2'].display()} [F2]", f"{led.facts['F2'].display()} [F2]")
    rep = audit(memo, led, str(active_db_path()))
    assert not rep.passed and "reconciliation" in {i["type"] for i in rep.issues}


@needs_db
def test_llm_loop_revises_then_passes_with_scripted_model():
    """Exercise plan -> write -> audit FAIL -> revise -> audit PASS without any API key."""
    from cca.agent import CostAnalystAgent
    from cca.llm import ScriptedLLM

    plan = '{"steps": [{"tool": "optimization_opportunities", "args": {}}]}'
    llm = ScriptedLLM(responses=[plan, '{"steps": []}', "PLACEHOLDER", "PLACEHOLDER"])
    agent = CostAnalystAgent(llm=llm)
    # Pre-compute the true fact to build a wrong draft and a corrected revision.
    probe = CostAnalystAgent(llm=None).answer("Where can we save money?")
    right = probe.ledger.facts["F2"].display()
    llm.responses[1] = '{"steps": []}'
    llm.responses[2] = "**Answer:** Hard savings are $9.99M [F2]."
    llm.responses[3] = f"**Answer:** Hard savings are {right} [F2]."
    res = agent.answer("Where can we save money?")
    assert not res.first_draft_audit.passed
    assert res.audit.passed and res.revisions == 1 and not res.fell_back


# ------------------------------------------------------------- root cause
def _toy_rootcause_db():
    """One team, four deployments: one launched, one shut down, one migrated, one unchanged."""
    import duckdb
    con = duckdb.connect()
    con.execute("""CREATE TABLE dim_deployment AS SELECT * FROM (VALUES
        ('DEP-1', 'SUB-T', 10, 8, 30, 'Interactive', 'Standard_D2s_v5'),
        ('DEP-2', 'SUB-T', 5, 1, 7, 'Unknown', 'Standard_D2s_v5'),
        ('DEP-3', 'SUB-T', 20, 1, 30, 'Interactive', 'custom_2vcpu_2gb'),
        ('DEP-4', 'SUB-T', 4, 1, 30, 'Interactive', 'Standard_D2s_v5'))
        t(deployment_alias, subscription_alias, vm_count, first_day, last_day, primary_category, primary_sku)""")
    rows = []
    for d in range(1, 31):
        if d >= 8:
            rows.append((d, "DEP-1", "SUB-T", 500.0, 0, 10 if d == 8 else 0))
        if d <= 7:
            rows.append((d, "DEP-2", "SUB-T", 200.0, 0, 0))
        rows.append((d, "DEP-3", "SUB-T", 300.0 if d <= 7 else 50.0, 0, 0))
        rows.append((d, "DEP-4", "SUB-T", 100.0, 0, 0))
    con.execute("CREATE TABLE mart_daily_deployment (trace_day INT, deployment_alias VARCHAR, subscription_alias VARCHAR,"
                " cost_usd DOUBLE, vm_hours DOUBLE, vms_created INT)")
    con.executemany("INSERT INTO mart_daily_deployment VALUES (?, ?, ?, ?, ?, ?)", rows)
    con.execute("ALTER TABLE mart_daily_deployment ADD COLUMN active_vms INT DEFAULT 0")
    return con


def test_root_cause_decomposes_exactly_and_classifies_drivers():
    from cca.rootcause import explain_change
    r = explain_change(_toy_rootcause_db(), "SUB-T", 1, 7, 8, 14, top_n=10)
    kinds = {d["name"]: d["kind"] for d in r["drivers"]}
    assert kinds == {"DEP-1": "launched", "DEP-3": "scaled down", "DEP-2": "shut down"}   # DEP-4 did not move
    assert r["change_usd_per_day"] == pytest.approx(500 - 200 - 250)
    assert sum(d["change_usd_per_day"] for d in r["drivers"]) == pytest.approx(r["change_usd_per_day"])
    assert r["drivers"][0]["vms_created_in_comparison"] == 10
    assert r["drivers"][[d["name"] for d in r["drivers"]].index("DEP-3")]["sku"] == "2 vCPU / 2 GB VMs"


def test_root_cause_flags_a_migration():
    from cca.rootcause import explain_change
    con = _toy_rootcause_db()
    con.execute("DELETE FROM mart_daily_deployment WHERE deployment_alias = 'DEP-2'")
    r = explain_change(con, "SUB-T", 1, 7, 8, 14, top_n=10)
    # DEP-3 lost 250/day and DEP-1 gained 500/day: not similar enough to call a migration
    assert r["possible_migrations"] == []
    con.execute("UPDATE mart_daily_deployment SET cost_usd = 270 WHERE deployment_alias = 'DEP-1'")
    r = explain_change(con, "SUB-T", 1, 7, 8, 14, top_n=10)
    assert r["possible_migrations"] == [{"from": "DEP-3", "to": "DEP-1", "from_change_usd_per_day": -250.0,
                                         "to_change_usd_per_day": 270.0}]


def test_auditor_treats_deployment_ids_as_identifiers():
    assert [n.text for n in _parse("DEP-00034 launched 1,934 VMs")] == ["1,934"]


@needs_db
def test_agent_root_cause_memo_passes_audit():
    from cca.agent import CostAnalystAgent
    res = CostAnalystAgent(llm=None).answer("Why did SUB-0041 spend jump on day 8?")
    assert [s["tool"] for s in res.steps] == ["explain_change"]
    assert res.audit.passed and "DEP-" in res.memo


# ------------------------------------------------------------------- QBR pack
@needs_db
def test_qbr_budget_arithmetic_and_audited_deck():
    import io

    from pptx import Presentation

    from cca.qbr import PERIOD_END, PERIOD_START, PLAN_END, PLAN_START, budget_vs_actual, build_qbr
    from cca.qbr_pptx import build_pptx
    con = duckdb.connect(str(active_db_path()), read_only=True)
    teams = budget_vs_actual(con)
    plan = con.execute("SELECT SUM(cost_usd) FROM mart_daily_driver WHERE trace_day BETWEEN ? AND ?",
                       [PLAN_START, PLAN_END]).fetchone()[0]
    days_plan, days_period = PLAN_END - PLAN_START + 1, PERIOD_END - PERIOD_START + 1
    assert teams.budget_usd.sum() == pytest.approx(plan / days_plan * days_period)
    d = build_qbr(con, str(active_db_path()))
    assert d.audit.passed and d.audit.facts_reconciled > 0
    assert all("[F" not in line for line in d.commentary)           # citations stripped for executives
    prs = Presentation(io.BytesIO(build_pptx(d)))
    assert len(prs.slides) == 9
    charts = [sh for sl in prs.slides for sh in sl.shapes if sh.has_chart]
    assert len(charts) == 4                                          # native, editable charts


def test_team_week_ahead_routes_to_validated_baseline():
    assert F.choose_model("SUB-0041", 168) == "seasonal_naive_day"
    assert F.choose_model("SUB-0041", 24) == "blend"
    assert F.choose_model("total", 168) == "blend"
    assert F.CloudSpendForecaster().step_aware is False       # v2 experiment stays off by default


def test_auditor_binds_each_number_to_its_own_citation():
    """Regression: with a $0 base, comparison and change are equal; a swapped citation must still fail."""
    led = Ledger()
    led.post("Base", 0.0, "usd", "x", {}, "k")
    led.post("Comparison", 4100.0, "usd", "x", {}, "k")
    led.post("Change in spend", 4100.0, "usd", "x", {}, "k")
    good = "Spend rose from $0.00 [F1] to $4.1K [F2], a move of $4.1K [F3]."
    swapped = "Spend rose from $0.00 [F1] to $4.1K [F1], a move of $4.1K [F3]."
    dropped = "Spend rose from $0.00 [F1] to $4.1K [F2], a move of $4.1K."
    assert audit(good, led).passed
    assert not audit(swapped, led).passed
    assert not audit(dropped, led).passed


def test_auditor_checks_the_cited_fact_is_about_the_named_entity():
    led = Ledger()
    led.post("Change in SUB-0010 daily avg spend (comparison - base)", 1312.6, "usd", "x", {}, "k")
    led.post("Change in daily avg spend from DEP-00098 (scaled up)", 1271.7, "usd", "x", {}, "k")
    assert audit("DEP-00098 scaled up, adding $1.3K [F2] per day.", led).passed
    rep = audit("DEP-00098 scaled up, adding $1.3K [F1] per day.", led)   # same rounding, wrong entity
    assert not rep.passed and "citation_entity" in {i["type"] for i in rep.issues}
