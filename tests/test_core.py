"""Unit tests that run without the warehouse (fast) plus integration tests that use it.

    pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

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
