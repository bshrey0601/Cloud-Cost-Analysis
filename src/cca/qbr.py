"""Automated QBR (quarterly business review) pack for cloud spend.

Builds everything a Product Finance analyst assembles by hand each review cycle:
budget vs. actual, variance by team with root cause, anomalies, savings and outlook --
plus executive commentary whose every number is verified by the Auditor before it goes
on a slide. Output: an editable PowerPoint with native (editable) charts.

Budgeting convention (standard FP&A run-rate method): the plan is set on `plan_end` day
using the average daily spend of the preceding week; the budget for the review period is
that run-rate x the number of days in the period. Teams with no spend in the plan week
have no budget, so any spend they incur is reported as unbudgeted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import duckdb
import numpy as np
import pandas as pd

from cca import anomaly as A
from cca import forecasting as F
from cca import optimization as O
from cca import rootcause as RC
from cca.auditor import AuditReport, audit
from cca.ledger import Ledger, format_value
from cca.tools import run_tool

PLAN_START, PLAN_END, PERIOD_START, PERIOD_END = 8, 14, 15, 30


def budget_vs_actual(con: duckdb.DuckDBPyConnection, dimension: str = "subscription",
                     plan_start: int = PLAN_START, plan_end: int = PLAN_END,
                     period_start: int = PERIOD_START, period_end: int = PERIOD_END) -> pd.DataFrame:
    col = {"subscription": "subscription_alias", "category": "vm_category"}[dimension]
    n_plan, n_period = plan_end - plan_start + 1, period_end - period_start + 1
    df = con.execute(f"""
        SELECT {col} AS name,
               SUM(CASE WHEN trace_day BETWEEN ? AND ? THEN cost_usd ELSE 0 END) / ? * ? AS budget_usd,
               SUM(CASE WHEN trace_day BETWEEN ? AND ? THEN cost_usd ELSE 0 END)          AS actual_usd
        FROM mart_daily_driver GROUP BY 1""",
        [plan_start, plan_end, n_plan, n_period, period_start, period_end]).df()
    df["variance_usd"] = df.actual_usd - df.budget_usd
    df["variance_pct"] = np.where(df.budget_usd > 0, df.variance_usd / df.budget_usd.where(df.budget_usd > 0), np.nan)
    df["unbudgeted"] = df.budget_usd <= 0
    return df.sort_values("variance_usd", key=abs, ascending=False).reset_index(drop=True)


@dataclass
class QBRData:
    period: tuple[int, int]
    plan: tuple[int, int]
    total: dict
    by_category: pd.DataFrame
    by_team: pd.DataFrame
    team_drivers: list[dict]
    anomalies: list[dict]
    savings: dict
    outlook: dict
    reconciliation: pd.DataFrame
    ledger: Ledger = field(default_factory=Ledger)
    commentary: list[str] = field(default_factory=list)
    audit: AuditReport | None = None


def _same_direction(drivers: list[dict], change: float) -> dict | None:
    """The largest deployment pushing the SAME way as the change (an overrun is explained by what
    grew, an underrun by what shrank), falling back to the largest mover."""
    same = [d for d in drivers if (d["change_usd_per_day"] >= 0) == (change >= 0)]
    return same[0] if same else (drivers[0] if drivers else None)


def _strip(text: str) -> str:
    import re
    return re.sub(r"\s*\[F\d+(?:\s*,\s*F\d+)*\]", "", text)


def build_qbr(con: duckdb.DuckDBPyConnection, db_path: str, top_teams: int = 8) -> QBRData:
    teams = budget_vs_actual(con, "subscription")
    cats = budget_vs_actual(con, "category")
    total = {"budget_usd": float(teams.budget_usd.sum()), "actual_usd": float(teams.actual_usd.sum())}
    total["variance_usd"] = total["actual_usd"] - total["budget_usd"]
    total["variance_pct"] = total["variance_usd"] / total["budget_usd"]
    total["unbudgeted_usd"] = float(teams.loc[teams.unbudgeted, "actual_usd"].sum())

    # Root cause for the biggest team variances: which deployment moved?
    drivers = []
    for r in teams.head(top_teams).itertuples():
        try:
            rc = RC.explain_change(con, r.name, PLAN_START, PLAN_END, PERIOD_START, PERIOD_END, top_n=10)
            d = _same_direction(rc["drivers"], r.variance_usd)
        except ValueError:
            d = None
        drivers.append({"team": r.name, "budget_usd": r.budget_usd, "actual_usd": r.actual_usd,
                        "variance_usd": r.variance_usd, "variance_pct": r.variance_pct,
                        "top_deployment": d["name"] if d else "-", "what_happened": d["kind"] if d else "-",
                        "deployment_first_day": d["first_day"] if d else None,
                        "deployment_share": d["share_of_gross_change"] if d else None})

    # Anomalies that started inside the review period, with their root cause
    hourly = con.execute("SELECT subscription_alias s, hour_idx, cost_usd FROM mart_hourly_top_subscription").df()
    piv = hourly.pivot(index="hour_idx", columns="s", values="cost_usd").reindex(range(720)).fillna(0.0)
    det = A.RobustSeasonalDetector()
    events = [e.as_dict() for c in piv.columns for e in det.detect(piv[c].to_numpy(), c)]
    events = [e for e in events if e["start_day"] >= PERIOD_START]
    events.sort(key=lambda e: -abs(e["excess_cost_usd"]))
    anomalies = []
    for e in events[:4]:
        rc = RC.explain_event(con, e["series"], e["start_day"], top_n=10)
        d = _same_direction(rc["drivers"], e["excess_cost_usd"]) if rc else None
        anomalies.append({**e, "top_deployment": d["name"] if d else "-", "what_happened": d["kind"] if d else "-",
                          "deployment_vms": d["vm_count"] if d else None})

    opt = O.recommendations(con)
    y = con.execute("SELECT cost_usd FROM mart_hourly_total ORDER BY hour_idx").df().cost_usd.to_numpy()
    fc = F.forecast_with_intervals(y, 7 * 24, level=0.9)
    outlook = {
        "history_days": list(range(17, 31)), "history": [float(y[(d - 1) * 24: d * 24].sum()) for d in range(17, 31)],
        "forecast_days": list(range(31, 38)),
        "forecast": [float(fc.point[i * 24:(i + 1) * 24].sum()) for i in range(7)],
        "lower": [float(fc.lower[i * 24:(i + 1) * 24].sum()) for i in range(7)],
        "upper": [float(fc.upper[i * 24:(i + 1) * 24].sum()) for i in range(7)],
        "total": fc.total_point, "total_lower": fc.total_lower, "total_upper": fc.total_upper,
    }
    data = QBRData((PERIOD_START, PERIOD_END), (PLAN_START, PLAN_END), total, cats, teams, drivers, anomalies,
                   {"hard_savings_usd": opt["hard_savings_usd"], "total_cost_usd": opt["total_cost_usd"],
                    "levers": opt["summary"].to_dict("records")},
                   outlook, con.execute("SELECT * FROM dq_reconciliation").df())
    _audited_commentary(con, data, db_path)
    return data


def _audited_commentary(con, data: QBRData, db_path: str) -> None:
    """Executive commentary built from ledger facts, then checked by the Auditor."""
    led = Ledger()
    res = []
    plan = {"plan_start": PLAN_START, "plan_end": PLAN_END, "period_start": PERIOD_START, "period_end": PERIOD_END}
    bva = run_tool(con, led, "budget_vs_actual", {**plan, "dimension": "subscription", "top_n": 3})
    res.append(bva)
    lab = {led.facts[i].label: i for i in bva["_facts"]}
    f = lambda fid: f"{led.facts[fid].display()} [{fid}]"  # noqa: E731
    fa = lambda fid: f"{format_value(abs(led.facts[fid].value), led.facts[fid].unit)} [{fid}]"  # noqa: E731
    over = bva["variance_usd"] >= 0
    lines = [
        f"Spend in days {PERIOD_START}-{PERIOD_END} was {f(lab['Actual spend, review period'])} against a run-rate "
        f"budget of {f(lab['Budget, review period'])}: {fa(lab['Variance vs budget'])} "
        f"({fa(lab['Variance vs budget, %'])}) {'over' if over else 'under'} plan.",
    ]
    if bva["unbudgeted_usd"] > 0:
        lines.append(f"Teams with no spend in the plan week incurred {f(lab['Unbudgeted spend (teams with no budget)'])} "
                     f"of unbudgeted spend.")
    top = bva["top"][0]
    rc = run_tool(con, led, "explain_change", {"subscription": top["name"], "base_start": PLAN_START,
                                                "base_end": PLAN_END, "comp_start": PERIOD_START,
                                                "comp_end": PERIOD_END, "top_n": 1})
    rlab = {led.facts[i].label: i for i in rc["_facts"]}
    tlab = lab[f"Variance vs budget: {top['name']}"]
    if rc["drivers"]:
        d = rc["drivers"][0]
        share = rlab[f"Share of {top['name']} gross change explained by the top deployment"]
        lines.append(f"The largest team variance is {top['name']}, {fa(tlab)} {'over' if top['variance_usd'] >= 0 else 'under'} "
                     f"budget; {d['name']} ({d['kind']}) explains {f(share)} of its change.")
    if data.anomalies:
        a = data.anomalies[0]
        an = run_tool(con, led, "detect_anomalies", {"scope": a["series"], "top_n": 1})
        alab = {led.facts[i].label: i for i in an["_facts"]}
        ev = an["top_events"][0]
        tag = f"{ev['series']} {ev['direction']} starting day {ev['start_day']}"
        word = "above" if ev["direction"] == "spike" else "below"
        lines.append(f"The largest anomaly in the period was a {ev['direction']} in {ev['series']} on day "
                     f"{ev['start_day']}, {fa(alab[f'Excess cost vs expected: {tag}'])} {word} expected.")
    op = run_tool(con, led, "optimization_opportunities", {})
    olab = {led.facts[i].label: i for i in op["_facts"]}
    lines.append(f"Idle shutdown and rightsizing could save {f(olab['Hard savings (idle + rightsizing), 30 days'])} "
                 f"per month, {f(olab['Hard savings as % of spend'])} of spend.")
    fc = run_tool(con, led, "forecast_spend", {"scope": "total", "horizon_days": 7})
    flab = {led.facts[i].label: i for i in fc["_facts"]}
    w = "total, next 7 days"
    lines.append(f"Spend for the next 7 days is forecast at {f(flab[f'Forecast spend, {w}'])} "
                 f"(90% range {f(flab[f'Forecast 90% lower bound, {w}'])} to {f(flab[f'Forecast 90% upper bound, {w}'])}).")
    memo = "\n".join(lines)
    data.audit = audit(memo, led, db_path)
    data.ledger = led
    data.commentary = [_strip(x) for x in lines]
    if not data.audit.passed:
        raise RuntimeError("QBR commentary failed audit:\n" + data.audit.summary())
