"""Agent tools. Each tool queries the warehouse (SQL) or runs a model (Python),
returns a JSON-able result, and posts the numbers it found to the fact ledger.

Every tool is deterministic given (warehouse, args), which is what lets the auditor
re-run it on a fresh connection and reconcile each fact.
"""

from __future__ import annotations

import functools
import re
from typing import Any, Callable

import duckdb
import numpy as np

from cca import anomaly as A
from cca import forecasting as F
from cca import optimization as O
from cca import rootcause as RC
from cca.ledger import Ledger
from cca.pricing import PRICE_SOURCE

DIMENSIONS = {"subscription": "subscription_alias", "category": "vm_category", "shape": "shape_id"}
CATEGORIES = ["Delay-insensitive", "Interactive", "Unknown"]


def _dim(dimension: str) -> str:
    if dimension in DIMENSIONS.values():
        return dimension
    if dimension not in DIMENSIONS:
        raise ValueError(f"dimension must be one of {list(DIMENSIONS)}")
    return DIMENSIONS[dimension]


def _days(start_day: int, end_day: int) -> tuple[int, int]:
    s, e = int(start_day), int(end_day)
    if not (1 <= s <= e <= 30):
        raise ValueError("days must satisfy 1 <= start_day <= end_day <= 30 (the trace covers 30 days)")
    return s, e


def get_path(obj: Any, path: str) -> Any:
    for part in path.split("."):
        obj = obj[int(part)] if isinstance(obj, list) else obj[part]
    return obj


# ------------------------------------------------------------------- tools
def describe_dataset(con: duckdb.DuckDBPyConnection) -> dict:
    total = con.execute("SELECT SUM(cost_usd) FROM mart_hourly_total").fetchone()[0]
    subs = con.execute("SELECT COUNT(*) FROM dim_subscription").fetchone()[0]
    vms = con.execute("SELECT SUM(vm_count) FROM dim_subscription").fetchone()[0]
    return {"days": 30, "subscriptions": int(subs), "vms": int(vms), "total_usd": float(total),
            "daily_avg_usd": float(total) / 30,
            "source": "Microsoft Azure Public Dataset V2 (2019 first-party VM trace)",
            "price_source": PRICE_SOURCE}


def spend_summary(con, start_day: int = 1, end_day: int = 30, dimension: str = "subscription",
                  top_n: int = 5) -> dict:
    s, e = _days(start_day, end_day)
    col = _dim(dimension)
    total = con.execute("SELECT SUM(cost_usd) FROM mart_daily_driver WHERE trace_day BETWEEN ? AND ?",
                        [s, e]).fetchone()[0]
    rows = con.execute(
        f"SELECT {col} AS name, SUM(cost_usd) AS cost_usd FROM mart_daily_driver "
        f"WHERE trace_day BETWEEN ? AND ? GROUP BY 1 ORDER BY 2 DESC, 1 LIMIT ?", [s, e, int(top_n)]).fetchall()
    top = [{"name": n, "cost_usd": float(c), "share": float(c) / total} for n, c in rows]
    return {"start_day": s, "end_day": e, "dimension": dimension, "total_usd": float(total),
            "daily_avg_usd": float(total) / (e - s + 1), "top": top}


def compare_periods(con, base_start: int, base_end: int, comp_start: int, comp_end: int,
                    dimension: str = "subscription", top_n: int = 5) -> dict:
    """Variance analysis on *daily average* spend, so windows of different length compare fairly.

    change = comp_avg - base_avg, decomposed exactly by `dimension`
    (driver contributions sum to the total change).
    """
    bs, be = _days(base_start, base_end)
    cs, ce = _days(comp_start, comp_end)
    col = _dim(dimension)
    nb, nc = be - bs + 1, ce - cs + 1
    df = con.execute(f"""
        SELECT {col} AS name,
               SUM(CASE WHEN trace_day BETWEEN ? AND ? THEN cost_usd ELSE 0 END) / ? AS base_avg,
               SUM(CASE WHEN trace_day BETWEEN ? AND ? THEN cost_usd ELSE 0 END) / ? AS comp_avg
        FROM mart_daily_driver GROUP BY 1""", [bs, be, nb, cs, ce, nc]).df()
    df["change"] = df.comp_avg - df.base_avg
    base_avg, comp_avg = float(df.base_avg.sum()), float(df.comp_avg.sum())
    change = comp_avg - base_avg
    df["abs_change"] = df.change.abs()
    df = df.sort_values(["abs_change", "name"], ascending=[False, True]).head(int(top_n))
    drivers = [{"name": r.name, "base_daily_avg_usd": float(r.base_avg), "comp_daily_avg_usd": float(r.comp_avg),
                "change_usd_per_day": float(r.change),
                "share_of_change": float(r.change / change) if change else 0.0}
               for r in df.itertuples()]
    return {"base_days": [bs, be], "comp_days": [cs, ce], "dimension": dimension,
            "base_daily_avg_usd": base_avg, "comp_daily_avg_usd": comp_avg,
            "change_usd_per_day": change, "change_pct": change / base_avg, "drivers": drivers}


@functools.lru_cache(maxsize=16)
def _hourly_series(db_key: str, scope: str) -> np.ndarray:
    con = duckdb.connect(db_key, read_only=True)
    try:
        if scope == "total":
            q, p = "SELECT cost_usd FROM mart_hourly_total ORDER BY hour_idx", []
        elif scope in CATEGORIES:
            q, p = "SELECT cost_usd FROM mart_hourly_category WHERE vm_category = ? ORDER BY hour_idx", [scope]
        elif re.fullmatch(r"SUB-\d{4}", scope):
            q, p = ("SELECT h.hour_idx, COALESCE(s.cost_usd, 0) AS cost_usd FROM range(0, 720) h(hour_idx) "
                    "LEFT JOIN mart_hourly_top_subscription s ON s.hour_idx = h.hour_idx AND s.subscription_alias = ? "
                    "ORDER BY 1", [scope])
            df = con.execute(q, p).df()
            if df.cost_usd.sum() == 0:
                raise ValueError(f"{scope} has no hourly series (hourly detail is kept for SUB-0001..SUB-0050)")
            return df.cost_usd.to_numpy()
        else:
            raise ValueError("scope must be 'total', a category, or a subscription alias like SUB-0007")
        return con.execute(q, p).df().cost_usd.to_numpy()
    finally:
        con.close()


def _db_key(con) -> str:
    return con.execute("SELECT file FROM pragma_database_list WHERE name = current_database()").fetchone()[0]


def detect_anomalies(con, scope: str = "subscriptions", top_n: int = 5) -> dict:
    """Run the RobustSeasonalDetector. scope = 'subscriptions' (top 50), 'total', a category, or SUB-xxxx."""
    det = A.RobustSeasonalDetector()
    key = _db_key(con)
    if scope == "subscriptions":
        names = [r[0] for r in con.execute(
            "SELECT DISTINCT subscription_alias FROM mart_hourly_top_subscription ORDER BY 1").fetchall()]
    else:
        names = [scope]
    events = []
    for n in names:
        y = _hourly_series(key, "total" if n == "total" else n)
        events += [e.as_dict() for e in det.detect(y, n)]
    events.sort(key=lambda e: (-abs(e["excess_cost_usd"]), e["series"], e["start_hour"]))
    out = [{"series": e["series"], "start_day": e["start_day"], "start_hour_of_day": e["start_hour"] % 24,
            "duration_hours": e["duration_hours"], "direction": e["direction"], "pattern": e["pattern"],
            "excess_cost_usd": e["excess_cost_usd"], "pct_vs_expected": e["pct_vs_expected"]}
           for e in events[: int(top_n)]]
    return {"scope": scope, "series_scanned": len(names), "events_found": len(events),
            "total_excess_usd": float(sum(e["excess_cost_usd"] for e in events if e["direction"] == "spike")),
            "top_events": out}


def forecast_spend(con, scope: str = "total", horizon_days: int = 7) -> dict:
    h = int(horizon_days)
    if not 1 <= h <= 14:
        raise ValueError("horizon_days must be between 1 and 14")
    y = _hourly_series(_db_key(con), scope)
    fc = F.forecast_with_intervals(y, h * 24, level=0.9, model=F.choose_model(scope, h * 24))
    last7 = float(y[-7 * 24:].sum()) / 7
    return {"scope": scope, "horizon_days": h, "forecast_days": [31, 30 + h],
            "forecast_total_usd": fc.total_point, "lower_90_usd": fc.total_lower, "upper_90_usd": fc.total_upper,
            "forecast_daily_avg_usd": fc.total_point / h, "last7_daily_avg_usd": last7,
            "change_vs_last7_pct": (fc.total_point / h - last7) / last7, "method": fc.method}


def optimization_opportunities(con, commit_discount: float = 0.0, spot_discount: float = 0.0) -> dict:
    r = O.recommendations(con, commit_discount=float(commit_discount), spot_discount=float(spot_discount))
    levers = [{"lever": row.lever, "vms": None if row.vms is None or np.isnan(row.vms) else int(row.vms),
               "addressable_cost_usd": float(row.addressable_cost_usd), "savings_usd": float(row.savings_usd),
               "pct_of_total_spend": float(row.pct_of_total_spend), "basis": row.basis}
              for row in r["summary"].itertuples()]
    top = [{"name": x.subscription_alias, "savings_usd": float(x.savings_usd)}
           for x in r["by_subscription"].head(5).itertuples()]
    return {"total_cost_usd": r["total_cost_usd"], "hard_savings_usd": r["hard_savings_usd"],
            "hard_savings_pct": r["hard_savings_usd"] / r["total_cost_usd"], "levers": levers,
            "top_subscriptions_by_savings": top,
            "note": "Hard savings = idle + rightsizing (no assumed discounts). Levers overlap; do not add Spot to rightsizing."}


def explain_change(con, subscription: str, base_start: int, base_end: int, comp_start: int, comp_end: int,
                   top_n: int = 3) -> dict:
    """Root cause of a team's spend change, decomposed exactly by deployment."""
    return RC.explain_change(con, subscription, base_start, base_end, comp_start, comp_end, top_n)


def budget_vs_actual(con, plan_start: int = 8, plan_end: int = 14, period_start: int = 15, period_end: int = 30,
                     dimension: str = "subscription", top_n: int = 5) -> dict:
    """Run-rate budget set on `plan_end` (avg daily spend of the plan window x period days) vs actual."""
    from cca.qbr import budget_vs_actual as bva
    for a, b in ((plan_start, plan_end), (period_start, period_end)):
        _days(a, b)
    df = bva(con, dimension, int(plan_start), int(plan_end), int(period_start), int(period_end))
    budget, actual = float(df.budget_usd.sum()), float(df.actual_usd.sum())
    top = [{"name": r.name, "budget_usd": float(r.budget_usd), "actual_usd": float(r.actual_usd),
            "variance_usd": float(r.variance_usd), "unbudgeted": bool(r.unbudgeted)}
           for r in df.head(int(top_n)).itertuples()]
    return {"plan_days": [int(plan_start), int(plan_end)], "period_days": [int(period_start), int(period_end)],
            "dimension": dimension, "budget_usd": budget, "actual_usd": actual,
            "variance_usd": actual - budget, "variance_pct": (actual - budget) / budget,
            "unbudgeted_usd": float(df.loc[df.unbudgeted, "actual_usd"].sum()), "top": top}


# --------------------------------------------- fact posting per tool result
def _post_facts(ledger: Ledger, tool: str, args: dict, res: dict) -> None:
    P = lambda label, key, unit: ledger.post(label, get_path(res, key), unit, tool, args, key)  # noqa: E731
    if tool == "describe_dataset":
        P("Total 30-day spend (all subscriptions)", "total_usd", "usd")
        P("Average daily spend", "daily_avg_usd", "usd")
        P("Number of subscriptions", "subscriptions", "count")
        P("Number of VMs", "vms", "count")
    elif tool == "spend_summary":
        w = f"days {res['start_day']}-{res['end_day']}"
        P(f"Total spend, {w}", "total_usd", "usd")
        P(f"Average daily spend, {w}", "daily_avg_usd", "usd")
        for i, t in enumerate(res["top"]):
            P(f"Spend of {t['name']}, {w}", f"top.{i}.cost_usd", "usd")
            P(f"Share of spend for {t['name']}, {w}", f"top.{i}.share", "pct")
    elif tool == "compare_periods":
        b, c = res["base_days"], res["comp_days"]
        P(f"Daily avg spend, days {b[0]}-{b[1]} (base)", "base_daily_avg_usd", "usd")
        P(f"Daily avg spend, days {c[0]}-{c[1]} (comparison)", "comp_daily_avg_usd", "usd")
        P("Change in daily avg spend (comparison - base)", "change_usd_per_day", "usd")
        P("Change in daily avg spend, %", "change_pct", "pct")
        for i, d in enumerate(res["drivers"]):
            P(f"Change in daily avg spend from {d['name']}", f"drivers.{i}.change_usd_per_day", "usd")
            P(f"{d['name']} share of total change", f"drivers.{i}.share_of_change", "pct")
    elif tool == "detect_anomalies":
        P(f"Anomaly events found ({res['scope']})", "events_found", "count")
        P(f"Total excess spend in spike events ({res['scope']})", "total_excess_usd", "usd")
        for i, e in enumerate(res["top_events"]):
            tag = f"{e['series']} {e['direction']} starting day {e['start_day']}"
            P(f"Excess cost vs expected: {tag}", f"top_events.{i}.excess_cost_usd", "usd")
            P(f"Deviation vs expected: {tag}", f"top_events.{i}.pct_vs_expected", "pct")
            P(f"Duration: {tag}", f"top_events.{i}.duration_hours", "hours")
            P(f"Start day: {tag}", f"top_events.{i}.start_day", "day")
    elif tool == "forecast_spend":
        w = f"{res['scope']}, next {res['horizon_days']} days"
        P(f"Forecast spend, {w}", "forecast_total_usd", "usd")
        P(f"Forecast 90% lower bound, {w}", "lower_90_usd", "usd")
        P(f"Forecast 90% upper bound, {w}", "upper_90_usd", "usd")
        P(f"Forecast daily average, {w}", "forecast_daily_avg_usd", "usd")
        P(f"Actual daily average, last 7 days ({res['scope']})", "last7_daily_avg_usd", "usd")
        P(f"Forecast vs last-7-day daily average, % ({res['scope']})", "change_vs_last7_pct", "pct")
    elif tool == "budget_vs_actual":
        P("Budget, review period", "budget_usd", "usd")
        P("Actual spend, review period", "actual_usd", "usd")
        P("Variance vs budget", "variance_usd", "usd")
        P("Variance vs budget, %", "variance_pct", "pct")
        P("Unbudgeted spend (teams with no budget)", "unbudgeted_usd", "usd")
        for i, t in enumerate(res["top"]):
            P(f"Budget: {t['name']}", f"top.{i}.budget_usd", "usd")
            P(f"Actual spend: {t['name']}", f"top.{i}.actual_usd", "usd")
            P(f"Variance vs budget: {t['name']}", f"top.{i}.variance_usd", "usd")
    elif tool == "explain_change":
        sub, b, c = res["subscription"], res["base_days"], res["comp_days"]
        P(f"{sub} daily avg spend, days {b[0]}-{b[1]} (base)", "base_daily_avg_usd", "usd")
        P(f"{sub} daily avg spend, days {c[0]}-{c[1]} (comparison)", "comp_daily_avg_usd", "usd")
        P(f"Change in {sub} daily avg spend (comparison - base)", "change_usd_per_day", "usd")
        if res["change_pct"] is not None:
            P(f"Change in {sub} daily avg spend, %", "change_pct", "pct")
        P(f"Deployments in {sub} whose spend changed", "deployments_changed", "count")
        P(f"Share of {sub} gross change explained by the top deployment", "top_driver_share_of_gross_change", "pct")
        for i, d in enumerate(res["drivers"]):
            P(f"Change in daily avg spend from {d['name']} ({d['kind']})", f"drivers.{i}.change_usd_per_day", "usd")
            P(f"{d['name']} share of {sub} net change", f"drivers.{i}.share_of_change", "pct")
            P(f"Total VMs in {d['name']}", f"drivers.{i}.vm_count", "count")
            P(f"VMs {d['name']} launched in days {c[0]}-{c[1]}", f"drivers.{i}.vms_created_in_comparison", "count")
    elif tool == "optimization_opportunities":
        P("Total 30-day spend", "total_cost_usd", "usd")
        P("Hard savings (idle + rightsizing), 30 days", "hard_savings_usd", "usd")
        P("Hard savings as % of spend", "hard_savings_pct", "pct")
        for i, lv in enumerate(res["levers"]):
            P(f"{lv['lever']}: addressable spend", f"levers.{i}.addressable_cost_usd", "usd")
            P(f"{lv['lever']}: savings", f"levers.{i}.savings_usd", "usd")
            if lv["vms"] is not None:
                P(f"{lv['lever']}: VMs", f"levers.{i}.vms", "count")
        for i, t in enumerate(res["top_subscriptions_by_savings"]):
            P(f"Hard savings available in {t['name']}", f"top_subscriptions_by_savings.{i}.savings_usd", "usd")


TOOLS: dict[str, Callable] = {
    "describe_dataset": describe_dataset,
    "spend_summary": spend_summary,
    "compare_periods": compare_periods,
    "detect_anomalies": detect_anomalies,
    "forecast_spend": forecast_spend,
    "optimization_opportunities": optimization_opportunities,
    "explain_change": explain_change,
    "budget_vs_actual": budget_vs_actual,
}

TOOL_SPECS = """
describe_dataset()  -- size of the dataset and total spend.
spend_summary(start_day=1, end_day=30, dimension="subscription"|"category"|"shape", top_n=5)
    -- total and daily-average spend for a window, plus top contributors.
compare_periods(base_start, base_end, comp_start, comp_end, dimension="subscription"|"category"|"shape", top_n=5)
    -- variance analysis on daily-average spend; drivers' contributions sum to the total change.
detect_anomalies(scope="subscriptions"|"total"|"Delay-insensitive"|"Interactive"|"Unknown"|"SUB-0001".."SUB-0050", top_n=5)
    -- robust seasonal anomaly detector; returns events ranked by $ impact.
forecast_spend(scope="total"|category|"SUB-xxxx", horizon_days=7)
    -- forecast beyond day 30 with 90% conformal intervals.
optimization_opportunities(commit_discount=0.0, spot_discount=0.0)
    -- idle, rightsizing, commitment and Spot levers with savings.
explain_change(subscription="SUB-0041", base_start, base_end, comp_start, comp_end, top_n=3)
    -- ROOT CAUSE: decomposes one team's spend change exactly by deployment (launched / shut down /
       scaled up / scaled down) and flags likely migrations. Use for "why did SUB-xxxx change?".
budget_vs_actual(plan_start=8, plan_end=14, period_start=15, period_end=30, dimension="subscription"|"category", top_n=5)
    -- run-rate budget (set on plan_end from the plan window's daily average) vs actual, total and by team/category.
compute(expression, label, unit="usd"|"pct"|"count")
    -- arithmetic over existing fact ids, e.g. expression="(F7 - F6) / F6". Use it for ANY derived number.
""".strip()

_SAFE_EXPR = re.compile(r"^[\sF0-9+\-*/().]+$")


def eval_expression(expression: str, values: dict[str, float]) -> float:
    if not _SAFE_EXPR.match(expression):
        raise ValueError("expression may only contain fact ids, numbers and + - * / ( )")
    expr = re.sub(r"F(\d+)", lambda m: repr(values[f"F{m.group(1)}"]), expression)
    return float(eval(expr, {"__builtins__": {}}, {}))  # noqa: S307 -- input validated by regex above


def run_tool(con, ledger: Ledger, tool: str, args: dict | None = None) -> dict:
    """Execute a tool, post its facts, and return its result (with the new fact ids)."""
    args = dict(args or {})
    before = set(ledger.facts)
    if tool == "compute":
        values = {k: f.value for k, f in ledger.facts.items()}
        unit = args.get("unit", "usd")
        val = eval_expression(args["expression"], values)
        ledger.post(args.get("label", args["expression"]), val, unit, "compute",
                    {"expression": args["expression"]}, "value", formula=args["expression"])
        res = {"value": val}
    else:
        if tool not in TOOLS:
            raise ValueError(f"unknown tool {tool}")
        res = TOOLS[tool](con, **args)
        _post_facts(ledger, tool, args, res)
    res["_facts"] = [k for k in ledger.facts if k not in before]
    return res
