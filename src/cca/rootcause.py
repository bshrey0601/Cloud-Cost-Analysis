"""Root-cause attribution: which deployment caused a team's spend to change?

A "deployment" in Microsoft's trace is a group of VMs launched together -- in practice a
service, a test environment or a batch fleet. When a subscription's (team's) daily spend
moves, we decompose the change EXACTLY by deployment (the per-deployment changes sum to the
team's change, to the cent) and classify each driver:

    launched   -- no spend in the base window, spend in the comparison window
    shut down  -- spend in the base window, none in the comparison window
    scaled up / scaled down -- present in both, spend changed

This is the question Finance asks Engineering every quarter: *what changed, and who owns it?*
"""

from __future__ import annotations

import duckdb
import pandas as pd


def _window(a: int, b: int) -> tuple[int, int]:
    a, b = int(a), int(b)
    if not 1 <= a <= b <= 30:
        raise ValueError("days must satisfy 1 <= start <= end <= 30")
    return a, b


def explain_change(con: duckdb.DuckDBPyConnection, subscription: str, base_start: int, base_end: int,
                   comp_start: int, comp_end: int, top_n: int = 5) -> dict:
    bs, be = _window(base_start, base_end)
    cs, ce = _window(comp_start, comp_end)
    nb, nc = be - bs + 1, ce - cs + 1
    df = con.execute("""
        SELECT m.deployment_alias AS name,
               SUM(CASE WHEN trace_day BETWEEN ? AND ? THEN cost_usd ELSE 0 END) / ? AS base_avg,
               SUM(CASE WHEN trace_day BETWEEN ? AND ? THEN cost_usd ELSE 0 END) / ? AS comp_avg,
               SUM(CASE WHEN trace_day BETWEEN ? AND ? THEN vms_created ELSE 0 END)   AS vms_created_comp,
               ANY_VALUE(d.vm_count) AS vm_count, ANY_VALUE(d.first_day) AS first_day,
               ANY_VALUE(d.last_day) AS last_day, ANY_VALUE(d.primary_category) AS category,
               ANY_VALUE(d.primary_sku) AS sku
        FROM mart_daily_deployment m
        JOIN dim_deployment d USING (deployment_alias)
        WHERE m.subscription_alias = ?
        GROUP BY 1""", [bs, be, nb, cs, ce, nc, cs, ce, subscription]).df()
    if df.empty:
        raise ValueError(f"no deployments found for {subscription}")
    df["change"] = df.comp_avg - df.base_avg
    base_avg, comp_avg = float(df.base_avg.sum()), float(df.comp_avg.sum())
    change = comp_avg - base_avg

    def kind(r) -> str:
        if r.base_avg <= 1e-9 and r.comp_avg > 0:
            return "launched"
        if r.comp_avg <= 1e-9 and r.base_avg > 0:
            return "shut down"
        return "scaled up" if r.change >= 0 else "scaled down"

    df["kind"] = df.apply(kind, axis=1)
    df["abs_change"] = df.change.abs()
    gross = float(df.abs_change.sum())
    df = df.sort_values(["abs_change", "name"], ascending=[False, True])
    movers = df[df.abs_change >= 0.005]          # ignore deployments whose spend did not move
    drivers = [{
        "name": r.name, "kind": r.kind, "category": r.category, "sku": sku_label(r.sku),
        "first_day": int(r.first_day), "last_day": int(r.last_day), "vm_count": int(r.vm_count),
        "vms_created_in_comparison": int(r.vms_created_comp),
        "base_daily_avg_usd": float(r.base_avg), "comp_daily_avg_usd": float(r.comp_avg),
        "change_usd_per_day": float(r.change),
        "share_of_change": float(r.change / change) if change else 0.0,
        # share of the team's total movement (sum of |changes|): always 0-100%, easy to read
        "share_of_gross_change": float(r.abs_change / gross) if gross else 0.0,
    } for r in movers.head(int(top_n)).itertuples()]
    top_share = float(df.abs_change.iloc[0] / gross) if gross else 0.0
    return {
        "possible_migrations": find_migrations(df),
        "subscription": subscription, "base_days": [bs, be], "comp_days": [cs, ce],
        "base_daily_avg_usd": base_avg, "comp_daily_avg_usd": comp_avg,
        "change_usd_per_day": change, "change_pct": change / base_avg if base_avg else None,
        "deployments_changed": int((df.abs_change > 0.01).sum()),
        "top_driver_share_of_gross_change": top_share,
        "drivers": drivers,
    }


def sku_label(sku: str) -> str:
    """'custom_2vcpu_2gb' (a trace shape with no exact Azure SKU) -> '2 vCPU / 2 GB VMs'."""
    import re
    m = re.fullmatch(r"custom_(\d+)vcpu_(\d+)gb", sku or "")
    return f"{m.group(1)} vCPU / {m.group(2)} GB VMs" if m else sku


def find_migrations(df: pd.DataFrame, tolerance: float = 0.35, min_usd_per_day: float = 50.0) -> list[dict]:
    """Pair a shrinking deployment with a growing one of similar size in the same team:
    the signature of a migration (blue-green deploy, version upgrade, region move).
    A heuristic flag for a human to confirm, not a fact."""
    ups = df[(df.change >= min_usd_per_day)].sort_values("change", ascending=False)
    downs = df[(df.change <= -min_usd_per_day)].sort_values("change")
    pairs, used = [], set()
    for d in downs.itertuples():
        for u in ups.itertuples():
            if u.name in used:
                continue
            if abs(u.change + d.change) <= tolerance * max(u.change, -d.change):
                pairs.append({"from": d.name, "to": u.name, "from_change_usd_per_day": float(d.change),
                              "to_change_usd_per_day": float(u.change)})
                used.add(u.name)
                break
    return pairs


def event_windows(start_day: int) -> tuple[int, int, int, int] | None:
    """Base = up to 7 days before the event; comparison = the event day and the next two."""
    if start_day <= 1:
        return None
    return max(1, start_day - 7), start_day - 1, start_day, min(30, start_day + 2)


def explain_event(con: duckdb.DuckDBPyConnection, subscription: str, start_day: int, top_n: int = 3) -> dict | None:
    w = event_windows(int(start_day))
    if w is None:
        return None
    return explain_change(con, subscription, *w, top_n=top_n)


def attribution_coverage(con: duckdb.DuckDBPyConnection, events: pd.DataFrame) -> dict:
    """How often a single deployment explains most of a detected anomaly (a measure of how
    actionable the alerts are: one owner to call, not twenty)."""
    rows = []
    for e in events.itertuples():
        r = explain_event(con, e.series, int(e.start_hour) // 24 + 1)
        if r is None or r["change_usd_per_day"] == 0:
            continue
        rows.append({"series": e.series, "start_day": int(e.start_hour) // 24 + 1,
                     "top_driver": r["drivers"][0]["name"], "top_kind": r["drivers"][0]["kind"],
                     "top_share": r["top_driver_share_of_gross_change"],
                     "deployments_changed": r["deployments_changed"]})
    df = pd.DataFrame(rows)
    return {
        "events_explained": len(df),
        "share_with_one_deployment_explaining_majority": float((df.top_share >= 0.5).mean()),
        "median_top_driver_share": float(df.top_share.median()),
        "kinds_of_top_driver": df.top_kind.value_counts().to_dict(),
        "examples": df.sort_values("top_share", ascending=False).head(10).to_dict("records"),
    }
