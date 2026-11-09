"""Optimisation recommendations, computed in SQL on VM-level utilisation.

All savings are for the 30-day trace window (≈ one month of run-rate).

1. Idle VMs           avg CPU < 1% and P95 CPU < 5%, running >= 72 h
                      -> shut down or schedule off; saving = 100% of its cost (needs owner review).
2. Rightsizing        not idle, >= 4 vCPU, P95 of max CPU < 40%
                      -> move one size down in the same family. Azure v5 list prices scale
                         linearly with size (D2s/D4s/D8s_v5 = $0.096/$0.192/$0.384), so one
                         size down halves the price; P95 < 40% keeps post-resize P95 < 80%.
                         Caveat: the trace has no memory utilisation, so memory fit is unverified.
3. Commitment cover   the always-on floor of spend (5th percentile of hourly cost over the
                      last 14 days x 720 h), MINUS the idle + rightsizing savings ("optimise
                      first, then commit"), is eligible for a 1- or 3-year commitment. The discount is
                      an input (`commit_discount`), not a fact, because it depends on contract.
4. Spot eligibility   Delay-insensitive VMs (Microsoft's own label) that are not idle are
                      candidates for interruptible/Spot capacity; discount is also an input.
"""

from __future__ import annotations

import duckdb
import pandas as pd

IDLE_SQL = """
SELECT subscription_alias, vm_category, shape_id, COUNT(*) AS vms, SUM(cost_usd) AS cost_usd
FROM mart_vm_utilization
WHERE avg_cpu < 1 AND p95_max_cpu < 5 AND lifetime_hours >= 72
GROUP BY ALL
"""

RIGHTSIZE_SQL = """
SELECT subscription_alias, vm_category, shape_id, COUNT(*) AS vms,
       SUM(cost_usd) AS cost_usd, SUM(cost_usd) * 0.5 AS savings_usd
FROM mart_vm_utilization
WHERE vcpu >= 4 AND p95_max_cpu < 40
  AND NOT (avg_cpu < 1 AND p95_max_cpu < 5 AND lifetime_hours >= 72)
GROUP BY ALL
"""

SPOT_SQL = """
SELECT subscription_alias, COUNT(*) AS vms, SUM(cost_usd) AS cost_usd
FROM mart_vm_utilization
WHERE vm_category = 'Delay-insensitive'
  AND NOT (avg_cpu < 1 AND p95_max_cpu < 5 AND lifetime_hours >= 72)
GROUP BY ALL
"""

COMMIT_SQL = """
WITH last14 AS (SELECT cost_usd FROM mart_hourly_total WHERE trace_day > 16)
SELECT QUANTILE_CONT(cost_usd, 0.05) AS baseline_hourly_usd,
       (SELECT SUM(cost_usd) FROM mart_hourly_total) AS total_cost_usd
FROM last14
"""


def recommendations(con: duckdb.DuckDBPyConnection, commit_discount: float = 0.0,
                    spot_discount: float = 0.0) -> dict:
    idle = con.execute(IDLE_SQL).df()
    rs = con.execute(RIGHTSIZE_SQL).df()
    spot = con.execute(SPOT_SQL).df()
    base, total = con.execute(COMMIT_SQL).fetchone()
    # "Optimise first, then commit": never commit to spend the other levers will remove.
    removable = float(idle.cost_usd.sum() + rs.savings_usd.sum())
    commit_eligible = max(0.0, base * 720 - removable)
    summary = pd.DataFrame([
        {"lever": "Shut down idle VMs", "vms": int(idle.vms.sum()),
         "addressable_cost_usd": float(idle.cost_usd.sum()), "savings_usd": float(idle.cost_usd.sum()),
         "basis": "100% of idle VM cost; requires owner sign-off"},
        {"lever": "Rightsize over-provisioned VMs", "vms": int(rs.vms.sum()),
         "addressable_cost_usd": float(rs.cost_usd.sum()), "savings_usd": float(rs.savings_usd.sum()),
         "basis": "one size down halves list price (Azure v5 linear pricing)"},
        {"lever": "Commit the always-on baseline", "vms": None,
         "addressable_cost_usd": float(commit_eligible), "savings_usd": float(commit_eligible * commit_discount),
         "basis": f"5th-pct hourly floor x 720 h, minus idle + rightsizing savings; discount input = {commit_discount:.0%}"},
        {"lever": "Move delay-insensitive work to Spot", "vms": int(spot.vms.sum()),
         "addressable_cost_usd": float(spot.cost_usd.sum()), "savings_usd": float(spot.cost_usd.sum() * spot_discount),
         "basis": f"Microsoft's Delay-insensitive label; discount input = {spot_discount:.0%}"},
    ])
    summary["pct_of_total_spend"] = summary.addressable_cost_usd / total
    by_sub = (rs.groupby("subscription_alias").savings_usd.sum()
                .add(idle.groupby("subscription_alias").cost_usd.sum(), fill_value=0)
                .sort_values(ascending=False).rename("savings_usd").reset_index())
    hard = float(summary.savings_usd.iloc[:2].sum())  # idle + rightsizing: no assumed discounts
    return {"summary": summary, "hard_savings_usd": hard, "by_subscription": by_sub, "total_cost_usd": float(total),
            "idle_detail": idle, "rightsize_detail": rs}
