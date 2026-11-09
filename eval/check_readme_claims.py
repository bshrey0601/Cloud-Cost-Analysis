"""Check every headline number in the README against freshly generated result files.

    python eval/check_readme_claims.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from cca import optimization as O  # noqa: E402
from cca.warehouse import connect  # noqa: E402

R = ROOT / "eval" / "results"


def main() -> bool:
    con = connect()
    fs = pd.read_csv(R / "forecast_summary.csv")
    t = fs[fs.series == "Total"].set_index(["horizon_hours", "model"]).daily_total_mape
    ours, hw = "CloudSpendForecaster (ours, self-selecting blend)", "Holt-Winters (statsmodels)"
    cov = json.loads((R / "forecast_interval_coverage.json").read_text())["Total"]
    an = json.loads((R / "anomaly_eval.json").read_text())["detectors"]
    fi = json.loads((R / "auditor_fault_injection.json").read_text())
    bm = json.loads((R / "benchmark_offline.json").read_text())["summary"]
    opt = O.recommendations(con)
    total = con.execute("SELECT SUM(cost_usd) FROM mart_hourly_total").fetchone()[0]
    recon = [r[0] for r in con.execute("SELECT status FROM dq_reconciliation").fetchall()]
    rz = an["Rolling 24h z-score (k=3)"]["clean_alarms_per_series_week"]
    us = an["RobustSeasonalDetector (ours)"]
    faulty = {k: v for k, v in fi["by_fault"].items() if not k.startswith("none")}

    claims = [
        ("30-day spend $21.23M", round(total / 1e6, 2), 21.23),
        ("Hard savings $3.75M", round(opt["hard_savings_usd"] / 1e6, 2), 3.75),
        ("Hard savings 17.7% of spend", round(opt["hard_savings_usd"] / total * 100, 1), 17.7),
        ("Reconciliation: 8 checks, all PASS", (len(recon), set(recon)), (8, {"PASS"})),
        ("Next-day error, ours 0.86%", round(t[(24, ours)] * 100, 2), 0.86),
        ("~Next-day error, Holt-Winters 1.04%", round(t[(24, hw)] * 100, 2), 1.04),
        ("Next-week error, ours 0.91%", round(t[(168, ours)] * 100, 2), 0.91),
        ("~Next-week error, Holt-Winters 2.06%", round(t[(168, hw)] * 100, 2), 2.06),
        ("Next-week: 56% lower than Holt-Winters", round((1 - t[(168, ours)] / t[(168, hw)]) * 100), 56),
        ("Next-week: last-week naive best at 0.64%",
         round(t[(168, "Seasonal naive (same hour last week)")] * 100, 2), 0.64),
        ("90% interval coverage 90.3%", round(cov["hourly_coverage"] * 100, 1), 90.3),
        ("Anomaly F1 0.946", round(us["f1"], 3), 0.946),
        ("Anomaly precision 0.983", round(us["precision"], 3), 0.983),
        ("17.6x fewer false alarms", round(rz / us["clean_alarms_per_series_week"], 1), 17.6),
        ("252 injected anomaly events", us["injected_events"], 252),
        ("Auditor catches 99.8%", round(fi["overall_detection_rate"] * 100, 1), 99.8),
        ("Auditor: 531 faulty memos", sum(v["cases"] for v in faulty.values()), 531),
        ("Auditor: 0 false positives on clean memos", fi["by_fault"]["none (clean memo)"]["detected"], 0),
        ("Benchmark 30/30 correct", (bm["questions"], bm["accuracy"]), (30, 1.0)),
    ]
    un = json.loads((R / "untouched_forecast_check.json").read_text())["horizons"]
    p24, p168 = un["24h"]["pooled_period_total_error"], un["168h"]["pooled_period_total_error"]
    claims += [
        ("Untouched next-day: ours 1.31%", round(p24[ours] * 100, 2), 1.31),
        ("Untouched next-day: Holt-Winters 1.62%", round(p24[hw] * 100, 2), 1.62),
        ("Untouched next-day: 19% lower than HW", round((1 - p24[ours] / p24[hw]) * 100), 19),
        ("Untouched next-day: last value best on 27/50", un["24h"]["series_where_model_is_best"]["Naive (last value)"], 27),
        ("Untouched next-week: ours 2.76%", round(p168[ours] * 100, 2), 2.76),
        ("Untouched next-week: Holt-Winters 2.20%", round(p168[hw] * 100, 2), 2.2),
        ("Untouched next-week: yesterday naive 1.97%", round(p168["Seasonal naive (same hour yesterday)"] * 100, 2), 1.97),
        ("Untouched next-week: ours worst of five", max(p168, key=p168.get), ours),
    ]
    rc = json.loads((R / "root_cause_eval.json").read_text())
    tt = json.loads((R / "team_forecast_test.json").read_text())
    tu = json.loads((R / "team_forecast_tuning.json").read_text())["error"]
    te = tt["test_error_pct"]
    v1, v2 = "v1 blend (no step awareness)", "v2 step-aware (threshold 15%)"
    sny = "Seasonal naive (same hour yesterday)"
    claims += [
        ("Root cause: 34 anomalies explained", rc["events_explained"], 34),
        ("Root cause: one deployment explains majority in 56%",
         round(rc["share_with_one_deployment_explaining_majority"] * 100), 56),
        ("Root cause: decomposition error < 1e-12", rc["max_decomposition_error_usd_per_day"] < 1e-12, True),
        ("Team test: v1 next week 4.47%", round(te["next_week"][v1], 2), 4.47),
        ("Team test: v2 next week 3.55%", round(te["next_week"][v2], 2), 3.55),
        ("Team test: v2 21% better than v1 week-ahead", round((1 - te["next_week"][v2] / te["next_week"][v1]) * 100), 21),
        ("Team test: yesterday naive next week 2.19%", round(te["next_week"][sny], 2), 2.19),
        ("Team test: v1 next day 1.53%", round(te["next_day"][v1], 2), 1.53),
        ("Team test: v2 next day 1.78%", round(te["next_day"][v2], 2), 1.78),
        ("Team tuning: v2 next week 1.01%", round(tu["next_week"][v2], 2), 1.01),
        ("Team tuning: v1 next week 1.10%", round(tu["next_week"][v1], 2), 1.1),
    ]
    from cca.qbr import build_qbr
    from cca.warehouse import active_db_path
    q = build_qbr(con, str(active_db_path()))
    top = q.team_drivers[0]
    claims += [
        ("QBR: commentary 13/13 audited", (q.audit.passed, q.audit.numbers_checked, q.audit.facts_reconciled), (True, 13, 13)),
        ("QBR: 0.6% over plan", round(q.total["variance_pct"] * 100, 1), 0.6),
        ("QBR: SUB-0058 +$58.3K via DEP-00049 scaled up",
         (top["team"], round(top["variance_usd"] / 1e3, 1), top["top_deployment"], top["what_happened"]),
         ("SUB-0058", 58.3, "DEP-00049", "scaled up")),
    ]
    ok = True
    for name, actual, expected in claims:
        # "~" marks the Holt-Winters baseline: statsmodels' optimizer varies in the last decimal between runs
        good = abs(actual - expected) <= 0.0101 if name.startswith("~") else actual == expected
        ok &= good
        print(f"{'OK  ' if good else 'DIFF'} {name:45s} actual={actual}  readme={expected}")
    print("\nALL README CLAIMS MATCH" if ok else "\nSOME CLAIMS DIFFER -- update README")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
