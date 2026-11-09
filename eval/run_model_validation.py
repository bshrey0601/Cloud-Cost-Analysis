"""Reproducible validation of the forecasting and anomaly models on the real Azure data.

    python eval/run_model_validation.py

Writes eval/results/forecast_backtest.csv, forecast_summary.csv,
anomaly_eval.json and real_anomalies.csv.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from cca import anomaly as A  # noqa: E402
from cca import forecasting as F  # noqa: E402
from cca.warehouse import connect  # noqa: E402

OUT = ROOT / "eval" / "results"
OUT.mkdir(parents=True, exist_ok=True)


def load_series(con) -> tuple[dict, dict]:
    totals = {"Total": con.execute("SELECT cost_usd FROM mart_hourly_total ORDER BY hour_idx").df().cost_usd.values}
    for cat in ["Delay-insensitive", "Interactive", "Unknown"]:
        totals[cat] = con.execute(
            "SELECT cost_usd FROM mart_hourly_category WHERE vm_category = ? ORDER BY hour_idx", [cat]
        ).df().cost_usd.values
    df = con.execute("SELECT subscription_alias s, hour_idx, cost_usd FROM mart_hourly_top_subscription").df()
    piv = df.pivot(index="hour_idx", columns="s", values="cost_usd").reindex(range(720)).fillna(0.0)
    subs = {c: piv[c].values for c in piv.columns}
    return totals, subs


def validate_forecasting(totals: dict) -> pd.DataFrame:
    """Rolling-origin backtests at two horizons: next day (24 h) and next week (168 h)."""
    frames = []
    for horizon in (24, 168):
        for name, y in totals.items():
            bt = F.rolling_backtest(y, first_origin_day=15, horizon=horizon)
            bt["series"] = name
            bt["horizon_hours"] = horizon
            frames.append(bt)
    bt = pd.concat(frames)
    bt.to_csv(OUT / "forecast_backtest.csv", index=False)
    summary = (bt.groupby(["horizon_hours", "series", "model"])
                 .agg(folds=("origin_day", "count"), hourly_wape=("hourly_wape", "mean"),
                      daily_total_mape=("total_pct_error", lambda e: float(np.mean(np.abs(e)))))
                 .reset_index())
    summary.to_csv(OUT / "forecast_summary.csv", index=False)
    cov = {name: F.interval_coverage(y, level=0.9) for name, y in totals.items()}
    (OUT / "forecast_interval_coverage.json").write_text(json.dumps(cov, indent=2))
    return summary


def validate_anomaly(subs: dict, totals: dict) -> dict:
    names = sorted(subs)
    tune = {n: subs[n] for n in names[0::2]}   # used only to choose thresholds
    test = {n: subs[n] for n in names[1::2]}   # held out
    test.update({f"category:{k}": v for k, v in totals.items() if k != "Total"})
    detectors = {
        "RobustSeasonalDetector (ours)": A.RobustSeasonalDetector(),
        "Rolling 24h z-score (k=3)": A.RollingZScoreDetector(k=3.0),
        "Static weekly threshold (mean+3sd)": A.StaticThresholdDetector(k=3.0),
    }
    results = {"tuning_series": len(tune), "test_series": len(test), "seeds": [11, 12, 13], "detectors": {}}
    for name, det in detectors.items():
        runs = [A.evaluate_detector(det, test, seed=s) for s in results["seeds"]]
        agg = {k: float(np.mean([r[k] for r in runs])) for k in
               ["recall", "precision", "f1", "clean_alarms_per_series_week"]}
        agg["recall_by_type"] = {t: float(np.mean([r["recall_by_type"][t] for r in runs])) for t in A.ANOMALY_TYPES}
        agg["recall_by_magnitude"] = {
            b: float(np.mean([r["recall_by_magnitude"][b] for r in runs if r["recall_by_magnitude"][b] is not None]))
            for b in ["<15%", "15-30%", ">30%"]}
        agg["injected_events"] = int(sum(r["injected_events"] for r in runs))
        results["detectors"][name] = agg
    (OUT / "anomaly_eval.json").write_text(json.dumps(results, indent=2))
    return results


def real_anomalies(subs: dict) -> pd.DataFrame:
    det = A.RobustSeasonalDetector()
    events = [e.as_dict() for n, y in subs.items() for e in det.detect(y, n)]
    df = pd.DataFrame(events).sort_values("excess_cost_usd", key=np.abs, ascending=False)
    df.to_csv(OUT / "real_anomalies.csv", index=False)
    return df


if __name__ == "__main__":
    con = connect()
    totals, subs = load_series(con)
    print("Forecasting backtest ...")
    s = validate_forecasting(totals)
    pd.set_option("display.width", 250)
    for h in (24, 168):
        print(f"horizon {h}h  hourly WAPE %")
        print((s[s.horizon_hours == h].pivot(index="model", columns="series", values="hourly_wape") * 100).round(3).to_string())
        print(f"horizon {h}h  period-total MAPE %")
        print((s[s.horizon_hours == h].pivot(index="model", columns="series", values="daily_total_mape") * 100).round(3).to_string())
    print(json.loads((OUT / "forecast_interval_coverage.json").read_text())["Total"])
    print("Anomaly detection (held-out) ...")
    r = validate_anomaly(subs, totals)
    for name, m in r["detectors"].items():
        print(f"  {name:40s} F1={m['f1']:.3f} P={m['precision']:.3f} R={m['recall']:.3f} "
              f"clean-alarms/series-week={m['clean_alarms_per_series_week']:.2f} by_mag={m['recall_by_magnitude']}")
    print("Real anomalies found:")
    print(real_anomalies(subs).head(10).round(1).to_string(index=False))
