"""Fixing week-ahead forecasts for individual teams -- with an honest tuning/test split.

The 50 team series (top-50 subscriptions) were the "untouched" test set in the last round,
which showed v1 failing week-ahead. They have now been looked at once, so this round splits
them: odd-ranked teams are used to choose the design (TUNE), even-ranked teams are only
scored at the end (TEST). The four development series (total + 3 categories) are re-run to
check the change does not hurt where v1 was already good.

    python eval/run_team_forecast_experiment.py tune   # compare variants on the tuning half
    python eval/run_team_forecast_experiment.py test   # score the chosen model on the test half
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

from cca import forecasting as F  # noqa: E402
from cca.warehouse import connect  # noqa: E402

OUT = ROOT / "eval" / "results"


def variants() -> dict:
    C = F.CloudSpendForecaster
    return {
        "v1 blend (no step awareness)": lambda y, h: C(horizon=h, step_aware=False).fit(y).predict(h),
        "v2 4-way blend, no override":
            lambda y, h: C(horizon=h, step_aware=True, step_override=False).fit(y).predict(h),
        "v2 step-aware (threshold 10%)":
            lambda y, h: C(horizon=h, step_aware=True, step_threshold=0.10).fit(y).predict(h),
        "v2 step-aware (threshold 15%)":
            lambda y, h: C(horizon=h, step_aware=True, step_threshold=0.15).fit(y).predict(h),
        "v2 step-aware (threshold 25%)":
            lambda y, h: C(horizon=h, step_aware=True, step_threshold=0.25).fit(y).predict(h),
        "RecentLevel alone": lambda y, h: F.RecentLevel().fit(y).predict(h),
        "Holt-Winters (statsmodels)": F.holt_winters,
        "Seasonal naive (same hour yesterday)": lambda y, h: F.seasonal_naive(y, h, F.DAY),
        "Naive (last value)": F.naive,
    }


def team_series(con) -> dict:
    df = con.execute("SELECT subscription_alias s, hour_idx, cost_usd FROM mart_hourly_top_subscription").df()
    piv = df.pivot(index="hour_idx", columns="s", values="cost_usd").reindex(range(720)).fillna(0.0)
    return {c: piv[c].to_numpy() for c in sorted(piv.columns) if piv[c].to_numpy()[: 15 * 24].sum() > 0}


def score(series: dict, models: dict) -> pd.DataFrame:
    rows = []
    for h in (24, 168):
        for name, y in series.items():
            bt = F.rolling_backtest(y, first_origin_day=15, horizon=h, models=models)
            bt["series"], bt["horizon"] = name, h
            rows.append(bt)
    bt = pd.concat(rows)
    bt["abs_err"] = (bt.total_pred - bt.total_actual).abs()
    g = bt.groupby(["horizon", "model"])
    return (g.abs_err.sum() / g.total_actual.sum()).unstack(0).rename(columns={24: "next_day", 168: "next_week"})


def main(stage: str) -> None:
    con = connect()
    teams = team_series(con)
    names = sorted(teams)
    tune = {n: teams[n] for n in names[0::2]}
    test = {n: teams[n] for n in names[1::2]}
    if stage == "tune":
        res = score(tune, variants())
        print(f"TUNING half ({len(tune)} teams) -- dollar-weighted error on the period total, %")
        print((res * 100).round(3).sort_values("next_week").to_string())
        (OUT / "team_forecast_tuning.json").write_text(json.dumps({
            "teams": list(tune), "error": (res * 100).round(4).to_dict()}, indent=2))
    else:
        chosen = {k: v for k, v in variants().items() if k in (
            "v1 blend (no step awareness)", "v2 step-aware (threshold 15%)", "Holt-Winters (statsmodels)",
            "Seasonal naive (same hour yesterday)", "Naive (last value)")}
        res = score(test, chosen)
        dev = {"Total": con.execute("SELECT cost_usd FROM mart_hourly_total ORDER BY hour_idx").df().cost_usd.to_numpy()}
        for cat in ["Delay-insensitive", "Interactive", "Unknown"]:
            dev[cat] = con.execute("SELECT cost_usd FROM mart_hourly_category WHERE vm_category = ? ORDER BY hour_idx",
                                   [cat]).df().cost_usd.to_numpy()
        reg = {name: score({name: y}, chosen) for name, y in dev.items()}
        print(f"TEST half ({len(test)} teams, never used for design choices) -- error on the period total, %")
        print((res * 100).round(3).sort_values("next_week").to_string())
        for name, r in reg.items():
            print(f"\nRegression check -- {name}")
            print((r * 100).round(3).to_string())
        (OUT / "team_forecast_test.json").write_text(json.dumps({
            "teams": list(test), "test_error_pct": (res * 100).round(4).to_dict(),
            "regression_check_pct": {k: (v * 100).round(4).to_dict() for k, v in reg.items()}}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tune")
