"""Out-of-development check for the forecaster.

The forecaster was designed while looking at backtests on four series (total + 3 categories),
so those scores can be optimistic. This script evaluates the FINAL, unchanged model on series
that were never used during forecaster development: the hourly spend of the top-50
subscriptions (individual teams). No parameter is tuned here.

    python eval/run_untouched_forecast_check.py
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

from cca import forecasting as F  # noqa: E402
from cca.warehouse import connect  # noqa: E402

MODELS = {k: F.MODELS[k] for k in [
    "CloudSpendForecaster (ours, self-selecting blend)", "Holt-Winters (statsmodels)",
    "Seasonal naive (same hour last week)", "Seasonal naive (same hour yesterday)", "Naive (last value)"]}


def main() -> dict:
    con = connect()
    df = con.execute("SELECT subscription_alias s, hour_idx, cost_usd FROM mart_hourly_top_subscription").df()
    piv = df.pivot(index="hour_idx", columns="s", values="cost_usd").reindex(range(720)).fillna(0.0)
    out: dict = {"series": int(piv.shape[1]), "origins": "days 15-29", "horizons": {}}
    for horizon in (24, 168):
        frames = []
        for name in piv.columns:
            y = piv[name].to_numpy()
            if y[: 15 * 24].sum() == 0:      # no history before the first origin
                continue
            bt = F.rolling_backtest(y, first_origin_day=15, horizon=horizon, models=MODELS)
            bt["series"] = name
            frames.append(bt)
        bt = pd.concat(frames)
        # Pooled WAPE across all series and folds (dollar-weighted), plus per-series wins.
        bt["abs_err_total"] = (bt.total_pred - bt.total_actual).abs()
        pooled = (bt.groupby("model").abs_err_total.sum() / bt.groupby("model").total_actual.sum()).sort_values()
        per_series = bt.groupby(["series", "model"]).apply(
            lambda g: g.abs_err_total.sum() / g.total_actual.sum()).unstack()
        wins = per_series.idxmin(axis=1).value_counts()
        ours = "CloudSpendForecaster (ours, self-selecting blend)"
        beats_hw = float((per_series[ours] <= per_series["Holt-Winters (statsmodels)"]).mean())
        out["horizons"][f"{horizon}h"] = {
            "series_evaluated": int(per_series.shape[0]),
            "pooled_period_total_error": {k: float(v) for k, v in pooled.items()},
            "series_where_model_is_best": {k: int(v) for k, v in wins.items()},
            "share_of_series_ours_beats_or_ties_holt_winters": beats_hw,
            "median_series_error": {k: float(v) for k, v in per_series.median().items()},
        }
        print(f"\nHorizon {horizon}h -- {per_series.shape[0]} subscriptions")
        print((pooled * 100).round(3).to_string())
        print(f"ours <= Holt-Winters on {beats_hw:.0%} of series; best-model counts: {dict(wins)}")
    (ROOT / "eval" / "results" / "untouched_forecast_check.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
