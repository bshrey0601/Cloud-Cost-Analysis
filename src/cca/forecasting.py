"""Cloud-spend forecasting, built from scratch in NumPy.

Models
------
CloudSpendForecaster = blend of the two from-scratch models below plus a weekly seasonal
naive, with weights chosen per series AND per horizon on validation folds inside the
training data.

Component 1: "SeasonalRidge"
For an hourly cost series y_t we fit, by ridge-regularised least squares,

    y_t = b0 + b1 * t                                  (trend)
        + sum_k [c_k sin(2πkt/24)  + d_k cos(2πkt/24)]  (daily shape, K=4)
        + sum_k [e_k sin(2πkt/168) + f_k cos(2πkt/168)] (weekly shape, K=3)
        + g1 * y_{t-24} + g2 * y_{t-168}                (seasonal memory)

Forecasts more than 24 h ahead are produced recursively (predictions feed the
lags). Uncertainty comes from split-conformal calibration: the empirical
quantiles of the model's own out-of-sample residuals from earlier backtest
folds, so the interval width is learned, not assumed Gaussian.

The module also contains the baselines the model must beat (naive, seasonal
naive 24h/168h, Holt-Winters from statsmodels) and a rolling-origin backtest.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DAY, WEEK = 24, 168


# --------------------------------------------------------------------- model
@dataclass
class SeasonalRidge:
    daily_k: int = 4
    weekly_k: int = 3
    alpha: float = 1.0
    lags: tuple[int, ...] = (DAY, WEEK)
    coef_: np.ndarray | None = field(default=None, repr=False)
    mu_: np.ndarray | None = field(default=None, repr=False)
    sd_: np.ndarray | None = field(default=None, repr=False)
    y_: np.ndarray | None = field(default=None, repr=False)
    scale_: float = 1.0

    def _features(self, t: np.ndarray, y_hist: np.ndarray) -> np.ndarray:
        cols = [t / 1000.0]
        for k in range(1, self.daily_k + 1):
            cols += [np.sin(2 * np.pi * k * t / DAY), np.cos(2 * np.pi * k * t / DAY)]
        for k in range(1, self.weekly_k + 1):
            cols += [np.sin(2 * np.pi * k * t / WEEK), np.cos(2 * np.pi * k * t / WEEK)]
        for lag in self.lags:
            cols.append(y_hist[t - lag] / self.scale_)
        return np.column_stack(cols)

    def fit(self, y: np.ndarray) -> "SeasonalRidge":
        y = np.asarray(y, dtype=float)
        self.y_ = y.copy()
        self.scale_ = float(np.mean(y)) or 1.0
        start = max(self.lags)
        if len(y) <= start + 24:
            raise ValueError(f"need more than {start + 24} observations, got {len(y)}")
        t = np.arange(start, len(y))
        X = self._features(t, y)
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-12
        Xs = (X - self.mu_) / self.sd_
        Xs = np.column_stack([np.ones(len(t)), Xs])
        target = y[start:] / self.scale_
        reg = self.alpha * np.eye(Xs.shape[1])
        reg[0, 0] = 0.0  # do not shrink the intercept
        self.coef_ = np.linalg.solve(Xs.T @ Xs + reg, Xs.T @ target)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        hist = list(self.y_)
        n0 = len(self.y_)
        out = []
        for h in range(horizon):
            t = np.array([n0 + h])
            X = (self._features(t, np.asarray(hist)) - self.mu_) / self.sd_
            yhat = float(np.concatenate([[1.0], X[0]]) @ self.coef_) * self.scale_
            out.append(yhat)
            hist.append(yhat)
        return np.asarray(out)


@dataclass
class ProfileDampedTrend:
    """Our production model: robust daily profile + self-tuned damped-trend smoothing.

    1. Daily profile p(h): over the last `profile_days` days, take each hour's deviation
       from a centred 24-hour moving average and keep the *median* per hour of day
       (medians make the profile robust to the very anomalies we later want to detect).
    2. De-seasonalise: z_t = y_t - p(t mod 24).
    3. Fit Holt's damped trend on z_t, choosing (alpha, beta, phi) by grid search on
       the squared error of 24-hour-ahead forecasts over the last `tune_days` days
       (horizon-matched tuning; one-step tuning over-trusted short-term trends).
    4. Forecast: z_hat(n+h) = level + (phi + ... + phi^h) * trend, then add p back.

    Every step is implemented below in NumPy; no forecasting library is used.
    """

    profile_days: int = 7
    tune_days: int = 14
    tune_horizon: int = DAY
    alphas: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0)
    betas: tuple[float, ...] = (0.0, 0.01, 0.05, 0.1, 0.2)
    phis: tuple[float, ...] = (0.8, 0.9, 0.95, 0.98)
    params_: dict = field(default_factory=dict)
    profile_: np.ndarray | None = field(default=None, repr=False)
    level_: float = 0.0
    trend_: float = 0.0
    n_: int = 0

    @staticmethod
    def _daily_profile(y: np.ndarray, days: int) -> np.ndarray:
        seg = y[-days * DAY:] if len(y) >= days * DAY else y
        offset = len(y) - len(seg)
        ma = np.convolve(seg, np.ones(DAY) / DAY, mode="same")
        dev = seg - ma
        valid = np.arange(len(seg))
        valid = (valid >= DAY // 2) & (valid < len(seg) - DAY // 2)
        hours = (np.arange(len(seg)) + offset) % DAY
        prof = np.array([np.median(dev[valid & (hours == h)]) if np.any(valid & (hours == h)) else 0.0
                         for h in range(DAY)])
        return prof - prof.mean()

    @staticmethod
    def _holt(z: np.ndarray, a: float, b: float, phi: float, score_from: int, h: int):
        """Run Holt's damped filter; score the h-step-ahead forecast made at every time t.

        Scoring at the horizon we actually forecast (h = 24 by default) instead of one
        step ahead stops the grid search from rewarding a trend that chases the last
        few hours -- a trend that compounds into large errors days ahead.
        """
        n = len(z)
        levels = np.empty(n)
        trends = np.empty(n)
        level, trend = z[0], 0.0
        for t in range(n):
            if t > 0:
                new_level = a * z[t] + (1 - a) * (level + phi * trend)
                trend = b * (new_level - level) + (1 - b) * phi * trend
                level = new_level
            levels[t], trends[t] = level, trend
        damp_h = float(np.sum(phi ** np.arange(1, h + 1)))
        origins = np.arange(max(0, score_from - h), n - h)
        pred = levels[origins] + damp_h * trends[origins]
        sse = float(np.sum((z[origins + h] - pred) ** 2))
        return sse, level, trend

    def fit(self, y: np.ndarray) -> "ProfileDampedTrend":
        y = np.asarray(y, float)
        self.n_ = len(y)
        self.profile_ = self._daily_profile(y, self.profile_days)
        z = y - self.profile_[np.arange(len(y)) % DAY]
        score_from = max(1, len(z) - self.tune_days * DAY)
        best = None
        for a in self.alphas:
            for b in self.betas:
                for phi in self.phis:
                    sse, lvl, tr = self._holt(z, a, b, phi, score_from, self.tune_horizon)
                    if best is None or sse < best[0]:
                        best = (sse, a, b, phi, lvl, tr)
        _, a, b, phi, self.level_, self.trend_ = best
        self.params_ = {"alpha": a, "beta": b, "phi": phi}
        return self

    def predict(self, horizon: int) -> np.ndarray:
        phi = self.params_["phi"]
        damp = np.cumsum(phi ** np.arange(1, horizon + 1))
        z_hat = self.level_ + damp * self.trend_
        hours = (np.arange(self.n_, self.n_ + horizon)) % DAY
        return z_hat + self.profile_[hours]


# ----------------------------------------------------------------- baselines
def naive(y: np.ndarray, horizon: int) -> np.ndarray:
    return np.repeat(y[-1], horizon)


def seasonal_naive(y: np.ndarray, horizon: int, period: int) -> np.ndarray:
    reps = int(np.ceil(horizon / period))
    return np.tile(y[-period:], reps)[:horizon]


def holt_winters(y: np.ndarray, horizon: int) -> np.ndarray:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    model = ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal="add",
                                 seasonal_periods=DAY, initialization_method="estimated")
    return model.fit(optimized=True).forecast(horizon)


@dataclass
class CloudSpendForecaster:
    """Production model: a self-selecting, horizon-aware blend of three forecasters.

        forecast = w1 * ProfileDampedTrend + w2 * SeasonalRidge + w3 * SeasonalNaive(last week)

    Inside `fit`, the model runs its own rolling backtest on the *last `val_days`
    forecast origins inside the training data*, at the SAME horizon it will be asked
    to forecast, and picks (w1, w2, w3) from a simplex grid by lowest validation WAPE.
    Why horizon-aware: day-ahead, the damped-trend level is hard to beat; week-ahead,
    Microsoft's spend has a weekly rhythm and "same hour last week" becomes valuable.
    If there is not enough history to validate, it falls back to equal weights.
    No information from the forecast period is used, so backtests stay honest.
    """

    horizon: int = DAY
    val_days: int = 4
    step: float = 0.25
    weights_: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
    val_wape_: float | None = None
    components_: list = field(default_factory=list, repr=False)
    y_: np.ndarray | None = field(default=None, repr=False)

    @staticmethod
    def _component_preds(train: np.ndarray, h: int) -> list[np.ndarray]:
        return [ProfileDampedTrend().fit(train).predict(h), SeasonalRidge().fit(train).predict(h),
                seasonal_naive(train, h, WEEK)]

    def _grid(self) -> list[tuple[float, float, float]]:
        k = int(round(1 / self.step))
        return [(i * self.step, j * self.step, (k - i - j) * self.step)
                for i in range(k + 1) for j in range(k + 1 - i)]

    def fit(self, y: np.ndarray) -> "CloudSpendForecaster":
        y = np.asarray(y, float)
        self.y_ = y
        h, n = self.horizon, len(y)
        folds = []
        for end in range(n - h - (self.val_days - 1) * DAY, n - h + 1, DAY):
            train, test = y[:end], y[end:end + h]
            if end <= WEEK + 2 * DAY or len(test) < h:
                continue
            folds.append((test, self._component_preds(train, h)))
        if folds:
            denom = sum(np.abs(t).sum() for t, _ in folds)
            best = None
            for w in self._grid():
                err = sum(np.abs(t - sum(wi * p for wi, p in zip(w, preds))).sum() for t, preds in folds) / denom
                if best is None or err < best[0]:
                    best = (err, w)
            self.val_wape_, self.weights_ = best
        return self

    def predict(self, horizon: int) -> np.ndarray:
        preds = self._component_preds(self.y_, horizon)
        return sum(w * p for w, p in zip(self.weights_, preds))


MODELS = {
    "CloudSpendForecaster (ours, self-selecting blend)": lambda y, h: CloudSpendForecaster(horizon=h).fit(y).predict(h),
    "ProfileDampedTrend (ours)": lambda y, h: ProfileDampedTrend().fit(y).predict(h),
    "SeasonalRidge (ours)": lambda y, h: SeasonalRidge().fit(y).predict(h),
    "Holt-Winters (statsmodels)": holt_winters,
    "Seasonal naive (same hour last week)": lambda y, h: seasonal_naive(y, h, WEEK),
    "Seasonal naive (same hour yesterday)": lambda y, h: seasonal_naive(y, h, DAY),
    "Naive (last value)": naive,
}


# ------------------------------------------------------------------ metrics
def wape(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sum(np.abs(actual - pred)) / np.sum(np.abs(actual)))


def mape(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs((actual - pred) / actual)))


def rolling_backtest(y: np.ndarray, first_origin_day: int = 15, horizon: int = DAY,
                     models: dict | None = None) -> pd.DataFrame:
    """Rolling-origin evaluation: train on days [1..d], forecast the next `horizon` hours.

    Returns one row per (model, origin) with hourly WAPE and the daily-total error.
    """
    models = models or MODELS
    rows = []
    n_days = len(y) // DAY
    for d in range(first_origin_day, n_days - horizon // DAY + 1):
        train, test = y[: d * DAY], y[d * DAY: d * DAY + horizon]
        if len(test) < horizon:
            break
        for name, fn in models.items():
            pred = fn(train, horizon)
            rows.append({
                "model": name, "origin_day": d, "forecast_days": f"{d + 1}-{d + horizon // DAY}",
                "hourly_wape": wape(test, pred),
                "total_actual": float(test.sum()), "total_pred": float(pred.sum()),
                "total_pct_error": float((pred.sum() - test.sum()) / test.sum()),
            })
    return pd.DataFrame(rows)


def summarize_backtest(bt: pd.DataFrame) -> pd.DataFrame:
    s = bt.groupby("model").agg(
        folds=("origin_day", "count"),
        hourly_wape=("hourly_wape", "mean"),
        total_mape=("total_pct_error", lambda e: float(np.mean(np.abs(e)))),
        total_bias=("total_pct_error", "mean"),
    ).sort_values("hourly_wape")
    return s


# --------------------------------------------------------- forecast + bands
@dataclass
class Forecast:
    hours: np.ndarray
    point: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    coverage_level: float
    total_point: float
    total_lower: float
    total_upper: float
    method: str


def _conformal_quantile(errors: list[float], level: float) -> float:
    """Split-conformal quantile with the finite-sample correction ceil((n+1)level)/n."""
    n = len(errors)
    if n == 0:
        return 0.05
    q = min(1.0, np.ceil((n + 1) * level) / n)
    return float(np.quantile(errors, q, method="higher"))


def forecast_with_intervals(y: np.ndarray, horizon: int, level: float = 0.9,
                            min_train_days: int = 9) -> Forecast:
    """Point forecast from CloudSpendForecaster plus split-conformal intervals.

    Calibration: for every earlier origin with at least `min_train_days` of history,
    refit on the data before it and record the relative error at each horizon step
    (and on the horizon total). Interval half-widths are conformal quantiles of those
    out-of-sample errors, so they are learned from the model's real track record.
    """
    y = np.asarray(y, float)
    h_days = int(np.ceil(horizon / DAY))
    n_days = len(y) // DAY
    rel_err_by_step: list[list[float]] = [[] for _ in range(horizon)]
    total_rel_err: list[float] = []
    for d in range(min_train_days, n_days - h_days + 1):
        train, test = y[: d * DAY], y[d * DAY: d * DAY + horizon]
        if len(test) < horizon:
            continue
        pred = CloudSpendForecaster(horizon=horizon).fit(train).predict(horizon)
        for i in range(horizon):
            rel_err_by_step[i].append(abs(test[i] - pred[i]) / abs(pred[i]))
        total_rel_err.append(abs(test.sum() - pred.sum()) / abs(pred.sum()))
    # Pool hours within each forecast day (more calibration points per quantile).
    q_step = np.zeros(horizon)
    for day in range(h_days):
        idx = range(day * DAY, min((day + 1) * DAY, horizon))
        pooled = [e for i in idx for e in rel_err_by_step[i]]
        q_step[list(idx)] = _conformal_quantile(pooled, level)
    q_step = np.maximum.accumulate(q_step)  # uncertainty never shrinks with horizon
    q_total = _conformal_quantile(total_rel_err, level)

    model = CloudSpendForecaster(horizon=horizon).fit(y)
    point = model.predict(horizon)
    tp = float(point.sum())
    return Forecast(
        hours=np.arange(len(y), len(y) + horizon), point=point,
        lower=point * (1 - q_step), upper=point * (1 + q_step), coverage_level=level,
        total_point=tp, total_lower=tp * (1 - q_total), total_upper=tp * (1 + q_total),
        method=(f"CloudSpendForecaster (weights damped-trend/ridge/last-week = "
                f"{'/'.join(f'{w:.2f}' for w in model.weights_)}) + split-conformal {int(level * 100)}% intervals "
                f"calibrated on {len(total_rel_err)} out-of-sample folds"),
    )


def interval_coverage(y: np.ndarray, horizon: int = DAY, level: float = 0.9,
                      first_origin_day: int = 18) -> dict:
    """Out-of-sample check that the 90% bands really contain ~90% of actuals."""
    hits, n, total_hits, folds = 0, 0, 0, 0
    n_days = len(y) // DAY
    for d in range(first_origin_day, n_days - horizon // DAY + 1):
        train, test = y[: d * DAY], y[d * DAY: d * DAY + horizon]
        fc = forecast_with_intervals(train, horizon, level)
        hits += int(np.sum((test >= fc.lower) & (test <= fc.upper)))
        n += len(test)
        total_hits += int(fc.total_lower <= test.sum() <= fc.total_upper)
        folds += 1
    return {"level": level, "hourly_coverage": hits / n, "hourly_points": n,
            "daily_total_coverage": total_hits / folds, "folds": folds}
