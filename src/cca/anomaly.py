"""Cost-anomaly detection, built from scratch in NumPy.

Detector: "RobustSeasonalDetector"
----------------------------------
For each hour t (after a 7-day warm-up):

    expected_t = level_t + profile(hour_of_day)
        level_t   = median of the de-seasonalised series over the previous 24 h
        profile   = robust daily shape (median deviation by hour, trailing 7 days)
    residual_t = y_t - expected_t
    scale_t    = 1.4826 * MAD(residuals over the previous 168 h)   (robust sigma)
    z_t        = residual_t / scale_t

An hour is flagged when |z_t| > k AND the deviation is *material* -- larger than
`min_rel` of expected spend and `min_abs_usd` dollars. Materiality thresholds are
what make this a FinOps tool rather than a statistics exercise: nobody wants a page
for a $3 blip. Consecutive flagged hours are merged into one event with a $ impact.

Medians/MAD (instead of means/std) keep the baseline from being dragged by the very
anomalies we are trying to find.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DAY, WEEK = 24, 168


@dataclass
class AnomalyEvent:
    series: str
    start_hour: int
    end_hour: int
    direction: str          # "spike" (over expected) or "drop"
    pattern: str            # "transient" (spend came back) or "sustained" (new level persisted after the alert)
    peak_z: float
    excess_cost_usd: float  # sum of (actual - expected) over the event
    expected_cost_usd: float
    actual_cost_usd: float

    @property
    def duration_hours(self) -> int:
        return self.end_hour - self.start_hour + 1

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["duration_hours"] = self.duration_hours
        d["start_day"] = self.start_hour // DAY + 1
        d["pct_vs_expected"] = (self.actual_cost_usd - self.expected_cost_usd) / self.expected_cost_usd
        return d


@dataclass
class RobustSeasonalDetector:
    # Defaults chosen on the *tuning* half of subscriptions only (see eval/run_model_validation.py):
    k: float = 6.0             # robust z threshold
    min_rel: float = 0.04      # at least 4% above/below expected spend
    min_abs_usd: float = 2.8   # at least $2.80/h  (~$2,000/month if it persisted)
    scale_floor_rel: float = 0.01  # robust sigma is at least 1% of expected spend
    warmup: int = WEEK

    def score(self, y: np.ndarray) -> pd.DataFrame:
        """Vectorised with NumPy sliding windows (identical maths to a per-hour loop)."""
        y = np.asarray(y, float)
        n = len(y)
        expected = np.full(n, np.nan)
        z = np.full(n, np.nan)
        resid = np.full(n, np.nan)
        if n > self.warmup:
            ts = np.arange(self.warmup, n)
            W = np.lib.stride_tricks.sliding_window_view(y, WEEK)[ts - WEEK]        # (m, 168) trailing week
            days = W.reshape(len(ts), 7, DAY)                                      # column c <-> hour (t + c) % 24
            day_med = np.median(days, axis=2, keepdims=True)
            profile = np.median(days - day_med, axis=1)                            # (m, 24) robust daily shape
            level = np.median(days[:, -1, :] - profile, axis=1)                     # de-seasonalised last 24 h
            expected[ts] = level + profile[:, 0]                                   # column 0 <-> hour t % 24
            resid[ts] = y[ts] - expected[ts]
        for t in range(self.warmup, n):
            past = resid[max(self.warmup, t - WEEK): t]
            if len(past) < DAY:
                # not enough residual history yet: fall back to a scale from raw volatility
                scale = 1.4826 * np.median(np.abs(np.diff(y[t - WEEK: t])))
            else:
                scale = 1.4826 * np.median(np.abs(past - np.median(past)))
            # Scale floor: steady always-on subscriptions have residuals that are exactly 0
            # most hours, which makes MAD = 0 and z explode. Never assume the series is
            # quieter than `scale_floor_rel` of its expected spend.
            scale = max(scale, self.scale_floor_rel * abs(expected[t]), 0.01)
            z[t] = resid[t] / scale
        material = (np.abs(resid) > self.min_rel * np.abs(expected)) & (np.abs(resid) > self.min_abs_usd)
        flag = (np.abs(z) > self.k) & material
        return pd.DataFrame({"hour_idx": np.arange(n), "actual": y, "expected": expected,
                             "residual": resid, "z": z, "flag": flag})

    def detect(self, y: np.ndarray, series: str = "series") -> list[AnomalyEvent]:
        s = self.score(y)
        return events_from_flags(s, series)


def events_from_flags(s: pd.DataFrame, series: str, gap: int = 1) -> list[AnomalyEvent]:
    """Merge flagged hours (allowing `gap` quiet hours inside an event) into events."""
    idx = s.index[s["flag"]].to_numpy()
    events: list[AnomalyEvent] = []
    if len(idx) == 0:
        return events
    groups = np.split(idx, np.where(np.diff(idx) > gap + 1)[0] + 1)
    for g in groups:
        seg = s.loc[g[0]: g[-1]]
        excess = float((seg["actual"] - seg["expected"]).sum())
        # Sustained vs transient: did spend in the 24 h AFTER the alert stay at least halfway
        # to the anomalous level, relative to what was expected when the event began?
        pre = float(seg["expected"].mean())
        post = s["actual"].iloc[g[-1] + 1: g[-1] + 25]
        mean_dev = excess / len(seg)
        sustained = len(post) > 0 and (float(post.median()) - pre) * np.sign(mean_dev) >= 0.5 * abs(mean_dev)
        events.append(AnomalyEvent(
            series=series, start_hour=int(g[0]), end_hour=int(g[-1]),
            direction="spike" if excess >= 0 else "drop",
            pattern="sustained" if sustained else "transient",
            peak_z=float(seg["z"].abs().max()), excess_cost_usd=excess,
            expected_cost_usd=float(seg["expected"].sum()), actual_cost_usd=float(seg["actual"].sum()),
        ))
    return events


# ----------------------------------------------------------------- baselines
@dataclass
class StaticThresholdDetector:
    """Flag hours above mean + k*std of the trailing week (common 'budget alert' logic)."""
    k: float = 3.0
    warmup: int = WEEK

    def score(self, y: np.ndarray) -> pd.DataFrame:
        y = np.asarray(y, float)
        n = len(y)
        flag = np.zeros(n, bool)
        expected = np.full(n, np.nan)
        z = np.full(n, np.nan)
        for t in range(self.warmup, n):
            hist = y[t - WEEK: t]
            mu, sd = hist.mean(), hist.std() + 1e-9
            expected[t], z[t] = mu, (y[t] - mu) / sd
            flag[t] = abs(z[t]) > self.k
        return pd.DataFrame({"hour_idx": np.arange(n), "actual": y, "expected": expected,
                             "residual": y - expected, "z": z, "flag": flag})


@dataclass
class RollingZScoreDetector:
    """Flag hours whose z-score vs the trailing 24 h mean/std exceeds k."""
    k: float = 3.0
    warmup: int = WEEK

    def score(self, y: np.ndarray) -> pd.DataFrame:
        y = np.asarray(y, float)
        n = len(y)
        flag = np.zeros(n, bool)
        expected = np.full(n, np.nan)
        z = np.full(n, np.nan)
        for t in range(self.warmup, n):
            hist = y[t - DAY: t]
            mu, sd = hist.mean(), hist.std() + 1e-9
            expected[t], z[t] = mu, (y[t] - mu) / sd
            flag[t] = abs(z[t]) > self.k
        return pd.DataFrame({"hour_idx": np.arange(n), "actual": y, "expected": expected,
                             "residual": y - expected, "z": z, "flag": flag})


# ---------------------------------------------------------------- evaluation
ANOMALY_TYPES = {
    # name: (duration range hours, relative magnitude range)
    # Magnitudes are deliberately modest (relative to the series' own level) so the
    # benchmark separates good detectors from bad ones.
    "spike": ((1, 4), (0.10, 1.00)),            # runaway job / forgotten large cluster for a few hours
    "level_shift": ((24, 72), (0.05, 0.30)),    # new environment left running for days
    "drop": ((6, 24), (-0.40, -0.10)),          # outage / mass shutdown / broken pipeline
}


def inject_anomalies(y: np.ndarray, rng: np.random.Generator, n_events: int = 3,
                     warmup: int = WEEK) -> tuple[np.ndarray, list[dict]]:
    """Inject labelled anomalies into a *real* series at random, non-overlapping times."""
    y = np.asarray(y, float).copy()
    labels: list[dict] = []
    occupied = np.zeros(len(y), bool)
    types = list(ANOMALY_TYPES)
    attempts = 0
    while len(labels) < n_events and attempts < 200:
        attempts += 1
        kind = types[len(labels) % len(types)]
        (dmin, dmax), (mmin, mmax) = ANOMALY_TYPES[kind]
        dur = int(rng.integers(dmin, dmax + 1))
        start = int(rng.integers(warmup + DAY, len(y) - dur))
        lo, hi = start - DAY, start + dur + DAY  # keep a day of clean data between events
        if occupied[max(0, lo): hi].any():
            continue
        mag = float(rng.uniform(mmin, mmax))
        local_level = np.median(y[start - DAY: start])
        y[start: start + dur] += mag * local_level
        y = np.maximum(y, 0)
        occupied[max(0, lo): hi] = True
        labels.append({"type": kind, "start": start, "end": start + dur - 1, "magnitude": mag})
    return y, labels


def evaluate_detector(detector, series: dict[str, np.ndarray], seed: int = 7,
                      n_events: int = 3, latency_tol: int = 3) -> dict:
    """Semi-synthetic evaluation on real series.

    * Alarms that already fire on the *clean* real series are treated as pre-existing
      (real anomalies we have no labels for) and are excluded from false positives.
    * An injected event counts as detected if any alarm hour falls within
      [start, start + latency_tol] for spikes/drops or [start, start + 24] for level shifts.
    """
    rng = np.random.default_rng(seed)
    tp_by_type = {k: 0 for k in ANOMALY_TYPES}
    n_by_type = {k: 0 for k in ANOMALY_TYPES}
    delays: list[int] = []
    false_alarm_events = 0
    alarm_events = 0
    clean_alarm_events = 0
    clean_weeks = 0.0
    mag_hits: dict[str, list[int]] = {"<15%": [], "15-30%": [], ">30%": []}
    for name, y in series.items():
        clean_flags = detector.score(y)["flag"].to_numpy()
        runs_clean = np.split(np.where(clean_flags)[0], np.where(np.diff(np.where(clean_flags)[0]) > 2)[0] + 1)
        clean_alarm_events += sum(1 for r in runs_clean if len(r))
        clean_weeks += (len(y) - WEEK) / WEEK
        y_inj, labels = inject_anomalies(y, rng, n_events=n_events)
        sc = detector.score(y_inj)
        flags = sc["flag"].to_numpy()
        label_mask = np.zeros(len(y), bool)
        for lab in labels:
            n_by_type[lab["type"]] += 1
            window_end = lab["start"] + (DAY if lab["type"] == "level_shift" else latency_tol)
            label_mask[lab["start"]: lab["end"] + 1] = True
            hits = np.where(flags[lab["start"]: window_end + 1])[0]
            m = abs(lab["magnitude"])
            bucket = "<15%" if m < 0.15 else ("15-30%" if m <= 0.30 else ">30%")
            mag_hits[bucket].append(int(len(hits) > 0))
            if len(hits):
                tp_by_type[lab["type"]] += 1
                delays.append(int(hits[0]))
        # count alarm events (contiguous runs) that are neither labelled nor pre-existing
        new_flags = flags & ~clean_flags
        runs = np.split(np.where(new_flags)[0], np.where(np.diff(np.where(new_flags)[0]) > 2)[0] + 1)
        for r in runs:
            if len(r) == 0:
                continue
            alarm_events += 1
            if not label_mask[max(0, r[0] - 1): r[-1] + 2].any():
                false_alarm_events += 1
    total = sum(n_by_type.values())
    detected = sum(tp_by_type.values())
    recall = detected / total
    precision = (alarm_events - false_alarm_events) / alarm_events if alarm_events else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "recall": recall, "precision": precision, "f1": f1,
        "recall_by_type": {k: tp_by_type[k] / n_by_type[k] for k in ANOMALY_TYPES if n_by_type[k]},
        "injected_events": total, "alarm_events": alarm_events, "false_alarm_events": false_alarm_events,
        "recall_by_magnitude": {k: (float(np.mean(v)) if v else None) for k, v in mag_hits.items()},
        "clean_alarms_per_series_week": clean_alarm_events / clean_weeks,
        "median_delay_hours": float(np.median(delays)) if delays else None,
        "series": len(series),
    }
