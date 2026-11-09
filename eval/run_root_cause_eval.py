"""How actionable are the anomaly alerts? For every anomaly the detector finds on the real
top-50 subscriptions, decompose the change by deployment and measure how often ONE deployment
explains most of it (one owner to call, not twenty).

    python eval/run_root_cause_eval.py
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

from cca import anomaly as A  # noqa: E402
from cca import rootcause as RC  # noqa: E402
from cca.warehouse import connect  # noqa: E402


def main() -> dict:
    con = connect()
    df = con.execute("SELECT subscription_alias s, hour_idx, cost_usd FROM mart_hourly_top_subscription").df()
    piv = df.pivot(index="hour_idx", columns="s", values="cost_usd").reindex(range(720)).fillna(0.0)
    det = A.RobustSeasonalDetector()
    events = pd.DataFrame([e.as_dict() for c in piv.columns for e in det.detect(piv[c].to_numpy(), c)])
    out = RC.attribution_coverage(con, events)
    out["anomaly_events"] = len(events)
    # exactness of the decomposition on every explained event
    worst = 0.0
    for e in events.itertuples():
        w = RC.event_windows(int(e.start_hour) // 24 + 1)
        if w:
            r = RC.explain_change(con, e.series, *w, top_n=10**6)
            worst = max(worst, abs(sum(d["change_usd_per_day"] for d in r["drivers"]) - r["change_usd_per_day"]))
    out["max_decomposition_error_usd_per_day"] = worst
    (ROOT / "eval" / "results" / "root_cause_eval.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "examples"}, indent=2))
    return out


if __name__ == "__main__":
    main()
