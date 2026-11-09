"""Benchmark questions with ground truth computed INDEPENDENTLY of the warehouse.

Ground truth is computed straight from the raw Microsoft parquet file with a different
method (interval-overlap arithmetic per VM, prices looked up in Python) than the
warehouse pipeline (range join to an hour calendar, SQL price dimension). Agreement
between the two is itself a check of the pipeline.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cca import pricing  # noqa: E402

RAW = ROOT / "data" / "raw" / "vmtable.parquet"
DAY_S = 86_400
TRACE_END_S = 30 * DAY_S


@dataclass
class BenchQuestion:
    qid: str
    kind: str
    question: str
    truth: float
    unit: str       # usd | pct


def _raw_con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    prices = pd.DataFrame(
        [{"core_bucket": c, "memory_bucket": m, "price": pricing.price_for(c, m)[1]}
         for c in pricing.CORE_BUCKET_VALUE for m in pricing.MEMORY_BUCKET_VALUE])
    con.register("prices", prices)
    con.execute(f"""
        CREATE TABLE vm AS
        WITH r AS (SELECT *, MAX(deleted_s) OVER () AS last_reading FROM read_parquet('{RAW}'))
        SELECT r.subscription_id, r.vm_category, r.created_s,
               CASE WHEN r.deleted_s >= r.last_reading THEN {TRACE_END_S} ELSE r.deleted_s + 300 END AS end_s,
               p.price
        FROM r JOIN prices p USING (core_bucket, memory_bucket)""")
    return con


def _spend(con, a: int, b: int, where: str = "1=1", group: str | None = None):
    lo, hi = (a - 1) * DAY_S, b * DAY_S
    overlap = f"GREATEST(0, LEAST(end_s, {hi}) - GREATEST(created_s, {lo})) / 3600.0 * price"
    if group:
        return con.execute(f"SELECT {group}, SUM({overlap}) s FROM vm WHERE {where} GROUP BY 1 ORDER BY 2 DESC").df()
    return float(con.execute(f"SELECT SUM({overlap}) FROM vm WHERE {where}").fetchone()[0])


def build_questions(seed: int = 2026, n_each: int = 6) -> list[BenchQuestion]:
    rng = random.Random(seed)
    con = _raw_con()
    qs: list[BenchQuestion] = []

    def window(min_len=3, max_len=10):
        L = rng.randint(min_len, max_len)
        s = rng.randint(1, 31 - L)
        return s, s + L - 1

    for i in range(n_each):
        a, b = window()
        qs.append(BenchQuestion(f"total-{i}", "total_spend", f"How much did we spend in total in days {a}-{b}?",
                                _spend(con, a, b), "usd"))
    for i in range(n_each):
        a, b = window()
        qs.append(BenchQuestion(f"davg-{i}", "daily_average",
                                f"What was the average daily spend over days {a}-{b}?",
                                _spend(con, a, b) / (b - a + 1), "usd"))
    for i in range(n_each):
        a, b = window()
        top = _spend(con, a, b, group="subscription_id").s.iloc[0]
        qs.append(BenchQuestion(f"topsub-{i}", "top_subscription",
                                f"Which subscription spent the most in days {a}-{b}, and how much?", float(top), "usd"))
    cats = ["Delay-insensitive", "Interactive", "Unknown"]
    for i in range(n_each):
        a, b = window()
        cat = cats[i % 3]
        share = _spend(con, a, b, where=f"vm_category = '{cat}'") / _spend(con, a, b)
        qs.append(BenchQuestion(f"catshare-{i}", "category_share",
                                f"What share of spend by workload category went to {cat} in days {a}-{b}?",
                                share, "pct"))
    for i in range(n_each):
        L = rng.randint(3, 7)
        b1 = rng.randint(1, 30 - 2 * L + 1)
        a2 = rng.randint(b1 + L, 31 - L)
        w1, w2 = (b1, b1 + L - 1), (a2, a2 + L - 1)
        avg1, avg2 = _spend(con, *w1) / L, _spend(con, *w2) / L
        qs.append(BenchQuestion(f"change-{i}", "period_change",
                                f"By what percent did average daily spend change between days {w1[0]}-{w1[1]} "
                                f"and days {w2[0]}-{w2[1]}?", (avg2 - avg1) / avg1, "pct"))
    return qs


if __name__ == "__main__":
    for q in build_questions():
        print(q.qid, "|", q.question, "|", round(q.truth, 4))
