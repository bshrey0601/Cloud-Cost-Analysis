"""Fault-injection test of the Auditor: corrupt correct memos the way an LLM (or a bug)
would, and measure how many corruptions the Auditor catches.

    python eval/run_auditor_fault_injection.py

Fault types
  number_drift_2pct / 10pct / 50pct   a cited number is changed by that much (re-formatted cleanly)
  unit_slip                            K <-> M (e.g. $705.6K -> $705.6M)
  flipped_direction                    increased <-> decreased, above <-> below, added <-> removed
  dropped_citation                     a number loses its [F#] citation
  wrong_citation                       a citation points at a different fact with the same unit
  fabricated_claim                     an invented, uncited dollar figure is appended
  ledger_tampering                     the memo is consistent but a ledger value was altered
                                       (a tool bug or a model editing facts) -> must fail reconciliation
The clean memos themselves must all PASS (false-positive rate).
"""

from __future__ import annotations

import json
import random
import re
import sys
import warnings
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))
warnings.filterwarnings("ignore")

from cca.agent import CostAnalystAgent  # noqa: E402
from cca.auditor import CITE_RE, audit  # noqa: E402
from cca.ledger import Fact  # noqa: E402
from cca.warehouse import active_db_path  # noqa: E402

QUESTIONS = [
    "Why did spend change in days 18-20 compared with days 11-17?",
    "Compare days 1-7 with days 8-14 by workload category",
    "How did spend by VM shape change between days 3-6 and days 25-28?",
    "Which subscriptions had unusual cost spikes?",
    "Any anomalies in the Unknown category?",
    "Forecast total spend for the next 7 days",
    "Forecast Interactive spend for the next 3 days",
    "Where can we save money?",
    "What are the top workload categories by spend in week 2?",
    "Which subscriptions spent the most in days 10-20?",
    "Top VM shapes by spend over the last 10 days",
    "How much did we spend in days 21-30?",
    "Why did SUB-0041 spend jump on day 8?",
    "What changed in SUB-0010 between days 11-13 and days 14-16?",
]

MONEY_RE = re.compile(r"(?<![\w.])\$(\d{1,3}(?:,\d{3})*(?:\.\d+)?)([KM])")
PCT_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)%")
FLIPS = [("increased", "decreased"), ("above", "below"), ("added", "removed"), ("rising", "falling")]


def _fmt_money(v: float, suffix: str, decimals: int) -> str:
    return f"${v:,.{decimals}f}{suffix}"


def drift(memo: str, rng: random.Random, pct: float) -> str | None:
    cands = [m for m in MONEY_RE.finditer(memo)] + [m for m in PCT_RE.finditer(memo)]
    cands = [m for m in cands if not memo[m.end(): m.end() + 12].lstrip().startswith(("range", "interval"))]
    if not cands:
        return None
    m = rng.choice(cands)
    sign = rng.choice([-1, 1])
    num = m.group(1).replace(",", "")
    dec = len(num.split(".")[1]) if "." in num else 0
    new_v = float(num) * (1 + sign * pct)
    new = (_fmt_money(new_v, m.group(2), dec) if m.re is MONEY_RE else f"{new_v:.{dec}f}%")
    if new == m.group(0):
        return None
    return memo[: m.start()] + new + memo[m.end():]


def unit_slip(memo: str, rng: random.Random) -> str | None:
    cands = list(MONEY_RE.finditer(memo))
    if not cands:
        return None
    m = rng.choice(cands)
    new_suffix = "M" if m.group(2) == "K" else "K"
    return memo[: m.start(2)] + new_suffix + memo[m.end(2):]


def flip_direction(memo: str, rng: random.Random) -> str | None:
    opts = [(a, b) for a, b in FLIPS if re.search(rf"\b{a}\b", memo)] + \
           [(b, a) for a, b in FLIPS if re.search(rf"\b{b}\b", memo)]
    if not opts:
        return None
    a, b = rng.choice(opts)
    hits = [m for m in re.finditer(rf"\b{a}\b", memo)]
    m = rng.choice(hits)
    return memo[: m.start()] + b + memo[m.end():]


def drop_citation(memo: str, rng: random.Random) -> str | None:
    cands = list(CITE_RE.finditer(memo))
    if not cands:
        return None
    m = rng.choice(cands)
    return memo[: m.start()].rstrip() + memo[m.end():]


def wrong_citation(memo: str, ledger, rng: random.Random) -> str | None:
    cands = [m for m in CITE_RE.finditer(memo) if "," not in m.group(1)]
    rng.shuffle(cands)
    for m in cands:
        fid = m.group(1)
        f = ledger.facts[fid]
        others = [o for o in ledger.facts.values()
                  if o.id != fid and o.unit == f.unit and abs(abs(o.value) - abs(f.value)) > 0.01 * abs(f.value) + 1e-9]
        if others:
            return memo[: m.start()] + f"[{rng.choice(others).id}]" + memo[m.end():]
    return None


def fabricate(memo: str, rng: random.Random) -> str:
    v = rng.uniform(5, 900)
    return memo.replace("**Caveats:**", f"- This also exposes ${v:.1f}K of unallocated spend.\n\n**Caveats:**", 1)


def tamper_ledger(ledger, rng: random.Random):
    led = deepcopy(ledger)
    cited = [f for f in led.facts.values() if f.tool != "compute"]
    f = rng.choice(cited)
    led.facts[f.id] = Fact(f.id, f.label, f.value * 1.07 + 1, f.unit, f.tool, f.args, f.key, f.formula)
    return led, f.id


def main(seeds: int = 5) -> dict:
    agent = CostAnalystAgent(llm=None)
    clean_results, rows = [], []
    for q in QUESTIONS:
        r = agent.answer(q)
        clean_results.append(r)
        rows.append({"question": q, "fault": "none (clean memo)", "detected": not r.audit.passed})
    for seed in range(seeds):
        rng = random.Random(seed)
        for r in clean_results:
            memo, ledger = r.memo, r.ledger
            faults = {
                "number_drift_2pct": lambda: drift(memo, rng, 0.02),
                "number_drift_10pct": lambda: drift(memo, rng, 0.10),
                "number_drift_50pct": lambda: drift(memo, rng, 0.50),
                "unit_slip": lambda: unit_slip(memo, rng),
                "flipped_direction": lambda: flip_direction(memo, rng),
                "dropped_citation": lambda: drop_citation(memo, rng),
                "wrong_citation": lambda: wrong_citation(memo, ledger, rng),
                "fabricated_claim": lambda: fabricate(memo, rng),
            }
            for name, fn in faults.items():
                bad = fn()
                if bad is None or bad == memo:
                    continue
                rep = audit(bad, ledger, str(active_db_path()))
                rows.append({"question": r.question, "fault": name, "detected": not rep.passed,
                             "issue_types": sorted({i["type"] for i in rep.issues})})
            led2, fid = tamper_ledger(ledger, rng)
            # Rewrite the memo so it agrees with the tampered ledger (worst case: everything looks consistent)
            old, new = ledger.facts[fid], led2.facts[fid]
            memo2 = memo.replace(f"{old.display()} [{fid}]", f"{new.display()} [{fid}]")
            if memo2 != memo:
                rep = audit(memo2, led2, str(active_db_path()))
                rows.append({"question": r.question, "fault": "ledger_tampering", "detected": not rep.passed,
                             "issue_types": sorted({i["type"] for i in rep.issues})})
    summary: dict[str, dict] = {}
    for row in rows:
        s = summary.setdefault(row["fault"], {"cases": 0, "detected": 0})
        s["cases"] += 1
        s["detected"] += int(row["detected"])
    for s in summary.values():
        s["rate"] = s["detected"] / s["cases"]
    faulty = [r_ for r_ in rows if r_["fault"] != "none (clean memo)"]
    out = {"questions": len(QUESTIONS), "seeds": seeds, "by_fault": summary,
           "overall_detection_rate": sum(r_["detected"] for r_ in faulty) / len(faulty),
           "false_positive_rate_on_clean": summary["none (clean memo)"]["rate"],
           "missed_examples": [r_ for r_ in faulty if not r_["detected"]][:10]}
    path = ROOT / "eval" / "results" / "auditor_fault_injection.json"
    path.write_text(json.dumps(out, indent=2))
    for k, v in summary.items():
        print(f"{k:22s} {v['detected']:4d}/{v['cases']:<4d} = {v['rate']:.1%}")
    print(f"overall detection {out['overall_detection_rate']:.1%}, false positives on clean memos "
          f"{out['false_positive_rate_on_clean']:.1%}")
    return out


if __name__ == "__main__":
    main()
