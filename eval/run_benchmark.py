"""End-to-end benchmark of the agent against independent ground truth.

    python eval/run_benchmark.py                      # offline (rule planner + template writer)
    python eval/run_benchmark.py --provider gemini    # needs GEMINI_API_KEY
    python eval/run_benchmark.py --provider anthropic # needs ANTHROPIC_API_KEY

A question is scored correct when the ground-truth value appears in the final memo,
correctly rounded (within half a unit of the last digit shown), with a direction word
that matches its sign. LLM runs also report how often the *first draft* passed audit,
how many were fixed by revision, and how many fell back to the template.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))
warnings.filterwarnings("ignore")

from benchmark_questions import build_questions  # noqa: E402
from cca.agent import CostAnalystAgent  # noqa: E402
from cca.auditor import CITE_RE, DOWN_WORDS, UP_WORDS, _parse, _sentences  # noqa: E402
from cca.llm import get_llm  # noqa: E402

OUT = ROOT / "eval" / "results"


def contains_truth(memo: str, truth: float, unit: str) -> bool:
    for sent in _sentences(memo):
        clean = CITE_RE.sub("", sent)
        for nm in _parse(clean):
            if nm.kind != unit:
                continue
            if abs(abs(nm.value) - abs(truth)) <= nm.tolerance:
                if unit == "pct" and truth < 0 and nm.value > 0 and not DOWN_WORDS.search(clean):
                    continue
                if unit == "pct" and truth > 0 and nm.value > 0 and DOWN_WORDS.search(clean) and not UP_WORDS.search(clean):
                    continue
                return True
    return False


def main(provider: str | None) -> dict:
    llm = get_llm(provider)
    agent = CostAnalystAgent(llm=llm)
    questions = build_questions()
    rows = []
    for q in questions:
        t0 = time.time()
        r = agent.answer(q.question)
        rows.append({
            "qid": q.qid, "kind": q.kind, "question": q.question, "truth": q.truth,
            "correct": contains_truth(r.memo, q.truth, q.unit),
            "final_audit_pass": r.audit.passed,
            "first_draft_pass": r.first_draft_audit.passed if r.first_draft_audit else None,
            "first_draft_issues": len(r.first_draft_audit.issues) if r.first_draft_audit else None,
            "revisions": r.revisions, "fell_back": r.fell_back,
            "tools": [s["tool"] for s in r.steps], "seconds": round(time.time() - t0, 2),
            "memo": r.memo,
        })
        print(f"{q.qid:12s} correct={rows[-1]['correct']!s:5s} audit={r.audit.passed!s:5s} "
              f"tools={rows[-1]['tools']}")
    n = len(rows)
    by_kind: dict[str, list[bool]] = {}
    for r_ in rows:
        by_kind.setdefault(r_["kind"], []).append(r_["correct"])
    summary = {
        "mode": llm.name if llm else "offline",
        "model": getattr(llm, "model", None),
        "questions": n,
        "accuracy": sum(r_["correct"] for r_ in rows) / n,
        "accuracy_by_kind": {k: sum(v) / len(v) for k, v in by_kind.items()},
        "final_audit_pass_rate": sum(r_["final_audit_pass"] for r_ in rows) / n,
        "first_draft_pass_rate": (sum(bool(r_["first_draft_pass"]) for r_ in rows) / n) if llm else None,
        "fixed_by_revision": sum(1 for r_ in rows if llm and not r_["first_draft_pass"] and not r_["fell_back"]),
        "fell_back_to_template": sum(r_["fell_back"] for r_ in rows),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    tag = summary["mode"]
    (OUT / f"benchmark_{tag}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=None, help="gemini | anthropic | offline")
    main(ap.parse_args().provider)
