"""The self-checking AI cost analyst.

    Planner/Retriever --> Analyst (tools: SQL + ML) --> Writer --> Auditor --+--> memo
          (LLM)              (deterministic)            (LLM)   (no LLM)    |
                                                          ^-- revise <------+ (max 2 rounds)

If the LLM still cannot produce a memo that passes audit, the agent falls back to a
deterministic template memo built directly from the ledger, so a failed audit can
never reach a stakeholder. With no LLM configured, the planner is rule-based and the
writer is the template -- the app still works end to end.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from cca.auditor import AuditReport, audit
from cca.ledger import Ledger
from cca.llm import LLM
from cca.tools import TOOL_SPECS, TOOLS, run_tool
from cca.warehouse import active_db_path

PROMPTS = Path(__file__).parent / "prompts"
CATEGORY_WORDS = {"delay": "Delay-insensitive", "batch": "Delay-insensitive", "interactive": "Interactive",
                  "unknown": "Unknown"}


def _prompt(name: str, **kw) -> str:
    return (PROMPTS / f"{name}.md").read_text().format(**kw)


@dataclass
class AgentResult:
    question: str
    memo: str
    audit: AuditReport
    ledger: Ledger
    steps: list[dict] = field(default_factory=list)
    mode: str = "offline"
    revisions: int = 0
    fell_back: bool = False
    first_draft_audit: AuditReport | None = None
    seconds: float = 0.0


# ------------------------------------------------------------ rule planner
def _windows(q: str) -> list[tuple[int, int]]:
    ql = q.lower()
    wins = [(int(a), int(b)) for a, b in re.findall(r"days?\s*(\d{1,2})\s*(?:-|–|to|through|and)\s*(\d{1,2})", ql)]
    wins += [(7 * (int(w) - 1) + 1, min(30, 7 * int(w))) for w in re.findall(r"week\s*(\d)", ql)]
    m = re.search(r"last\s+(\d{1,2})\s+days", ql)
    if m:
        n = int(m.group(1))
        wins.append((31 - n, 30))
    m = re.search(r"first\s+(\d{1,2})\s+days", ql)
    if m:
        wins.append((1, int(m.group(1))))
    wins += [(int(d), int(d)) for d in re.findall(r"(?<!days )\bday\s*(\d{1,2})\b(?!\s*(?:-|–|to|through|and)\s*\d)", ql)]
    return [w for w in wins if 1 <= w[0] <= w[1] <= 30]


def _scope(q: str) -> str | None:
    m = re.search(r"SUB-\d{4}", q, re.I)
    if m:
        return m.group(0).upper()
    ql = q.lower()
    for w, cat in CATEGORY_WORDS.items():
        if w in ql:
            return cat
    return None


def _dimension(q: str) -> str:
    ql = q.lower()
    if re.search(r"categor|workload|delay|interactive", ql):
        return "category"
    if re.search(r"shape|sku|size|instance type|vm type", ql):
        return "shape"
    return "subscription"


def rule_plan(question: str) -> list[dict]:
    q = question.lower()
    wins = _windows(question)
    scope = _scope(question)
    steps: list[dict] = []
    if re.search(r"forecast|predict|next \d+ days|next week|project|run-?rate|budget", q):
        m = re.search(r"next\s+(\d{1,2})\s+days", q)
        h = int(m.group(1)) if m else 7
        steps.append({"tool": "forecast_spend", "args": {"scope": scope or "total", "horizon_days": min(max(h, 1), 14)}})
    if re.search(r"anomal|spike|unusual|outlier|unexpected|alert", q):
        steps.append({"tool": "detect_anomalies", "args": {"scope": scope or "subscriptions", "top_n": 5}})
    if re.search(r"sav|optimi|waste|idle|rightsiz|right-siz|reduce (?:cost|spend)|cut cost|commit|reserv|spot", q):
        steps.append({"tool": "optimization_opportunities", "args": {}})
    if re.search(r"compar|change|why|increase|decrease|grow|drop|variance|vs\.?|versus|differ", q) and wins:
        if len(wins) >= 2:
            # base = the earlier window, comparison = the later one, whatever order they're written in
            (bs, be), (cs, ce) = sorted(wins[:2])
        else:
            cs, ce = wins[0]
            n = ce - cs + 1
            bs, be = max(1, cs - n), cs - 1
            if be < 1:
                bs, be, cs, ce = cs, ce, ce + 1, min(30, ce + n)
        steps.append({"tool": "compare_periods", "args": {"base_start": bs, "base_end": be, "comp_start": cs,
                                                           "comp_end": ce, "dimension": _dimension(question)}})
    has_compare = any(st["tool"] == "compare_periods" for st in steps)
    if not steps or (not has_compare and re.search(r"\btop\b|largest|biggest|most expensive|how much|breakdown|by spend|spend by", q)):
        s, e = wins[0] if wins else (1, 30)
        steps.append({"tool": "spend_summary", "args": {"start_day": s, "end_day": e,
                                                         "dimension": _dimension(question), "top_n": 5}})
    return steps


# ---------------------------------------------------------- template writer
def _f(ledger: Ledger, fid: str) -> str:
    return f"{ledger.facts[fid].display()} [{fid}]"


def _fa(ledger: Ledger, fid: str) -> str:
    """Absolute value display -- the sentence's verb carries the direction."""
    f = ledger.facts[fid]
    from cca.ledger import format_value
    return f"{format_value(abs(f.value), f.unit)} [{fid}]"


def template_memo(question: str, ledger: Ledger, results: list[tuple[dict, dict]]) -> str:
    """Deterministic memo built straight from the ledger: correct by construction."""
    answer, drivers, actions = [], [], []
    for step, res in results:
        ids = res.get("_facts", [])
        t = step["tool"]
        if t == "describe_dataset" and ids:
            answer.append(f"Across the 30-day trace, spend totalled {_f(ledger, ids[0])}, or {_f(ledger, ids[1])} per day.")
        elif t == "spend_summary" and ids:
            w = f"days {res['start_day']}-{res['end_day']}"
            answer.append(f"Spend in {w} was {_f(ledger, ids[0])}, averaging {_f(ledger, ids[1])} per day.")
            for i, top in enumerate(res["top"][:4]):
                drivers.append(f"{top['name']}: {_f(ledger, ids[2 + 2 * i])}, {_f(ledger, ids[3 + 2 * i])} of spend.")
        elif t == "compare_periods" and ids:
            b, c = res["base_days"], res["comp_days"]
            verb = "increased" if res["change_usd_per_day"] >= 0 else "decreased"
            answer.append(f"Average daily spend {verb} from {_f(ledger, ids[0])} in days {b[0]}-{b[1]} to "
                          f"{_f(ledger, ids[1])} in days {c[0]}-{c[1]}, {"rising" if verb == "increased" else "falling"} by {_fa(ledger, ids[2])} per day "
                          f"({_fa(ledger, ids[3])}).")
            for i, d in enumerate(res["drivers"][:4]):
                v = "added" if d["change_usd_per_day"] >= 0 else "removed"
                same_way = (d["change_usd_per_day"] >= 0) == (res["change_usd_per_day"] >= 0)
                share = "explaining" if same_way else "offsetting"
                drivers.append(f"{d['name']} {v} {_fa(ledger, ids[4 + 2 * i])} per day, "
                               f"{share} {_fa(ledger, ids[5 + 2 * i])} of the net change.")
            if res["drivers"]:
                top = res["drivers"][0]["name"]
                owner = f"the owner of {top}" if res["dimension"] == "subscription" else f"the teams running {top} workloads"
                actions.append(f"Ask {owner} to confirm whether the change is planned.")
        elif t == "detect_anomalies" and ids:
            answer.append(f"The detector found {_f(ledger, ids[0])} anomaly events in {res['scope']}, with "
                          f"{_f(ledger, ids[1])} of excess spend in spike events.")
            for i, e in enumerate(res["top_events"][:4]):
                k = 2 + 4 * i
                word = "above" if e["direction"] == "spike" else "below"
                kind = {("spike", "sustained"): "new higher level that persisted",
                        ("drop", "sustained"): "new lower level that persisted",
                        ("spike", "transient"): "temporary spike", ("drop", "transient"): "temporary drop"}[(e["direction"], e["pattern"])]
                drivers.append(f"{e['series']} ({kind}, day {e['start_day']}): {_fa(ledger, ids[k])} "
                               f"{word} expected over {_f(ledger, ids[k + 2])} ({_fa(ledger, ids[k + 1])} {word} expected).")
            if res["top_events"]:
                actions.append(f"Route the {res['top_events'][0]['series']} event to its engineering owner for root cause.")
        elif t == "forecast_spend" and ids:
            word = "above" if res["change_vs_last7_pct"] >= 0 else "below"
            answer.append(f"Forecast spend for {res['scope']} over the next {res['horizon_days']} days is "
                          f"{_f(ledger, ids[0])} (90% range {_f(ledger, ids[1])} to {_f(ledger, ids[2])}).")
            drivers.append(f"That is {_f(ledger, ids[3])} per day, {_fa(ledger, ids[5])} {word} the last-7-day "
                           f"average of {_f(ledger, ids[4])}.")
        elif t == "optimization_opportunities" and ids:
            answer.append(f"Idle shutdown and rightsizing could save {_f(ledger, ids[1])} over 30 days, "
                          f"{_f(ledger, ids[2])} of {_f(ledger, ids[0])} total spend, with no assumed discounts.")
            labels = {ledger.facts[i].label: i for i in ids}
            for lever in ["Shut down idle VMs", "Rightsize over-provisioned VMs"]:
                if f"{lever}: savings" in labels:
                    drivers.append(f"{lever}: {_f(ledger, labels[lever + ': VMs'])} VMs, saving "
                                   f"{_f(ledger, labels[lever + ': savings'])}.")
            spot = labels.get("Move delay-insensitive work to Spot: addressable spend")
            if spot:
                drivers.append(f"Delay-insensitive spend eligible for Spot capacity: {_f(ledger, spot)}.")
            top = res["top_subscriptions_by_savings"]
            if top:
                actions.append(f"Start with {top[0]['name']}, the subscription with the largest hard savings.")
        elif t == "compute" and ids:
            answer.append(f"{ledger.facts[ids[0]].label}: {_f(ledger, ids[0])}.")
    parts = ["**Answer:** " + " ".join(answer or ["No tool returned results for this question."])]
    if drivers:
        parts.append("**What drives it:**\n" + "\n".join(f"- {d}" for d in drivers))
    if actions:
        parts.append("**Recommended action:**\n" + "\n".join(f"- {a}" for a in actions))
    parts.append("**Caveats:**\n- Costs apply current Azure list prices to Microsoft's 2019 usage trace; "
                 "Spot and commitment discounts are inputs, not facts.")
    return "\n\n".join(parts)


# ----------------------------------------------------------------- agent
def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in response")
    return json.loads(m.group(0))


def _valid_steps(obj: dict) -> list[dict]:
    steps = obj.get("steps", [])
    out = []
    for s in steps:
        if isinstance(s, dict) and (s.get("tool") in TOOLS or s.get("tool") == "compute"):
            out.append({"tool": s["tool"], "args": s.get("args") or {}})
    return out


class CostAnalystAgent:
    def __init__(self, llm: LLM | None = None, db_path: str | Path | None = None, max_revisions: int = 2):
        self.llm = llm
        self.db_path = str(db_path or active_db_path())
        self.max_revisions = max_revisions

    def _execute(self, con, ledger: Ledger, steps: list[dict], trace: list[dict], results: list) -> list[str]:
        errors = []
        for st in steps:
            t0 = time.time()
            try:
                res = run_tool(con, ledger, st["tool"], st["args"])
                results.append((st, res))
                trace.append({**st, "facts": res["_facts"], "ok": True, "ms": int((time.time() - t0) * 1000)})
            except Exception as exc:  # tool errors are reported back to the planner, not raised
                errors.append(f"{st['tool']}({st['args']}): {exc}")
                trace.append({**st, "ok": False, "error": str(exc)})
        return errors

    def answer(self, question: str) -> AgentResult:
        import duckdb

        t0 = time.time()
        ledger, trace, results = Ledger(), [], []
        mode = self.llm.name if self.llm else "offline"
        con = duckdb.connect(self.db_path, read_only=True)
        try:
            # 1. plan + retrieve
            steps = None
            if self.llm:
                try:
                    raw = self.llm.generate(_prompt("planner", tool_specs=TOOL_SPECS), f"Question: {question}")
                    steps = _valid_steps(_extract_json(raw))
                except Exception as exc:
                    trace.append({"tool": "planner", "ok": False, "error": f"planner output unusable: {exc}"})
            if not steps:
                steps = rule_plan(question)
            errors = self._execute(con, ledger, steps, trace, results)
            if self.llm:
                try:
                    raw = self.llm.generate(_prompt("planner", tool_specs=TOOL_SPECS), _prompt(
                        "planner_round2", fact_table=ledger.table(), errors=errors or "none", question=question))
                    more = _valid_steps(_extract_json(raw))
                    if more:
                        self._execute(con, ledger, more, trace, results)
                except Exception as exc:
                    trace.append({"tool": "planner_round2", "ok": False, "error": str(exc)})
        finally:
            con.close()

        # 2. write + 3. audit (+ revise)
        fallback = template_memo(question, ledger, results)
        if not self.llm:
            rep = audit(fallback, ledger, self.db_path)
            return AgentResult(question, fallback, rep, ledger, trace, mode, 0, False, rep, time.time() - t0)

        memo = self.llm.generate(_prompt("writer"), _prompt("writer_task", question=question, fact_table=ledger.table()))
        rep = audit(memo, ledger, self.db_path)
        first = rep
        revisions = 0
        while not rep.passed and revisions < self.max_revisions:
            revisions += 1
            issues = "\n".join(f"- {i['detail']}" for i in rep.issues)
            memo = self.llm.generate(_prompt("writer"), _prompt("reviser", issues=issues, memo=memo,
                                                                 fact_table=ledger.table()))
            rep = audit(memo, ledger, self.db_path)
        fell_back = False
        if not rep.passed:
            memo, fell_back = fallback, True
            rep = audit(memo, ledger, self.db_path)
        return AgentResult(question, memo, rep, ledger, trace, mode, revisions, fell_back, first, time.time() - t0)
