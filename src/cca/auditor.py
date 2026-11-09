"""The auditor agent: deterministic verification of every number in a memo.

It never trusts the writer. For each sentence it:
  1. finds every number (dollars, percentages, counts, hours);
  2. requires a fact citation [F#] in the same sentence;
  3. checks the number is a correct *rounding* of one of the cited facts
     (tolerance = half a unit of the last digit shown, e.g. "$1.2M" must be within $0.05M);
  4. checks direction words (increase / decrease) agree with the sign of cited change facts;
and then 5. re-runs every cited fact's tool on a fresh read-only connection and
reconciles the value -- exactly like tying a report back to the source system.

No LLM is involved, so the auditor cannot be talked out of a failure.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import duckdb

from cca.ledger import Ledger
from cca.tools import TOOLS, eval_expression, get_path

# A leading minus only counts as a sign when it is not glued to a word or digit
# ("SUB-0100" and "18-20" are identifiers/ranges, not negative numbers).
NUM_RE = re.compile(
    r"(?<![\w.])(?:(?P<neg>[-−])(?=\$?\d))?(?P<dollar>\$)?\s?(?<![\w.])(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"\s?(?P<suffix>%|[KMB](?![a-z])|k(?![a-z])|thousand|million|billion)?(?P<hours>\s?(?:h\b|hours?\b|hrs?\b))?"
)
CITE_RE = re.compile(r"\[(F\d+(?:\s*,\s*F\d+)*)\]")
UP_WORDS = re.compile(r"\b(add(?:ed|s)?|increase[sd]?|rising|rose|rise[sn]?|grew|grow(?:th|s)?|higher|up|spike[sd]?|above|over expected|jump(?:ed)?)\b", re.I)
DOWN_WORDS = re.compile(r"\b(remov(?:ed|es?)|falling|decrease[sd]?|fell|fall(?:s|en)?|drop(?:ped|s)?|declin\w*|lower|down|below|shr[ai]nk\w*|reduc\w*)\b", re.I)
SCALE = {"K": 1e3, "k": 1e3, "thousand": 1e3, "M": 1e6, "million": 1e6, "B": 1e9, "billion": 1e9}


@dataclass
class NumberMention:
    text: str
    value: float          # in base units (dollars, fraction for %, raw count)
    kind: str             # usd | pct | count | hours
    tolerance: float
    start: int


@dataclass
class AuditReport:
    passed: bool
    issues: list[dict] = field(default_factory=list)
    numbers_checked: int = 0
    facts_reconciled: int = 0

    def summary(self) -> str:
        if self.passed:
            return (f"PASS: {self.numbers_checked} numbers matched cited facts; "
                    f"{self.facts_reconciled} facts reconciled to source.")
        lines = [f"FAIL: {len(self.issues)} issue(s)"]
        lines += [f"- [{i['type']}] {i['detail']}" for i in self.issues]
        return "\n".join(lines)


def _sentences(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        out += [s for s in re.split(r"(?<=[.!?;])\s+", line) if s.strip()]
    return out


def _skip(sentence: str, m: re.Match) -> bool:
    """Numbers that are references rather than claims: day numbers, IDs, ranks, confidence levels."""
    s, e = m.start("num"), m.end()
    before = sentence[max(0, s - 12): s].lower()
    after = sentence[e: e + 14].lower()
    raw = m.group(0)
    if re.search(r"(sub-|f)$", before):
        return True                                   # SUB-0007, F12
    if not m.group("suffix") and not m.group("hours") and re.match(r"[A-Za-z_]", sentence[m.end("num"):m.end("num") + 1]):
        return True                                   # identifiers such as VM shape "24c_64g"
    if not m.group("dollar") and not m.group("suffix") and re.match(r"\s*(?:-\s*)?days?\b", after) \
            and "." not in m.group("num") and float(m.group("num")) <= 31:
        return True                                   # window lengths: "over 30 days", "a 7-day forecast"
    if re.search(r"\b(days?|top|rank|#|weeks?|hour of day|scenario)\s*$", before):
        return True                                   # day 18, top 5
    if re.search(r"\bdays?\s*\d{1,2}\s*(?:-|–|to|through|and)\s*$", before):
        return True                                   # the "30" in "days 24-30"
    if re.search(r"^\s*[-–]\s*\d", sentence[e:e + 4]) and re.search(r"days?\s*$", before):
        return True
    if re.search(r"^\s*(?:[-–]|to)\s*\d+\s*$", sentence[e:e + 6]) and "day" in before:
        return True
    if m.group("suffix") == "%" and re.match(r"\s*(confidence|prediction|interval|range|band)", after):
        return True                                   # "90% interval"
    if re.match(r"^:\d", sentence[e:e + 2]):
        return True                                   # 14:00
    if not m.group("dollar") and not m.group("suffix") and not m.group("hours"):
        val = float(m.group("num").replace(",", ""))
        if "," not in m.group("num") and val < 10 and "." not in m.group("num"):
            return True                               # small integers: "3 drivers", "2 rounds"
        if re.fullmatch(r"(19|20)\d\d", m.group("num")):
            return True                               # years
    return not raw.strip()


def _parse(sentence: str) -> list[NumberMention]:
    out = []
    for m in NUM_RE.finditer(sentence):
        if _skip(sentence, m):
            continue
        num = m.group("num").replace(",", "")
        decimals = len(num.split(".")[1]) if "." in num else 0
        val = float(num)
        if m.group("neg"):
            val = -val
        suffix = m.group("suffix")
        unit_tol = 0.5 * 10 ** (-decimals)
        if suffix == "%":
            kind, value, tol = "pct", val / 100, unit_tol / 100
        elif m.group("hours"):
            kind, value, tol = "hours", val, unit_tol
        elif m.group("dollar"):
            sc = SCALE.get(suffix or "", 1.0)
            kind, value, tol = "usd", val * sc, unit_tol * sc
        else:
            sc = SCALE.get(suffix or "", 1.0)
            kind, value, tol = "count", val * sc, unit_tol * sc
        out.append(NumberMention(m.group(0).strip(), value, kind, tol + 1e-9, m.start()))
    return out


def _direction_near(text: str, nm: NumberMention) -> str | None:
    """Direction word that governs one number: first look just after it ("$1.3K below
    expected"), then back to the previous number ("increased ... to $723.8K, rising by $18.9K")."""
    nums = [m.start() for m in NUM_RE.finditer(text)]
    end = nm.start + len(nm.text) + 1
    nxt = min([p for p in nums if p > nm.start + 1], default=len(text))
    after = text[end: min(nxt, end + 24)]
    prev = max([p for p in nums if p < nm.start - 1], default=0)
    before = text[max(prev, nm.start - 60): nm.start]
    for window in (after, before):
        ups = [m.start() for m in UP_WORDS.finditer(window)]
        downs = [m.start() for m in DOWN_WORDS.finditer(window)]
        if ups or downs:
            if window is before:  # the word closest to the number wins
                return "up" if max(ups, default=-1) > max(downs, default=-1) else "down"
            return "up" if min(ups, default=10**6) < min(downs, default=10**6) else "down"
    return None


FROM_TO_RE = re.compile(r"(?P<word>\w+)\s+from\s+(?P<a>.{1,30}?)\s+(?:in [^,]{0,25}?\s+)?to\s+(?P<b>.{1,30})", re.I)


def _check_from_to(clean: str, sent: str) -> list[dict]:
    """'increased from A to B' must have B > A; 'decreased from A to B' must have B < A."""
    out = []
    for m in FROM_TO_RE.finditer(clean):
        word = m.group("word")
        up, down = bool(UP_WORDS.fullmatch(word)), bool(DOWN_WORDS.fullmatch(word))
        if not (up or down):
            continue
        a, b = _parse(m.group("a")), _parse(m.group("b"))
        if not a or not b or a[0].kind != b[0].kind:
            continue
        if up and b[0].value < a[0].value or down and b[0].value > a[0].value:
            out.append({"type": "direction", "sentence": sent, "number": f"{a[0].text} -> {b[0].text}",
                        "detail": f"'{word} from {a[0].text} to {b[0].text}' contradicts the numbers"})
    return out


def _unit_ok(kind: str, fact_unit: str) -> bool:
    return {"usd": {"usd"}, "pct": {"pct"}, "count": {"count", "day"}, "hours": {"hours"}}[kind].__contains__(fact_unit)


# Source re-runs are cached per (warehouse file, file mtime, tool, args): each distinct tool
# call is re-executed independently from the warehouse once per process, and any rebuild of
# the warehouse (new mtime) invalidates the cache.
_SOURCE_CACHE: dict[tuple, dict] = {}


def reconcile_facts(db_path: str, ledger: Ledger, fact_ids: set[str]) -> tuple[dict[str, float], list[dict]]:
    """Re-run each fact's tool on a fresh read-only connection and compare values."""
    import os

    issues: list[dict] = []
    recomputed: dict[str, float] = {}
    stamp = (db_path, os.path.getmtime(db_path))
    con = duckdb.connect(db_path, read_only=True)
    try:
        # resolve dependencies of derived facts first
        todo = set(fact_ids)
        for fid in list(todo):
            f = ledger.facts.get(fid)
            if f is not None and f.formula:
                todo |= set(re.findall(r"F\d+", f.formula))
        for fid in sorted(todo, key=lambda x: int(x[1:])):
            f = ledger.facts.get(fid)
            if f is None:
                continue
            if f.tool == "compute":
                val = eval_expression(f.formula, {**{k: v.value for k, v in ledger.facts.items()}, **recomputed})
            else:
                key = (*stamp, f.tool, repr(sorted(f.args.items())))
                if key not in _SOURCE_CACHE:
                    _SOURCE_CACHE[key] = TOOLS[f.tool](con, **f.args)
                val = float(get_path(_SOURCE_CACHE[key], f.key))
            recomputed[fid] = val
            if not math.isclose(val, f.value, rel_tol=1e-9, abs_tol=1e-6):
                issues.append({"type": "reconciliation", "fact": fid,
                               "detail": f"{fid} '{f.label}' is {f.value} in the ledger but {val} when re-run from source"})
    finally:
        con.close()
    return recomputed, issues


def audit(memo: str, ledger: Ledger, db_path: str | None = None) -> AuditReport:
    issues: list[dict] = []
    checked = 0
    cited_all: set[str] = set()
    for sent in _sentences(memo):
        cites = {c.strip() for grp in CITE_RE.findall(sent) for c in grp.split(",")}
        cited_all |= cites
        for c in cites:
            if c not in ledger.facts:
                issues.append({"type": "unknown_citation", "sentence": sent, "detail": f"{c} is not in the fact ledger"})
        clean = CITE_RE.sub("", sent)
        issues += _check_from_to(clean, sent)
        for nm in _parse(clean):
            checked += 1
            facts = [ledger.facts[c] for c in cites if c in ledger.facts]
            if not facts:
                issues.append({"type": "uncited_number", "sentence": sent, "number": nm.text,
                               "detail": f"'{nm.text}' has no fact citation in: \"{sent[:140]}\""})
                continue
            match = None
            for f in facts:
                if not _unit_ok(nm.kind, f.unit):
                    continue
                fv = abs(f.value) if nm.kind in ("usd", "pct") and nm.value >= 0 else f.value
                if abs(nm.value - fv) <= nm.tolerance:
                    match = f
                    break
            if match is None:
                shown = ", ".join(f"{f.id}={f.display()}" for f in facts)
                issues.append({"type": "value_mismatch", "sentence": sent, "number": nm.text,
                               "detail": f"'{nm.text}' does not match its cited fact(s) ({shown}) in: \"{sent[:140]}\""})
                continue
            is_change = re.match(r"(Change|Deviation|Forecast vs|Excess)", match.label) is not None
            if is_change and match.value != 0:
                word = _direction_near(clean, nm)
                if word == "up" and match.value < 0:
                    issues.append({"type": "direction", "sentence": sent, "number": nm.text,
                                   "detail": f"'{nm.text}' is described as an increase but {match.id} is negative"})
                if word == "down" and match.value > 0:
                    issues.append({"type": "direction", "sentence": sent, "number": nm.text,
                                   "detail": f"'{nm.text}' is described as a decrease but {match.id} is positive"})
    reconciled = 0
    if db_path:
        recomputed, rec_issues = reconcile_facts(db_path, ledger, {c for c in cited_all if c in ledger.facts})
        issues += rec_issues
        reconciled = len(recomputed)
    return AuditReport(passed=not issues, issues=issues, numbers_checked=checked, facts_reconciled=reconciled)
