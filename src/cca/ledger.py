"""Fact ledger: every number the agent is allowed to say lives here, with its provenance.

Think of it as the general ledger of the analysis. Tools post facts (F1, F2, ...);
the writer may only quote posted facts and must cite them; the auditor reconciles
each cited fact back to its source -- the same discipline as reconciling a
financial report to the transaction system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

UNITS = {"usd", "pct", "count", "hours", "day", "ratio"}


@dataclass
class Fact:
    id: str
    label: str
    value: float
    unit: str
    tool: str
    args: dict
    key: str                  # how to find this value again in the tool's output
    formula: str | None = None  # for derived facts, e.g. "F3 - F2"

    def display(self) -> str:
        return format_value(self.value, self.unit)


def format_value(v: float, unit: str) -> str:
    if unit == "usd":
        a = abs(v)
        sign = "-" if v < 0 else ""
        if a >= 1e6:
            return f"{sign}${a / 1e6:,.2f}M"
        if a >= 1e3:
            return f"{sign}${a / 1e3:,.1f}K"
        return f"{sign}${a:,.2f}"
    if unit == "pct":
        return f"{v * 100:.1f}%"
    if unit in ("count", "day"):
        return f"{v:,.0f}"
    if unit == "hours":
        return f"{v:,.1f} h"
    return f"{v:,.4g}"


@dataclass
class Ledger:
    facts: dict[str, Fact] = field(default_factory=dict)

    def post(self, label: str, value: Any, unit: str, tool: str, args: dict, key: str,
             formula: str | None = None) -> str:
        if unit not in UNITS:
            raise ValueError(f"unknown unit {unit}")
        fid = f"F{len(self.facts) + 1}"
        self.facts[fid] = Fact(fid, label, float(value), unit, tool, dict(args), key, formula)
        return fid

    def table(self) -> str:
        """Compact fact table handed to the writer LLM."""
        lines = ["id | label | value | unit"]
        for f in self.facts.values():
            lines.append(f"{f.id} | {f.label} | {f.display()} | {f.unit}")
        return "\n".join(lines)

    def as_records(self) -> list[dict]:
        return [{"id": f.id, "label": f.label, "value": f.value, "display": f.display(),
                 "unit": f.unit, "tool": f.tool, "args": f.args, "formula": f.formula}
                for f in self.facts.values()]
