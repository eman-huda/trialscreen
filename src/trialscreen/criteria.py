"""Parse a free-text eligibility-criteria block into (type, text) pairs."""
from __future__ import annotations
import re

_HEAD = re.compile(r"^\s*(inclusion|exclusion)(\s+criteria)?\s*:?\s*$", re.I)
_INLINE = re.compile(r"^\s*(inc|inclusion|exc|exclusion)\s*[:\-]\s*(.+)$", re.I)
_BULLET = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s*")


def parse_criteria(block: str):
    """Accepts either section headers ('Inclusion criteria:' / 'Exclusion criteria:') followed by
    one criterion per line (bullets or numbers optional), or per-line prefixes 'I: ...' / 'E: ...'.
    Returns (criteria, problems)."""
    out, problems, section = [], [], None
    for n, raw in enumerate(block.replace("\r", "").split("\n"), 1):
        line = raw.strip()
        if not line:
            continue
        m = _HEAD.match(line)
        if m:
            section = m.group(1).lower(); continue
        m = _INLINE.match(line)
        if m and not _HEAD.match(line):
            t = "inclusion" if m.group(1).lower().startswith("inc") else "exclusion"
            out.append((t, _BULLET.sub("", m.group(2)).strip())); continue
        m = re.match(r"^\s*([IE])\s*:\s*(.+)$", line)
        if m:
            out.append(("inclusion" if m.group(1) == "I" else "exclusion", m.group(2).strip())); continue
        text = _BULLET.sub("", line).strip()
        if section is None:
            problems.append(f"line {n} has no section: add 'Inclusion criteria:' or 'Exclusion criteria:' above it")
            continue
        if len(text) < 3:
            problems.append(f"line {n} is too short to be a criterion"); continue
        out.append((section, text))
    return out, problems
