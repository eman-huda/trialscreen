"""Robust parsing of the model's JSON answer into (label, evidence ids)."""
from __future__ import annotations
import json, re
from .labels import labels_for, INVALID

_JSON_RE = re.compile(r"\{.*?\}", re.S)


def parse_output(text: str, criterion_type: str, n_sentences: int) -> dict:
    """Returns dict(label, evidence, valid_json, notes). Never invents a label:
    unparseable output becomes INVALID and is counted as an error in evaluation."""
    allowed = labels_for(criterion_type)
    notes = []
    obj = None
    for m in _JSON_RE.finditer(text):
        try:
            obj = json.loads(m.group(0))
            break
        except json.JSONDecodeError:
            continue
    label, evidence, valid = INVALID, [], obj is not None
    if obj is not None:
        lab = str(obj.get("label", "")).strip().lower()
        if lab in allowed:
            label = lab
        else:
            notes.append(f"label '{lab}' not allowed for {criterion_type}")
        ev = obj.get("evidence", [])
        if isinstance(ev, list):
            for e in ev:
                try:
                    e = int(e)
                except (TypeError, ValueError):
                    notes.append(f"non-integer evidence id {e!r} dropped")
                    continue
                if 0 <= e < n_sentences:
                    evidence.append(e)
                else:
                    notes.append(f"out-of-range evidence id {e} dropped")
    else:
        # last resort: an exact allowed label string in free text (longest first so
        # 'not included' is not read as 'included'); still flagged as invalid JSON
        for lab in sorted(allowed, key=len, reverse=True):
            if re.search(r"\b" + re.escape(lab) + r"\b", text.lower()):
                label = lab
                notes.append("label recovered from non-JSON text")
                break
    return {"label": label, "evidence": sorted(set(evidence)), "valid_json": valid, "notes": notes}
