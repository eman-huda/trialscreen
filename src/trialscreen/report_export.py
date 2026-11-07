"""Readable, self-contained HTML screening report with provenance."""
from __future__ import annotations
import hashlib, html, json, datetime
from . import __version__

DISCLAIMER = ("Educational screening prototype for a university assignment. It is NOT a clinical "
              "enrollment decision tool and must not be used to include or exclude real patients.")

_COL = {"included": "#1b7350", "not excluded": "#1b7350", "not included": "#b42318", "excluded": "#b42318",
        "not enough information": "#8a5a00", "not applicable": "#555", "invalid output": "#b42318"}


def provenance(note, criteria, settings):
    blob = json.dumps({"note": note, "criteria": criteria}, sort_keys=True).encode()
    return {"generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "app_version": __version__, "input_sha256": hashlib.sha256(blob).hexdigest(), **settings}


def html_report(trial_name, sentences, results, verdict, prov):
    e = html.escape
    rows = []
    for i, r in enumerate(results, 1):
        ev = "<br>".join(f"<b>[{sid}]</b> {e(t)}" for sid, t in zip(r["evidence"], r["evidence_text"])) or "<i>none cited</i>"
        rows.append(f"<tr><td>{i}</td><td>{e(r['criterion_type'])}</td><td>{e(r['criterion_text'])}</td>"
                    f"<td style='color:{_COL.get(r['label'], '#000')};font-weight:700'>{e(r['label'])}</td>"
                    f"<td>{ev}</td><td>{e(r['explanation'])}</td></tr>")
    note = "".join(f"<li value='{i}'>{e(s)}</li>" for i, s in sentences)
    missing = [r for r in results if r["label"] == "not enough information"]
    miss = "".join(f"<li>{e(r['criterion_text'])}</li>" for r in missing) or "<li>None</li>"
    return f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Eligibility screening report</title>
<style>body{{font-family:system-ui,Arial,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;color:#1c1a33;line-height:1.5}}
.warn{{background:#fff4e5;border:2px solid #8a5a00;border-radius:10px;padding:10px 14px}}
table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border:1px solid #ccd;padding:6px;vertical-align:top;text-align:left}}
th{{background:#eef2fb}}ol li{{margin-bottom:4px}}code{{font-size:12px}}</style>
<h1>Eligibility screening report</h1><p class="warn"><b>Notice:</b> {e(DISCLAIMER)}</p>
<h2>Trial</h2><p>{e(trial_name or 'Unnamed trial')}</p>
<h2>Preliminary outcome (rule-based aggregation)</h2><p><b>{e(verdict['verdict'])}</b> &mdash; {verdict['n_criteria']} criteria,
{verdict['n_blocking']} blocking, {verdict['n_unknown']} undetermined.<br><small>Rule: {e(verdict['rule'])} This aggregation is not a
ground-truth label; the dataset only labels individual criteria.</small></p>
<h2>Missing information</h2><ul>{miss}</ul>
<h2>Criterion-level results</h2><table><tr><th>#</th><th>Type</th><th>Criterion</th><th>Prediction</th><th>Evidence (sentence ID)</th><th>Explanation</th></tr>
{''.join(rows)}</table>
<h2>Patient record (sentence IDs)</h2><ol start="0">{note}</ol>
<h2>Provenance</h2><pre><code>{e(json.dumps(prov, indent=1))}</code></pre></html>"""
