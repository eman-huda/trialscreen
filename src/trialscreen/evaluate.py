"""Run criterion-level predictions over a split and compute the common metrics."""
from __future__ import annotations
import time
import torch
from .modeling import predict_one
from .metrics import evaluate_rows


@torch.no_grad()
def predict_records(tok, model, recs, mode="hf_cache", max_new_tokens=48, log=print, every=25):
    rows, t0 = [], time.time()
    model.eval()
    for i, r in enumerate(recs):
        out = predict_one(tok, model, r.note, r.criterion_type, r.criterion_text, mode=mode,
                          max_new_tokens=max_new_tokens)
        rows.append({"annotation_id": r.annotation_id, "patient_id": r.patient_id, "trial_id": r.trial_id,
                     "criterion_type": r.criterion_type, "criterion_text": r.criterion_text,
                     "gold_label": r.label, "pred_label": out["label"],
                     "gold_evidence": r.evidence, "pred_evidence": out["evidence"],
                     "valid_json": out["valid_json"], "raw_output": out["raw_output"],
                     "parse_notes": out["parse_notes"], "explanation": out["explanation"],
                     "prompt_tokens": out["prompt_tokens"], "latency_s": out["latency_s"]})
        if log and (i + 1) % every == 0:
            log(f"  {i + 1}/{len(recs)} done, {time.time() - t0:.0f}s")
    return rows


def evaluate_split(tok, model, recs, **kw):
    rows = predict_records(tok, model, recs, **kw)
    return rows, evaluate_rows(rows)
