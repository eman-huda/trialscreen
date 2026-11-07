"""Common evaluation metrics.

Classification: accuracy, macro-F1, per-class P/R/F1, confusion matrix; overall and
separately for inclusion and exclusion criteria. Macro-F1 averages over the classes
that occur in the gold labels of the evaluated subset (classes absent from the gold
subset are reported with support 0 but do not enter the macro average; a prediction of
an absent class still counts as an error through the other classes' precision).
Invalid outputs count as wrong predictions.

Evidence retrieval: sets of sentence IDs per record.
 * micro P/R/F1: TP/FP/FN summed over all records (empty sets contribute nothing).
 * macro P/R/F1 (non-empty gold): averaged over records whose gold set is non-empty;
   an empty prediction there gets P=0, R=0, F1=0.
 * empty-gold handling: records with an empty gold set are reported separately as
   'empty-gold agreement' = share of them where the model also cited nothing.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from .labels import ALL_LABELS, INCLUSION_LABELS, EXCLUSION_LABELS, INVALID


def classification_report(gold, pred, label_space=None):
    gold, pred = list(gold), list(pred)
    label_space = label_space or ALL_LABELS
    if not gold:
        return {"n": 0, "accuracy": None, "macro_f1": None, "macro_f1_classes": [], "per_class": {},
                "invalid_outputs": 0, "confusion_matrix": {"labels": label_space, "matrix": []}}
    present = [l for l in label_space if l in gold]
    cm_labels = label_space + ([INVALID] if INVALID in pred else [])
    p, r, f, s = precision_recall_fscore_support(gold, pred, labels=label_space, zero_division=0)
    per_class = {l: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                 for i, l in enumerate(label_space)}
    macro_f1 = float(np.mean([per_class[l]["f1"] for l in present])) if present else 0.0
    return {
        "n": len(gold),
        "accuracy": float(accuracy_score(gold, pred)) if gold else 0.0,
        "macro_f1": macro_f1,
        "macro_f1_classes": present,
        "per_class": per_class,
        "invalid_outputs": int(sum(x == INVALID for x in pred)),
        "confusion_matrix": {"labels": cm_labels,
                             "matrix": confusion_matrix(gold, pred, labels=cm_labels).tolist()},
    }


def classification_suite(rows):
    """rows: list of dicts with criterion_type, gold_label, pred_label."""
    out = {"overall": classification_report([r["gold_label"] for r in rows], [r["pred_label"] for r in rows])}
    for t, space in [("inclusion", INCLUSION_LABELS), ("exclusion", EXCLUSION_LABELS)]:
        sub = [r for r in rows if r["criterion_type"] == t]
        out[t] = classification_report([r["gold_label"] for r in sub], [r["pred_label"] for r in sub], space)
    return out


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def evidence_report(rows):
    """rows: dicts with gold_evidence, pred_evidence (lists of ints)."""
    TP = FP = FN = 0
    per = []
    empty_gold = empty_agree = 0
    for r in rows:
        g, p = set(r["gold_evidence"]), set(r["pred_evidence"])
        tp, fp, fn = len(g & p), len(p - g), len(g - p)
        TP, FP, FN = TP + tp, FP + fp, FN + fn
        if g:
            per.append(_prf(tp, fp, fn))
        else:
            empty_gold += 1
            empty_agree += int(not p)
    mp, mr, mf = _prf(TP, FP, FN)
    arr = np.array(per) if per else np.zeros((1, 3))
    return {
        "micro": {"precision": mp, "recall": mr, "f1": mf, "tp": TP, "fp": FP, "fn": FN},
        "macro_nonempty_gold": {"precision": float(arr[:, 0].mean()), "recall": float(arr[:, 1].mean()),
                                "f1": float(arr[:, 2].mean()), "n_records": len(per)},
        "empty_gold": {"n_records": empty_gold,
                       "model_also_empty": empty_agree,
                       "agreement_rate": (empty_agree / empty_gold) if empty_gold else None},
    }


def evaluate_rows(rows):
    res = {"classification": classification_suite(rows), "evidence": {"overall": evidence_report(rows)}}
    for t in ("inclusion", "exclusion"):
        res["evidence"][t] = evidence_report([r for r in rows if r["criterion_type"] == t])
    res["valid_json_rate"] = float(np.mean([r.get("valid_json", True) for r in rows])) if rows else 0.0
    return res
