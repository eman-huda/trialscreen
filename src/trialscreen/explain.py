"""Traceable explanations and the explicit trial-level aggregation rule.

Explanations are built only from the predicted label and the exact cited sentences, so
every statement in an explanation points to text that exists in the patient record. The
model is not asked to write free-text rationales: the dataset's only free-text
rationales are GPT-4 outputs, which the assignment forbids as inputs or targets.
"""
from __future__ import annotations
from .labels import LABEL_MEANING, INVALID


def explain(criterion_type, criterion_text, label, evidence_ids, sentences: dict[int, str]):
    if label == INVALID:
        return "The model output could not be parsed, so no decision is made for this criterion."
    cited = [f'[{i}] "{sentences[i]}"' for i in evidence_ids if i in sentences]
    base = f"Prediction '{label}': {LABEL_MEANING.get(label, label)}."
    if label == "not enough information":
        if cited:
            return base + " Related but insufficient sentences: " + "; ".join(cited) + "."
        return base + " No sentence in the note addresses this criterion; missing information is not treated as proof either way."
    if not cited:
        return base + " Warning: no supporting sentence was cited, so this decision is unsupported and should be checked."
    return base + " Supporting sentences: " + "; ".join(cited) + "."


def aggregate_trial(results):
    """Rule-based preliminary trial verdict from criterion-level predictions.
    This is NOT a ground-truth-labelled outcome; the dataset only labels criteria.

    Rules (applied in order):
      1. any exclusion predicted 'excluded' or inclusion predicted 'not included'
         -> 'likely ineligible'
      2. any 'not enough information' or invalid output -> 'undetermined: more information needed'
      3. otherwise (all inclusion 'included'/'not applicable', all exclusion
         'not excluded'/'not applicable') -> 'potentially eligible'
    """
    blockers = [r for r in results if (r["criterion_type"] == "exclusion" and r["label"] == "excluded")
                or (r["criterion_type"] == "inclusion" and r["label"] == "not included")]
    unknown = [r for r in results if r["label"] in ("not enough information", INVALID)]
    if blockers:
        verdict = "likely ineligible"
    elif unknown:
        verdict = "undetermined: more information needed"
    else:
        verdict = "potentially eligible"
    return {"verdict": verdict, "n_criteria": len(results), "n_blocking": len(blockers),
            "n_unknown": len(unknown),
            "rule": "exclusion 'excluded' or inclusion 'not included' blocks; any 'not enough information' "
                    "leaves the outcome undetermined; otherwise potentially eligible."}
