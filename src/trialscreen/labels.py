"""Label sets and output schema.

The TrialGPT criterion annotations use different label vocabularies for inclusion and
exclusion criteria. We keep all of them (no binary collapse):

inclusion: included | not included | not enough information | not applicable
exclusion: excluded | not excluded | not enough information | not applicable
"""
INCLUSION_LABELS = ["included", "not included", "not enough information", "not applicable"]
EXCLUSION_LABELS = ["excluded", "not excluded", "not enough information", "not applicable"]
ALL_LABELS = ["included", "not included", "excluded", "not excluded",
              "not enough information", "not applicable"]
INVALID = "invalid output"


def labels_for(criterion_type: str):
    t = criterion_type.strip().lower()
    if t == "inclusion":
        return INCLUSION_LABELS
    if t == "exclusion":
        return EXCLUSION_LABELS
    raise ValueError(f"criterion_type must be 'inclusion' or 'exclusion', got {criterion_type!r}")


# Short human meaning of each label, used for template explanations in the app/report.
LABEL_MEANING = {
    "included": "the patient record shows the inclusion criterion is met",
    "not included": "the patient record shows the inclusion criterion is not met",
    "excluded": "the patient record shows the exclusion criterion applies, so the patient would be excluded",
    "not excluded": "the patient record shows the exclusion condition is absent",
    "not enough information": "the record does not contain enough information to decide",
    "not applicable": "the criterion does not apply to this patient",
}
