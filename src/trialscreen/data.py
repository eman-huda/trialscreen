"""Dataset loading, the prescribed patient-grouped split, sentence parsing and prompts.

Only these columns are ever used as model input: note, criterion_type, criterion_text.
Targets: expert_eligibility, expert_sentences.
Columns produced by GPT-4 (gpt4_explanation, gpt4_sentences, gpt4_eligibility,
explanation_correctness) are dropped immediately after loading so they cannot leak.
"""
from __future__ import annotations
import json, re, hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd

from .labels import labels_for

DATASET_ID = "ncbi/TrialGPT-Criterion-Annotations"
LEAKY_COLUMNS = ["gpt4_explanation", "gpt4_sentences", "gpt4_eligibility", "explanation_correctness"]
SEED = 42
N_TRAIN, N_VAL, N_TEST = 37, 8, 8
SENT_RE = re.compile(r"^\s*(\d+)\.\s+(.*)$")


def load_raw(cache_dir: str | Path = "data/raw") -> pd.DataFrame:
    """Download the single parquet file of the dataset (public domain) and load it."""
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(DATASET_ID, "data/train-00000-of-00001.parquet",
                        repo_type="dataset", local_dir=str(cache_dir))
    return pd.read_parquet(p)


def drop_leaky(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in LEAKY_COLUMNS if c in df.columns])


def make_split(df: pd.DataFrame) -> dict[str, list[str]]:
    """Prescribed split: sort unique patient IDs, shuffle with default_rng(42),
    first 37 train, next 8 validation, remaining 8 test."""
    ids = np.array(sorted(df["patient_id"].unique()))
    if len(ids) != N_TRAIN + N_VAL + N_TEST:
        raise ValueError(f"expected 53 patients, found {len(ids)}")
    rng = np.random.default_rng(SEED)
    rng.shuffle(ids)
    return {"train": ids[:N_TRAIN].tolist(),
            "validation": ids[N_TRAIN:N_TRAIN + N_VAL].tolist(),
            "test": ids[N_TRAIN + N_VAL:].tolist()}


def check_no_leakage(split: dict[str, list[str]], df: pd.DataFrame) -> None:
    s = {k: set(v) for k, v in split.items()}
    assert not (s["train"] & s["validation"]), "train/validation patient overlap"
    assert not (s["train"] & s["test"]), "train/test patient overlap"
    assert not (s["validation"] & s["test"]), "validation/test patient overlap"
    assert set().union(*s.values()) == set(df["patient_id"]), "split does not cover all patients"
    for col in LEAKY_COLUMNS:
        assert col not in df.columns, f"leaky column {col} still present"


def split_sentences(note: str) -> list[tuple[int, str]]:
    """Return [(sentence_id, text)]. Dataset notes are already numbered '0. ...'.
    Unnumbered free text (from the app) is split into sentences and numbered from 0."""
    lines = [l for l in note.replace("\r", "").split("\n") if l.strip()]
    parsed = [SENT_RE.match(l) for l in lines]
    if lines and all(parsed):
        out = [(int(m.group(1)), m.group(2).strip()) for m in parsed]
        if [i for i, _ in out] == list(range(len(out))):
            return out
    text = " ".join(l.strip() for l in lines)
    # transparent splitter: break after . ! ? when followed by space and a capital/digit
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])", text)
    parts = [p.strip() for p in parts if p.strip()]
    return list(enumerate(parts))


def numbered_note(sentences: list[tuple[int, str]]) -> str:
    return "\n".join(f"{i}. {s}" for i, s in sentences)


SYSTEM_PROMPT = (
    "You are a clinical-trial eligibility screening assistant used for research. "
    "Judge one eligibility criterion against one patient note. Use only facts stated in the note. "
    "Absence of a fact is not evidence for or against eligibility: if the note does not say enough, "
    "answer 'not enough information'. Answer with one JSON object and nothing else."
)


def build_user_prompt(note_numbered: str, criterion_type: str, criterion_text: str) -> str:
    labels = labels_for(criterion_type)
    return (
        f"Patient note (each sentence has an ID):\n{note_numbered}\n\n"
        f"Criterion type: {criterion_type.lower()}\n"
        f"Criterion: {criterion_text.strip()}\n\n"
        f"Allowed labels: {', '.join(labels)}\n"
        'Return JSON: {"evidence": [sentence IDs that support the decision], "label": "<one allowed label>"}'
    )


def build_messages(note_numbered: str, criterion_type: str, criterion_text: str):
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(note_numbered, criterion_type, criterion_text)}]


def target_json(label: str, evidence: list[int]) -> str:
    # evidence first, label second: the model commits to cited sentences before the decision
    return json.dumps({"evidence": sorted(set(int(i) for i in evidence)), "label": label})


@dataclass
class Record:
    annotation_id: int
    patient_id: str
    trial_id: str
    criterion_type: str
    criterion_text: str
    note: str               # numbered note
    n_sentences: int
    label: str              # expert_eligibility
    evidence: list          # expert_sentences

    def to_dict(self):
        return asdict(self)


def invalid_annotations(df: pd.DataFrame) -> list[int]:
    """Annotations that cannot form a valid input (missing/empty criterion text).
    In the released dataset this is exactly one row: annotation 883 (train patient sigir-201519,
    exclusion, expert label 'not excluded', no evidence)."""
    bad = df["criterion_text"].map(lambda x: not isinstance(x, str) or not x.strip())
    return df.loc[bad, "annotation_id"].astype(int).tolist()


def to_records(df: pd.DataFrame) -> list[Record]:
    """Skips annotations without criterion text (see invalid_annotations); callers record them."""
    recs = []
    bad = set(invalid_annotations(df))
    for r in df.itertuples(index=False):
        if int(r.annotation_id) in bad:
            continue
        sents = split_sentences(r.note)
        recs.append(Record(int(r.annotation_id), r.patient_id, r.trial_id, r.criterion_type.lower(),
                           r.criterion_text, numbered_note(sents), len(sents),
                           r.expert_eligibility, json.loads(r.expert_sentences)))
    return recs


def save_jsonl(recs, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r.to_dict() if hasattr(r, "to_dict") else r) + "\n")


def load_jsonl(path) -> list[Record]:
    with open(path) as f:
        return [Record(**json.loads(l)) for l in f if l.strip()]


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()
