"""Download the TrialGPT criterion annotations, drop GPT-4 columns, make the fixed
37/8/8 patient split (seed 42), check leakage and write JSONL files + split manifest."""
from _common import ROOT, dump
from trialscreen.data import (load_raw, drop_leaky, make_split, check_no_leakage, to_records,
                              save_jsonl, file_sha256, invalid_annotations, DATASET_ID, LEAKY_COLUMNS)

raw = load_raw(ROOT / "data" / "raw")
parquet = ROOT / "data" / "raw" / "data" / "train-00000-of-00001.parquet"
print("raw shape", raw.shape)
df = drop_leaky(raw)
split = make_split(df)
check_no_leakage(split, df)
manifest = {"dataset": DATASET_ID, "parquet_sha256": file_sha256(parquet), "seed": 42,
            "procedure": "sorted unique patient_id -> numpy.random.default_rng(42).shuffle -> first 37 train, next 8 validation, last 8 test",
            "dropped_columns": LEAKY_COLUMNS,
            "excluded_annotations": {"ids": invalid_annotations(df),
                                     "reason": "criterion_text missing (NaN) in the released parquet; cannot form an input. "
                                               "The patient stays in its split; only this annotation is unusable."},
            "splits": {}}
for name, ids in split.items():
    sub = df[df.patient_id.isin(ids)]
    recs = to_records(sub.sort_values("annotation_id"))
    save_jsonl(recs, ROOT / "data" / "splits" / f"{name}.jsonl")
    manifest["splits"][name] = {
        "patient_ids": ids, "n_patients": len(ids), "n_annotations_raw": len(sub), "n_annotations": len(recs),
        "annotations_per_patient": sub.groupby("patient_id").size().to_dict(),
        "label_counts": {t: sub[sub.criterion_type == t].expert_eligibility.value_counts().to_dict()
                         for t in ("inclusion", "exclusion")},
        "empty_evidence": int((sub.expert_sentences == "[]").sum()),
    }
    print(name, len(ids), "patients", len(sub), "annotations")
assert sum(v["n_annotations_raw"] for v in manifest["splits"].values()) == len(df)
assert sum(v["n_annotations"] for v in manifest["splits"].values()) == len(df) - len(manifest["excluded_annotations"]["ids"])
dump(manifest, ROOT / "data" / "splits" / "split_manifest.json")
