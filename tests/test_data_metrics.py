import sys, pathlib, json
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trialscreen.data import load_jsonl, split_sentences, LEAKY_COLUMNS
from trialscreen.parsing import parse_output
from trialscreen.metrics import evidence_report, classification_report
from trialscreen.criteria import parse_criteria
from trialscreen.explain import aggregate_trial

SPLITS = ROOT / "data" / "splits"


def test_split_manifest_and_no_patient_leakage():
    man = json.load(open(SPLITS / "split_manifest.json"))
    ids = {k: set(v["patient_ids"]) for k, v in man["splits"].items()}
    assert [len(ids[s]) for s in ("train", "validation", "test")] == [37, 8, 8]
    assert not (ids["train"] & ids["validation"]) and not (ids["train"] & ids["test"]) and not (ids["validation"] & ids["test"])
    for s in ("train", "validation", "test"):
        recs = load_jsonl(SPLITS / f"{s}.jsonl")
        assert {r.patient_id for r in recs} == ids[s]
        assert len(recs) == man["splits"][s]["n_annotations"]
    assert set(man["dropped_columns"]) == set(LEAKY_COLUMNS)


def test_split_reproduces_from_seed():
    import numpy as np
    man = json.load(open(SPLITS / "split_manifest.json"))
    all_ids = sorted(set().union(*[v["patient_ids"] for v in man["splits"].values()]))
    a = np.array(all_ids); np.random.default_rng(42).shuffle(a)
    assert a[:37].tolist() == man["splits"]["train"]["patient_ids"]
    assert a[45:].tolist() == man["splits"]["test"]["patient_ids"]


def test_no_gpt4_fields_in_records():
    rec = load_jsonl(SPLITS / "train.jsonl")[0].to_dict()
    assert not any(k.startswith("gpt4") or k == "explanation_correctness" for k in rec)


def test_evidence_ids_within_note():
    for s in ("train", "validation", "test"):
        for r in load_jsonl(SPLITS / f"{s}.jsonl"):
            assert all(0 <= i < r.n_sentences for i in r.evidence)


def test_sentence_split_numbered_and_free_text():
    assert split_sentences("0. A man.\n1. He has X.") == [(0, "A man."), (1, "He has X.")]
    assert [i for i, _ in split_sentences("A 50-year-old man. He has diabetes! Is he pregnant? No.")] == [0, 1, 2, 3]


def test_parse_output():
    assert parse_output('{"evidence": [1, 9, "x"], "label": "included"}', "inclusion", 3)["evidence"] == [1]
    assert parse_output('{"evidence": [], "label": "excluded"}', "inclusion", 3)["label"] == "invalid output"
    assert parse_output("the answer is not included", "inclusion", 3)["label"] == "not included"
    assert parse_output("garbage", "exclusion", 3)["label"] == "invalid output"


def test_evidence_metrics_empty_handling():
    rows = [{"gold_evidence": [1, 2], "pred_evidence": [2, 3]},
            {"gold_evidence": [], "pred_evidence": []},
            {"gold_evidence": [], "pred_evidence": [4]},
            {"gold_evidence": [0], "pred_evidence": []}]
    r = evidence_report(rows)
    assert (r["micro"]["tp"], r["micro"]["fp"], r["micro"]["fn"]) == (1, 2, 2)
    assert r["macro_nonempty_gold"]["n_records"] == 2
    assert r["empty_gold"] == {"n_records": 2, "model_also_empty": 1, "agreement_rate": 0.5}


def test_macro_f1_only_over_present_classes():
    r = classification_report(["included", "included", "not enough information"],
                              ["included", "not included", "not enough information"],
                              ["included", "not included", "not enough information", "not applicable"])
    assert r["macro_f1_classes"] == ["included", "not enough information"]
    assert abs(r["macro_f1"] - (2 / 3 + 1) / 2) < 1e-9


def test_criteria_parser_and_aggregation():
    crits, problems = parse_criteria("Inclusion criteria:\n- Age > 18\n2) Diabetes\nExclusion criteria:\n* Pregnant\nE: Smoker")
    assert crits == [("inclusion", "Age > 18"), ("inclusion", "Diabetes"), ("exclusion", "Pregnant"), ("exclusion", "Smoker")]
    assert not problems
    assert parse_criteria("no heading here")[1]
    res = [{"criterion_type": "inclusion", "label": "included"}, {"criterion_type": "exclusion", "label": "not excluded"}]
    assert aggregate_trial(res)["verdict"] == "potentially eligible"
    assert aggregate_trial(res + [{"criterion_type": "exclusion", "label": "not enough information"}])["verdict"].startswith("undetermined")
    assert aggregate_trial(res + [{"criterion_type": "exclusion", "label": "excluded"}])["verdict"] == "likely ineligible"
