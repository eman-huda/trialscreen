"""Fine-tune Qwen2.5-1.5B-Instruct with LoRA on the TRAIN split. Model selection uses
VALIDATION macro-F1 only; the test split is never loaded here.
Usage: python scripts/02_train_lora.py [--config configs/lora_qwen2.5_1.5b.json] [--epochs N]"""
import argparse, json
from _common import ROOT, RESULTS, dump, env_info
from trialscreen.data import load_jsonl
from trialscreen.train import train_lora
from trialscreen.evaluate import predict_records
from trialscreen.metrics import evaluate_rows

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=str(ROOT / "configs" / "lora_qwen2.5_1.5b.json"))
ap.add_argument("--epochs", type=int, default=None)
ap.add_argument("--out", default=str(ROOT / "checkpoints" / "qwen2.5-1.5b-trialscreen-lora"))
args = ap.parse_args()
cfg = json.load(open(args.config))
if args.epochs:
    cfg["epochs"] = args.epochs
train = load_jsonl(ROOT / "data" / "splits" / "train.jsonl")
val = load_jsonl(ROOT / "data" / "splits" / "validation.jsonl")
logf = open(RESULTS / "train_log.txt", "w")


def log(msg):
    print(msg, flush=True); logf.write(msg + "\n"); logf.flush()


def val_eval_fn(model, tok):
    model.eval()
    rows = predict_records(tok, model, val, log=None)
    m = evaluate_rows(rows)
    model.train()
    return {"val_accuracy": m["classification"]["overall"]["accuracy"],
            "val_macro_f1": m["classification"]["overall"]["macro_f1"],
            "val_evidence_micro_f1": m["evidence"]["overall"]["micro"]["f1"],
            "val_valid_json_rate": m["valid_json_rate"]}


log(json.dumps(env_info()))
best, history, meta = train_lora(cfg, train, val, args.out, log=log, val_eval_fn=val_eval_fn)
dump(meta, RESULTS / "train_meta.json")
dump(history, RESULTS / "train_history.json")
log(f"best adapter: {best}")
