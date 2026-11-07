"""Before/after evaluation with identical examples and identical decoding settings
(greedy, max_new_tokens=48, same prompt) for the base and the fine-tuned model.

  python scripts/03_evaluate.py --split validation          # allowed any time
  python scripts/03_evaluate.py --split test --final        # run ONCE, after all tuning is frozen
"""
import argparse, json, time
from _common import ROOT, RESULTS, dump, env_info
from trialscreen.data import load_jsonl
from trialscreen.modeling import load_model, TARGET_ID
from trialscreen.evaluate import evaluate_split

ap = argparse.ArgumentParser()
ap.add_argument("--split", choices=["validation", "test"], default="validation")
ap.add_argument("--final", action="store_true", help="required to touch the test split")
ap.add_argument("--adapter", default=str(ROOT / "checkpoints" / "qwen2.5-1.5b-trialscreen-lora" / "best_adapter"))
ap.add_argument("--models", default="base,finetuned")
ap.add_argument("--model-id", default=TARGET_ID)
ap.add_argument("--limit", type=int, default=None, help="smoke tests only")
args = ap.parse_args()
if args.split == "test" and not args.final:
    raise SystemExit("Refusing to evaluate on the test split without --final (no tuning on test).")
recs = load_jsonl(ROOT / "data" / "splits" / f"{args.split}.jsonl")[: args.limit]
settings = {"decoding": "greedy (argmax), HF forward + explicit DynamicCache loop", "max_new_tokens": 48,
            "prompt": "trialscreen.data.build_messages + Qwen chat template", "env": env_info()}
for name in args.models.split(","):
    t0 = time.time()
    tok, model = load_model(args.model_id, adapter=args.adapter if name == "finetuned" else None)
    rows, metrics = evaluate_split(tok, model, recs)
    metrics["settings"] = settings | {"model": args.model_id, "adapter": args.adapter if name == "finetuned" else None,
                                      "minutes": (time.time() - t0) / 60}
    with open(RESULTS / f"predictions_{name}_{args.split}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    dump(metrics, RESULTS / f"metrics_{name}_{args.split}.json")
    c = metrics["classification"]["overall"]
    print(f"{name}/{args.split}: acc {c['accuracy']:.3f} macroF1 {c['macro_f1']:.3f} "
          f"evid microF1 {metrics['evidence']['overall']['micro']['f1']:.3f} valid {metrics['valid_json_rate']:.3f}")
    del model
    import torch, gc; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
