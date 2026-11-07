"""Task 5: greedy speculative decoding vs ordinary greedy target decoding.

Target: Qwen2.5-1.5B-Instruct (+ merged LoRA adapter). Draft: Qwen2.5-0.5B-Instruct (same
151,665-token vocabulary, checked below). Both run through our explicit decoders with
explicit KV caches. Two workloads, all prompts from the VALIDATION split:
  * screening: the real criterion-level JSON task (short outputs, EOS respected)
  * summary:   a free-text summary with a fixed 128-token output (EOS ignored)
For each lookahead k we report token equality with greedy target decoding, acceptance
rate, tokens per round, latency, tokens/s, speedup and peak memory. As an external
reference only, Hugging Face assisted generation is also run and its tokens compared."""
import torch
from _common import ROOT, RESULTS, dump, env_info
from _bench_common import base_args, long_prompt_ids, median
from trialscreen.data import load_jsonl
from trialscreen.modeling import load_model, prompt_ids, eos_ids
from trialscreen.decoding import ExplicitQwen2Decoder, greedy_cached, first_mismatch
from trialscreen.speculative import speculative_greedy

ap = base_args("speculative decoding benchmark")
ap.add_argument("--draft", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--draft-adapter", default="none")
ap.add_argument("--ks", default="2,4,6")
ap.add_argument("--n-screening", type=int, default=20)
args = ap.parse_args()
if args.quick:
    args.ks, args.n_screening, args.repeats = "2,4", 3, 1
adapter = None if args.adapter == "none" else args.adapter
tok, target = load_model(args.model, adapter=adapter)
dtok, draft = load_model(args.draft, adapter=None if args.draft_adapter == "none" else args.draft_adapter,
                         device=str(target.device), dtype=next(target.parameters()).dtype)
vocab_equal = tok.get_vocab() == dtok.get_vocab()
assert vocab_equal, "draft and target tokenizers differ: speculative decoding would be invalid"
dev = target.device
EOS = eos_ids(tok)
T = ExplicitQwen2Decoder(target, "torch_sdpa", max_len=4096)
D = ExplicitQwen2Decoder(draft, "torch_sdpa", max_len=4096)

val = load_jsonl(ROOT / "data" / "splits" / "validation.jsonl")
workloads = {
    "screening": [(torch.tensor([prompt_ids(tok, r.note, r.criterion_type, r.criterion_text)], device=dev), 48, EOS)
                  for r in val[: args.n_screening]],
    "summary": [(long_prompt_ids(tok, n).to(dev), 32 if args.quick else 128, ()) for n in ([64] if args.quick else [256, 512])],
}


def run_greedy(ids, n, eos):
    return greedy_cached(T, ids, n, eos_ids=eos)


out = {"env": env_info(), "target": args.model, "target_adapter": adapter, "draft": args.draft,
       "draft_adapter": args.draft_adapter, "vocab_equal": vocab_equal, "dtype": str(next(target.parameters()).dtype),
       "workloads": {}}
for wname, items in workloads.items():
    # warm-up
    run_greedy(*items[0]); speculative_greedy(T, D, items[0][0], 8, k=2, eos_ids=items[0][2])
    base = []
    for ids, n, eos in items:
        runs = [run_greedy(ids, n, eos) for _ in range(args.repeats)]
        base.append({"tokens": runs[0].tokens, "s": median([r.total_s for r in runs]),
                     "mem": max(r.peak_mem_mb or 0 for r in runs) or None})
    w = {"n_prompts": len(items), "greedy": {"total_s": sum(b["s"] for b in base),
                                             "new_tokens": sum(len(b["tokens"]) for b in base),
                                             "peak_mem_mb": max((b["mem"] or 0) for b in base) or None}, "speculative": {}}
    w["greedy"]["tokens_per_s"] = w["greedy"]["new_tokens"] / w["greedy"]["total_s"]
    for k in map(int, args.ks.split(",")):
        agg = {"equal": 0, "mismatches": [], "drafted": 0, "accepted": 0, "rounds": 0, "total_s": 0.0,
               "new_tokens": 0, "peak_mem_mb": 0, "target_forwards": 0, "draft_forwards": 0}
        for (ids, n, eos), b in zip(items, base):
            runs = [speculative_greedy(T, D, ids, n, k=k, eos_ids=eos) for _ in range(args.repeats)]
            s = runs[0]
            mm = first_mismatch(b["tokens"], s.tokens)
            agg["equal"] += int(mm is None)
            if mm is not None:
                agg["mismatches"].append({"position": mm})
            agg["drafted"] += s.drafted; agg["accepted"] += s.accepted; agg["rounds"] += s.rounds
            agg["new_tokens"] += len(s.tokens); agg["total_s"] += median([r.total_s for r in runs])
            agg["peak_mem_mb"] = max(agg["peak_mem_mb"], max(r.peak_mem_mb or 0 for r in runs))
            agg["target_forwards"] += s.target_forwards; agg["draft_forwards"] += s.draft_forwards
        agg["acceptance_rate"] = agg["accepted"] / max(agg["drafted"], 1)
        agg["tokens_per_round"] = agg["new_tokens"] / max(agg["rounds"], 1)
        agg["tokens_per_s"] = agg["new_tokens"] / agg["total_s"]
        agg["speedup_vs_greedy"] = w["greedy"]["total_s"] / agg["total_s"]
        agg["token_equal_rate"] = agg["equal"] / len(items)
        w["speculative"][str(k)] = agg
        print(wname, "k", k, {x: (round(y, 3) if isinstance(y, float) else y) for x, y in agg.items() if x != "mismatches"}, flush=True)
    # external reference: HF assisted generation (not our implementation)
    try:
        ids, n, eos = items[0]
        g = target.generate(ids, attention_mask=torch.ones_like(ids), assistant_model=draft, do_sample=False,
                            max_new_tokens=n, eos_token_id=list(eos) or None, min_new_tokens=n if not eos else 0)
        hf_toks = g[0, ids.shape[1]:].tolist()
        w["hf_assisted_reference_equal_first_prompt"] = first_mismatch(base[0]["tokens"], hf_toks) is None
    except Exception as e:
        w["hf_assisted_reference_equal_first_prompt"] = f"skipped: {type(e).__name__}: {e}"[:200]
    out["workloads"][wname] = w
dump(out, RESULTS / "bench_speculative.json")
