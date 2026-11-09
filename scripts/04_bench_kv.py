"""Task 3 benchmark: KV-cached vs uncached greedy decoding.

For every (input length, output length) pair we run, with the SAME model and inputs:
  hf_nocache        HF forward, full recomputation every step   (reference, no cache reuse)
  explicit_nocache  our Qwen2 forward, full recomputation        (reference for our code)
  explicit_cache    our Qwen2 forward + our pre-allocated KV cache
  hf_cache          HF forward + DynamicCache driven by our loop
EOS is ignored so every run produces exactly the requested number of tokens.
One warm-up run, then `repeats` timed runs with torch.cuda.synchronize(); medians reported.
Token-level agreement is checked against hf_nocache; on a mismatch we record the step and
the reference top-2 logit margin at that step (near-ties explain fp16 divergences)."""
import json, torch
from _common import ROOT, RESULTS, dump, env_info
from _bench_common import base_args, long_prompt_ids, median
from trialscreen.modeling import load_model
from trialscreen.decoding import (ExplicitQwen2Decoder, HFCacheDecoder, greedy_cached, greedy_nocache,
                                  first_mismatch)

ap = base_args("KV cache benchmark")
ap.add_argument("--inputs", default="128,512,1024")
ap.add_argument("--outputs", default="32,128,256")
ap.add_argument("--no-fp32-recheck", action="store_true",
                help="skip re-running mismatched cases in float32 (mismatch investigation)")
args = ap.parse_args()
if args.quick:
    args.inputs, args.outputs, args.repeats = "32,64", "8,16", 1
adapter = None if args.adapter == "none" else args.adapter
tok, model = load_model(args.model, adapter=adapter)
dev = model.device
run_dtype = str(next(model.parameters()).dtype)
modes = {
    "hf_nocache": lambda: (HFCacheDecoder(model), greedy_nocache),
    "explicit_nocache": lambda: (ExplicitQwen2Decoder(model, "torch_sdpa", max_len=4096), greedy_nocache),
    "explicit_cache": lambda: (ExplicitQwen2Decoder(model, "torch_sdpa", max_len=4096), greedy_cached),
    "explicit_cache_refattn": lambda: (ExplicitQwen2Decoder(model, "reference", max_len=4096), greedy_cached),
    "hf_cache": lambda: (HFCacheDecoder(model), greedy_cached),
}


@torch.no_grad()
def margin_at(ids, ref_tokens, step):
    seq = torch.cat([ids, torch.tensor([ref_tokens[:step]], device=dev, dtype=ids.dtype)], 1) if step else ids
    lg = model(input_ids=seq, logits_to_keep=1).logits[0, -1].float()
    top = lg.topk(2).values
    return float(top[0] - top[1])


results = []
for P in map(int, args.inputs.split(",")):
    ids = long_prompt_ids(tok, P).to(dev)
    for N in map(int, args.outputs.split(",")):
        row = {"input_tokens": ids.shape[1], "output_tokens": N, "modes": {}}
        ref = None
        for name, mk in modes.items():
            dec, fn = mk()
            fn(dec, ids, min(N, 4))                      # warm-up
            runs = [fn(dec, ids, N) for _ in range(args.repeats)]
            toks = runs[0].tokens
            if ref is None:
                ref = toks
            mm = first_mismatch(ref, toks)
            row["modes"][name] = {
                "prefill_s": median([r.prefill_s for r in runs]), "decode_s": median([r.decode_s for r in runs]),
                "total_s": median([r.total_s for r in runs]), "tokens_per_s": median([r.tok_s for r in runs]),
                "decode_tokens_per_s": median([r.decode_tok_s for r in runs]),
                "peak_mem_mb": max([r.peak_mem_mb or 0 for r in runs]) or None,
                "cache_mb": runs[0].cache_mb, "tokens_equal_reference": mm is None,
                "first_mismatch": mm, "mismatch_margin": margin_at(ids, ref, mm) if mm is not None else None}
            print(P, N, name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row["modes"][name].items()}, flush=True)
            del dec
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        results.append(row)
# ---- mismatch investigation: rerun every mismatched (input, output) pair in float32.
# If the paths agree in fp32, the fp16 divergence is a floating-point near-tie (top-2 margin
# of 0 or one fp16 step), not a caching bug.
recheck = []
mism = [(r, n) for r in results for n, v in r["modes"].items() if not v["tokens_equal_reference"]]
if mism and not args.no_fp32_recheck and torch.cuda.is_available():
    del model; torch.cuda.empty_cache()
    tok, model = load_model(args.model, adapter=adapter, dtype=torch.float32)
    dev = model.device
    for r in {id(r): r for r, _ in mism}.values():
        ids = long_prompt_ids(tok, r["input_tokens"]).to(dev)
        N = r["output_tokens"]
        ref32 = greedy_nocache(HFCacheDecoder(model), ids, N).tokens
        entry = {"input_tokens": r["input_tokens"], "output_tokens": N, "fp16_mismatched_paths": [n for rr, n in mism if rr is r]}
        for name in entry["fp16_mismatched_paths"]:
            dec, fn = modes[name]()
            toks = fn(dec, ids, N).tokens
            entry[name] = {"fp32_tokens_equal_reference": first_mismatch(ref32, toks) is None,
                           "fp32_first_mismatch": first_mismatch(ref32, toks)}
        recheck.append(entry); print("fp32 recheck", entry, flush=True)
cfg = model.config
kv_bytes_per_token = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * (cfg.hidden_size // cfg.num_attention_heads) * 2  # fp16
dump({"env": env_info(), "model": args.model, "adapter": adapter, "dtype": run_dtype,
      "decoding": "greedy, EOS ignored", "repeats": args.repeats, "warmup": 1, "fp32_recheck": recheck,
      "kv_bytes_per_token_theory": kv_bytes_per_token, "results": results}, RESULTS / "bench_kv.json")
