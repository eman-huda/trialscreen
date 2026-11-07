"""Task 4: correctness and performance of reference vs tiled online-softmax attention.

Correctness: errors are measured against a float64 reference for many shapes, causal
masks (square prefill and bottom-right aligned decode masks), block sizes that do and do
not divide the sequence length, and float32/float16 inputs.
Performance: median time and peak memory for reference (materialised T x T scores),
tiled (blocks of 128) and PyTorch SDPA (fused backend)."""
import argparse, statistics, time, torch
from _common import RESULTS, dump, env_info
from trialscreen.attention import reference_attention, tiled_attention, torch_sdpa

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
args = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
shapes = [  # B, H, Tq, Tk, D, block_q, block_k
    (1, 1, 4, 4, 8, 2, 2), (2, 3, 17, 17, 16, 4, 5), (1, 12, 128, 128, 128, 64, 64),
    (1, 12, 1, 300, 128, 64, 64), (1, 12, 7, 300, 128, 4, 64), (2, 4, 257, 257, 64, 32, 48),
    (1, 2, 1000, 1000, 64, 128, 128)]
dtypes = [torch.float32] + ([torch.float16] if dev == "cuda" else [torch.bfloat16])
correct = []
for dt in dtypes:
    for (B, H, Tq, Tk, D, bq, bk) in shapes:
        for causal in (True, False):
            q, k, v = (torch.randn(B, H, n, D, device=dev, dtype=torch.float64) for n in (Tq, Tk, Tk))
            truth = reference_attention(q, k, v, causal)          # float64 ground truth
            qd, kd, vd = q.to(dt), k.to(dt), v.to(dt)
            row = {"shape": [B, H, Tq, Tk, D], "blocks": [bq, bk], "causal": causal, "dtype": str(dt)}
            for name, fn in [("reference", lambda: reference_attention(qd, kd, vd, causal)),
                             ("tiled", lambda: tiled_attention(qd, kd, vd, causal, block_q=bq, block_k=bk)),
                             ("torch_sdpa", lambda: torch_sdpa(qd, kd, vd, causal))]:
                err = (fn().double() - truth).abs()
                row[name] = {"max_abs_err": float(err.max()), "mean_abs_err": float(err.mean())}
            ta = tiled_attention(qd, kd, vd, causal, block_q=bq, block_k=bk).double()
            tr = reference_attention(qd, kd, vd, causal).double()
            row["tiled_vs_reference_max"] = float((ta - tr).abs().max())
            tol = 1e-5 if dt == torch.float32 else 2e-2
            row["pass"] = row["tiled_vs_reference_max"] < tol
            correct.append(row)
            print(row["dtype"], row["shape"], "causal" if causal else "full", "tiled-ref", f"{row['tiled_vs_reference_max']:.2e}",
                  "PASS" if row["pass"] else "FAIL", flush=True)


def bench(fn, reps):
    for _ in range(2):
        fn()
    ts, mem = [], None
    if dev == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
    for _ in range(reps):
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter(); fn()
        if dev == "cuda":
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    if dev == "cuda":
        mem = (torch.cuda.max_memory_allocated() - base) / 2**20
    return statistics.median(ts), mem


perf = []
lengths = [128, 256] if args.quick else [256, 512, 1024, 2048, 4096]
dt = torch.float16 if dev == "cuda" else torch.float32
for T in lengths:
    q, k, v = (torch.randn(1, 12, T, 128, device=dev, dtype=dt) for _ in range(3))
    row = {"T": T, "heads": 12, "head_dim": 128, "dtype": str(dt)}
    for name, fn in [("reference", lambda: reference_attention(q, k, v, True)),
                     ("tiled_b128", lambda: tiled_attention(q, k, v, True, block_q=128, block_k=128)),
                     ("torch_sdpa", lambda: torch_sdpa(q, k, v, True))]:
        try:
            t, m = bench(fn, 3 if args.quick else 10)
            row[name] = {"ms": t * 1000, "extra_peak_mem_mb": m}
        except torch.OutOfMemoryError:
            row[name] = {"ms": None, "extra_peak_mem_mb": "OOM"}
            torch.cuda.empty_cache()
    print(row, flush=True)
    perf.append(row)
dump({"env": env_info(), "device": dev, "correctness": correct, "all_pass": all(r["pass"] for r in correct),
      "performance": perf}, RESULTS / "attention.json")
