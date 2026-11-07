"""Turn results/*.json into LaTeX tables (report/tables/*.tex), figures (report/figures/*.png)
and an error-analysis worksheet (results/error_cases.md). Missing inputs are skipped."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from _common import ROOT, RESULTS

FIG, TAB = ROOT / "report" / "figures", ROOT / "report" / "tables"
FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)


def load(name):
    p = RESULTS / name
    return json.load(open(p)) if p.exists() else None


def f(x, d=3):
    return "--" if x is None else (f"{x:.{d}f}" if isinstance(x, (float, int)) else str(x))


def esc(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


def write(name, body):
    (TAB / name).write_text(body); print("wrote", TAB / name)


# ---------------------------------------------------------------- training curves
hist = load("train_history.json")
if hist:
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.6))
    ax[0].plot(hist["step"], hist["train_loss"], lw=1)
    ax[0].set_xlabel("optimizer step"); ax[0].set_ylabel("train loss (answer tokens)"); ax[0].set_title("Training loss")
    ep = hist["epoch_end"]
    ax[1].plot([e["epoch"] for e in ep], [e["val_loss"] for e in ep], "o-", label="val loss")
    if "val_macro_f1" in ep[0]:
        a2 = ax[1].twinx(); a2.plot([e["epoch"] for e in ep], [e["val_macro_f1"] for e in ep], "s--", c="C1", label="val macro-F1")
        a2.set_ylabel("val macro-F1")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("val loss"); ax[1].set_title("Validation")
    fig.tight_layout(); fig.savefig(FIG / "training_curves.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- before / after metrics
rows = []
for split in ("validation", "test"):
    for model in ("base", "finetuned"):
        m = load(f"metrics_{model}_{split}.json")
        if m:
            c, e = m["classification"], m["evidence"]["overall"]
            rows.append((split, model, c["overall"]["accuracy"], c["overall"]["macro_f1"],
                         c["inclusion"]["macro_f1"], c["exclusion"]["macro_f1"], e["micro"]["f1"],
                         e["macro_nonempty_gold"]["f1"], m["valid_json_rate"]))
if rows:
    body = ("\\begin{table}[t]\\centering\\caption{Same model before and after LoRA fine-tuning (greedy decoding, "
            "identical prompts). Macro-F1 averages classes present in the gold subset.}\\label{tab:beforeafter}\n"
            "\\resizebox{\\columnwidth}{!}{\\begin{tabular}{llrrrrrrr}\\toprule\n"
            "Split & Model & Acc & Macro-F1 & Incl.~F1 & Excl.~F1 & Ev.~micro-F1 & Ev.~macro-F1 & Valid JSON\\\\\\midrule\n")
    for r in rows:
        body += f"{r[0]} & {r[1]} & " + " & ".join(f(x) for x in r[2:]) + "\\\\\n"
    write("before_after.tex", body + "\\bottomrule\\end{tabular}}\\end{table}\n")
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    labels = [f"{r[0][:3]}-{r[1][:4]}" for r in rows]
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [r[2] for r in rows], 0.4, label="accuracy"); ax.bar(x + 0.2, [r[3] for r in rows], 0.4, label="macro-F1")
    ax.set_xticks(x, labels, rotation=20, fontsize=7); ax.set_ylim(0, 1); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "before_after.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- per-class + confusion (final test)
for model in ("finetuned", "base"):
    m = load(f"metrics_{model}_test.json")
    if not m:
        continue
    body = (f"\\begin{{table}}[t]\\centering\\caption{{Per-class test results, {model} model.}}\\label{{tab:perclass_{model}}}\n"
            "\\resizebox{\\columnwidth}{!}{\\begin{tabular}{llrrrr}\\toprule Subset & Class & P & R & F1 & Support\\\\\\midrule\n")
    for sub in ("inclusion", "exclusion"):
        for lab, v in m["classification"][sub]["per_class"].items():
            body += f"{sub} & {lab} & {f(v['precision'])} & {f(v['recall'])} & {f(v['f1'])} & {v['support']}\\\\\n"
        body += "\\midrule\n"
    write(f"perclass_{model}.tex", body + "\\bottomrule\\end{tabular}}\\end{table}\n")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax, sub in zip(axes, ("inclusion", "exclusion")):
        cm = m["classification"][sub]["confusion_matrix"]
        M = np.array(cm["matrix"]); labs = [l.replace("not enough information", "NEI").replace("not applicable", "N/A")
                                             .replace("invalid output", "invalid") for l in cm["labels"]]
        ax.imshow(M, cmap="Blues")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, M[i, j], ha="center", va="center", fontsize=7, color="black" if M[i, j] < M.max() / 2 else "white")
        ax.set_xticks(range(len(labs)), labs, rotation=35, fontsize=6, ha="right"); ax.set_yticks(range(len(labs)), labs, fontsize=6)
        ax.set_xlabel("predicted", fontsize=7); ax.set_ylabel("expert", fontsize=7); ax.set_title(f"{sub} ({model}, test)", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / f"confusion_{model}_test.png", dpi=200); plt.close(fig)
    ev = m["evidence"]
    body = (f"\\begin{{table}}[t]\\centering\\caption{{Evidence sentence retrieval on test, {model} model.}}\\label{{tab:evidence_{model}}}\n"
            "\\resizebox{\\columnwidth}{!}{\\begin{tabular}{lrrrrrrr}\\toprule Subset & micro-P & micro-R & micro-F1 & macro-P & macro-R & macro-F1 & empty-gold agree\\\\\\midrule\n")
    for sub in ("overall", "inclusion", "exclusion"):
        e = ev[sub]; eg = e["empty_gold"]
        body += (f"{sub} & {f(e['micro']['precision'])} & {f(e['micro']['recall'])} & {f(e['micro']['f1'])} & "
                 f"{f(e['macro_nonempty_gold']['precision'])} & {f(e['macro_nonempty_gold']['recall'])} & "
                 f"{f(e['macro_nonempty_gold']['f1'])} & {eg['model_also_empty']}/{eg['n_records']}\\\\\n")
    write(f"evidence_{model}.tex", body + "\\bottomrule\\end{tabular}}\\end{table}\n")

# ---------------------------------------------------------------- KV cache
kv = load("bench_kv.json")
if kv:
    body = ("\\begin{table*}[t]\\centering\\caption{KV-cache benchmark (" + esc(kv['dtype']) + ", " + esc(kv['env'].get('gpu', 'cpu')) +
            ", median of " + str(kv['repeats']) + " runs after warm-up). Prefill = time to first token; tok/s over all new tokens; "
            "=ref: greedy tokens identical to the uncached HF reference.}\\label{tab:kv}\n"
            "\\begin{tabular}{rrlrrrrrc}\\toprule In & Out & Path & Prefill (ms) & Decode (ms) & tok/s & Peak mem (MB) & Cache (MB) & =ref\\\\\\midrule\n")
    for r in kv["results"]:
        for name, v in r["modes"].items():
            body += (f"{r['input_tokens']} & {r['output_tokens']} & {esc(name)} & {f(v['prefill_s'] * 1000, 1)} & "
                     f"{f(v['decode_s'] * 1000, 1)} & {f(v['tokens_per_s'], 1)} & {f(v['peak_mem_mb'], 0)} & "
                     f"{f(v['cache_mb'], 1)} & {'yes' if v['tokens_equal_reference'] else 'no@' + str(v['first_mismatch'])}\\\\\n")
        body += "\\midrule\n"
    write("kv_bench.tex", body + "\\bottomrule\\end{tabular}\\end{table*}\n")
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.6))
    for name in ("hf_nocache", "explicit_cache", "hf_cache"):
        for P in sorted({r["input_tokens"] for r in kv["results"]}):
            pts = [(r["output_tokens"], r["modes"][name]["tokens_per_s"]) for r in kv["results"] if r["input_tokens"] == P]
            ax[0].plot(*zip(*pts), "o-", label=f"{name} in={P}", lw=1, ms=3)
    ax[0].set_xlabel("output tokens"); ax[0].set_ylabel("tokens/s"); ax[0].legend(fontsize=5); ax[0].set_title("Throughput")
    for name in ("hf_nocache", "explicit_cache"):
        pts = [(r["input_tokens"] + r["output_tokens"], r["modes"][name]["peak_mem_mb"] or 0) for r in kv["results"]]
        ax[1].scatter(*zip(*pts), label=name, s=12)
    ax[1].set_xlabel("total sequence length"); ax[1].set_ylabel("peak GPU memory (MB)"); ax[1].legend(fontsize=6); ax[1].set_title("Memory")
    fig.tight_layout(); fig.savefig(FIG / "kv_bench.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- attention
at = load("attention.json")
if at:
    body = ("\\begin{table}[t]\\centering\\caption{Attention correctness against a float64 reference (selection). "
            "Max absolute error of each implementation.}\\label{tab:attn_err}\n"
            "\\resizebox{\\columnwidth}{!}{\\begin{tabular}{llcrrr}\\toprule dtype & shape (B,H,Tq,Tk,D) & causal & reference & tiled & SDPA\\\\\\midrule\n")
    for r in at["correctness"]:
        if r["causal"] or r["shape"][2] == 1:
            body += (f"{esc(r['dtype'].replace('torch.', ''))} & {tuple(r['shape'])} & {'y' if r['causal'] else 'n'} & "
                     f"{r['reference']['max_abs_err']:.1e} & {r['tiled']['max_abs_err']:.1e} & {r['torch_sdpa']['max_abs_err']:.1e}\\\\\n")
    write("attention_err.tex", body + "\\bottomrule\\end{tabular}}\\end{table}\n")
    body = ("\\begin{table}[t]\\centering\\caption{Attention runtime (ms, median) and extra peak memory (MB), 12 heads, "
            "head dim 128, causal, " + esc(at['device']) + ".}\\label{tab:attn_perf}\n"
            "\\resizebox{\\columnwidth}{!}{\\begin{tabular}{rrrrrrr}\\toprule T & ref ms & tiled ms & SDPA ms & ref MB & tiled MB & SDPA MB\\\\\\midrule\n")
    for r in at["performance"]:
        g = lambda n, k: r[n][k] if isinstance(r[n][k], (int, float)) else None
        body += (f"{r['T']} & {f(g('reference','ms'),2)} & {f(g('tiled_b128','ms'),2)} & {f(g('torch_sdpa','ms'),2)} & "
                 f"{f(g('reference','extra_peak_mem_mb'),1)} & {f(g('tiled_b128','extra_peak_mem_mb'),1)} & {f(g('torch_sdpa','extra_peak_mem_mb'),1)}\\\\\n")
    write("attention_perf.tex", body + "\\bottomrule\\end{tabular}}\\end{table}\n")
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.6))
    for n, lab in (("reference", "reference"), ("tiled_b128", "tiled (128)"), ("torch_sdpa", "PyTorch SDPA")):
        pts = [(r["T"], r[n]["ms"]) for r in at["performance"] if isinstance(r[n]["ms"], (int, float))]
        if pts: ax[0].plot(*zip(*pts), "o-", label=lab)
        pts = [(r["T"], r[n]["extra_peak_mem_mb"]) for r in at["performance"] if isinstance(r[n]["extra_peak_mem_mb"], (int, float))]
        if pts: ax[1].plot(*zip(*pts), "o-", label=lab)
    for a, t in zip(ax, ("time (ms)", "extra peak memory (MB)")):
        a.set_xscale("log", base=2); a.set_yscale("log"); a.set_xlabel("sequence length T"); a.set_ylabel(t); a.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(FIG / "attention_perf.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- speculative
sp = load("bench_speculative.json")
if sp:
    body = ("\\begin{table}[t]\\centering\\caption{Greedy speculative decoding (target " + esc(sp['target'].split('/')[-1]) +
            ", draft " + esc(sp['draft'].split('/')[-1]) + "). Equal = share of prompts whose tokens match greedy target decoding.}"
            "\\label{tab:spec}\n\\resizebox{\\columnwidth}{!}{\\begin{tabular}{llrrrrrr}\\toprule Workload & Method & Equal & Accept & Tok/round & Tok/s & Speedup & Peak MB\\\\\\midrule\n")
    for wn, w in sp["workloads"].items():
        g = w["greedy"]
        body += f"{wn} & greedy & -- & -- & 1.00 & {f(g['tokens_per_s'],1)} & 1.00 & {f(g['peak_mem_mb'],0)}\\\\\n"
        for k, a in w["speculative"].items():
            body += (f"{wn} & spec k={k} & {f(a['token_equal_rate'],2)} & {f(a['acceptance_rate'],2)} & {f(a['tokens_per_round'],2)} & "
                     f"{f(a['tokens_per_s'],1)} & {f(a['speedup_vs_greedy'],2)} & {f(a['peak_mem_mb'],0)}\\\\\n")
        body += "\\midrule\n"
    write("speculative.tex", body + "\\bottomrule\\end{tabular}}\\end{table}\n")
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for wn, w in sp["workloads"].items():
        ks = sorted(w["speculative"], key=int)
        ax.plot([int(k) for k in ks], [w["speculative"][k]["speedup_vs_greedy"] for k in ks], "o-", label=f"{wn} speedup")
        ax.plot([int(k) for k in ks], [w["speculative"][k]["acceptance_rate"] for k in ks], "s--", label=f"{wn} acceptance")
    ax.axhline(1, c="gray", lw=0.8); ax.set_xlabel("draft lookahead k"); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(FIG / "speculative.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- error analysis worksheet (test, fine-tuned)
pf = RESULTS / "predictions_finetuned_test.jsonl"
if pf.exists():
    P = [json.loads(l) for l in open(pf)]
    errs = [p for p in P if p["pred_label"] != p["gold_label"] or set(p["pred_evidence"]) != set(p["gold_evidence"])]

    def category(p):
        g, pr, ge, pe = p["gold_label"], p["pred_label"], set(p["gold_evidence"]), set(p["pred_evidence"])
        if pr == "invalid output": return "invalid output"
        if pr != g and g == "not enough information": return "over-confident: decided despite missing information"
        if pr == "not enough information" and g != pr: return "missed available evidence (predicted not enough information)"
        if pr != g and pr != "not enough information" and not pe: return "unsupported decision (no evidence cited)"
        if pr != g and ({"included", "not included"} == {g, pr} or {"excluded", "not excluded"} == {g, pr}): return "contradictory polarity"
        if pr != g and "not applicable" in (g, pr): return "applicability confusion"
        if pr == g and ge and not (ge & pe): return "right label, wrong evidence"
        if pr == g: return "right label, partial evidence"
        return "other label error"
    by = {}
    for p in errs:
        by.setdefault(category(p), []).append(p)
    lines = ["# Error-analysis worksheet (test split, fine-tuned model)\n",
             f"{len(errs)} of {len(P)} predictions differ from the expert label or evidence set.\n"]
    for cat, items in sorted(by.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"\n## {cat} ({len(items)})\n")
        for p in items[:4]:
            lines.append(f"- annotation {p['annotation_id']} ({p['patient_id']}, {p['trial_id']}, {p['criterion_type']}): "
                         f"criterion: {p['criterion_text'][:160]!r} | expert: {p['gold_label']} {p['gold_evidence']} | "
                         f"model: {p['pred_label']} {p['pred_evidence']} | raw: {p['raw_output'][:120]!r}")
    (RESULTS / "error_cases.md").write_text("\n".join(lines)); print("wrote", RESULTS / "error_cases.md")
    body = ("\\begin{table}[t]\\centering\\caption{Test-set disagreements of the fine-tuned model by category.}\\label{tab:errcat}\n"
            "\\begin{tabular}{lr}\\toprule Category & Count\\\\\\midrule\n")
    for cat, items in sorted(by.items(), key=lambda kv: -len(kv[1])):
        body += f"{esc(cat)} & {len(items)}\\\\\n"
    write("error_categories.tex", body + "\\bottomrule\\end{tabular}\\end{table}\n")

# ---------------------------------------------------------------- split table (always available)
man = json.load(open(ROOT / "data" / "splits" / "split_manifest.json"))
body = ("\\begin{table}[t]\\centering\\caption{Fixed patient-grouped split (seed 42) and expert label counts.}\\label{tab:split}\n"
        "\\resizebox{\\columnwidth}{!}{\\begin{tabular}{lrrrrrrrrr}\\toprule Split & Pat. & Ann. & Incl & NotIncl & Excl & NotExcl & NEI & N/A & Empty ev.\\\\\\midrule\n")
for s, v in man["splits"].items():
    lc = {**{f"inclusion:{k}": n for k, n in v["label_counts"]["inclusion"].items()},
          **{f"exclusion:{k}": n for k, n in v["label_counts"]["exclusion"].items()}}
    g = lambda k: lc.get(k, 0)
    nei = g("inclusion:not enough information") + g("exclusion:not enough information")
    na = g("inclusion:not applicable") + g("exclusion:not applicable")
    body += (f"{s} & {v['n_patients']} & {v['n_annotations']} & {g('inclusion:included')} & {g('inclusion:not included')} & "
             f"{g('exclusion:excluded')} & {g('exclusion:not excluded')} & {nei} & {na} & {v['empty_evidence']}\\\\\n")
write("split.tex", body + "\\bottomrule\\end{tabular}}\\end{table}\n")
