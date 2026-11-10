"""TrialScreen browser application (Gradio).

Start:  python app/app.py            then open http://127.0.0.1:7860
Env vars:
  TS_MODEL    target model id            (default Qwen/Qwen2.5-1.5B-Instruct)
  TS_ADAPTER  LoRA adapter (Hub id or folder; default: local best_adapter if present)
  TS_DRAFT    draft model id             (default Qwen/Qwen2.5-0.5B-Instruct; 'none' disables)
  TS_DTYPE    float32 | bfloat16 | float16 (default: float32 on CPU, fp16/bf16 on GPU)
  TS_SHARE    1 = also create a public gradio.live link
"""
from __future__ import annotations
import html, json, os, sys, time, tempfile, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr
import pandas as pd
import torch

from trialscreen.data import split_sentences
from trialscreen.criteria import parse_criteria
from trialscreen.modeling import load_model, predict_one, prompt_ids, eos_ids, TARGET_ID, DRAFT_ID
from trialscreen.decoding import (ExplicitQwen2Decoder, HFCacheDecoder, greedy_cached, greedy_nocache,
                                  first_mismatch)
from trialscreen.speculative import speculative_greedy
from trialscreen.attention import reference_attention, tiled_attention, torch_sdpa
from trialscreen.explain import aggregate_trial
from trialscreen.report_export import html_report, provenance, DISCLAIMER

MAX_CRITERIA, MAX_NOTE_TOKENS, MAX_CRIT_CHARS = 40, 2500, 1500
EXAMPLES = json.load(open(ROOT / "app" / "examples.json"))
MODES = {"KV cache: explicit decoder (ours)": "explicit_cache",
         "KV cache: HF forward + DynamicCache": "hf_cache",
         "No cache: full recompute (reference)": "explicit_nocache",
         "Speculative: 0.5B draft + 1.5B target (ours)": "speculative"}

# ------------------------------------------------------------------ model loading
STATE = {"status": "loading"}


def _load():
    t0 = time.time()
    model_id = os.environ.get("TS_MODEL", TARGET_ID)
    local = ROOT / "checkpoints" / "qwen2.5-1.5b-trialscreen-lora" / "best_adapter"
    adapter = os.environ.get("TS_ADAPTER", str(local) if local.exists() else "")
    adapter = adapter if adapter and adapter.lower() != "none" else None
    dt = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}.get(os.environ.get("TS_DTYPE", ""))
    tok, model = load_model(model_id, adapter=adapter, dtype=dt)
    draft_id = os.environ.get("TS_DRAFT", DRAFT_ID)
    draft = None
    if draft_id.lower() != "none":
        try:
            dtok, draft = load_model(draft_id, device=str(model.device), dtype=next(model.parameters()).dtype)
            if dtok.get_vocab() != tok.get_vocab():
                draft, STATE["draft_error"] = None, "draft tokenizer differs from target; speculative disabled"
        except Exception as e:
            STATE["draft_error"] = f"draft not loaded: {e}"
    STATE.update(tok=tok, model=model, draft=draft, model_id=model_id, adapter=adapter, draft_id=draft_id if draft else None,
                 device=str(model.device), dtype=str(next(model.parameters()).dtype), status="ready",
                 load_s=time.time() - t0)


def status_md():
    if STATE.get("status") != "ready":
        return "Model status: **loading**…"
    s = (f"Model: `{STATE['model_id']}` | adapter: `{STATE['adapter'] or 'none (base model)'}` | draft: "
         f"`{STATE['draft_id'] or 'disabled'}` | device: `{STATE['device']}` ({STATE['dtype']}) | loaded in {STATE['load_s']:.0f}s")
    if STATE.get("draft_error"):
        s += f"  \n⚠ {STATE['draft_error']}"
    if not STATE["adapter"]:
        s += "  \n⚠ No fine-tuned adapter found: running the BASE model. Set TS_ADAPTER to use the fine-tuned one."
    return s


# ------------------------------------------------------------------ screening
def _validate(note, criteria_text):
    if not note or len(note.strip()) < 20:
        raise gr.Error("Please paste a patient record (at least one full sentence).")
    if not criteria_text or not criteria_text.strip():
        raise gr.Error("Please enter the trial's eligibility criteria.")
    crits, problems = parse_criteria(criteria_text)
    if problems:
        raise gr.Error("Criteria format problem: " + "; ".join(problems[:3]) +
                       ". Use 'Inclusion criteria:' and 'Exclusion criteria:' headings with one criterion per line.")
    if not crits:
        raise gr.Error("No criteria found. Put each criterion on its own line under an 'Inclusion criteria:' or 'Exclusion criteria:' heading.")
    if len(crits) > MAX_CRITERIA:
        raise gr.Error(f"{len(crits)} criteria given; the limit is {MAX_CRITERIA} per run.")
    long = [c for _, c in crits if len(c) > MAX_CRIT_CHARS]
    if long:
        raise gr.Error(f"A criterion is longer than {MAX_CRIT_CHARS} characters; split it into separate lines.")
    sents = split_sentences(note)
    n_tok = len(STATE["tok"](note).input_ids)
    if n_tok > MAX_NOTE_TOKENS:
        raise gr.Error(f"The record has {n_tok} tokens; the limit is {MAX_NOTE_TOKENS}. Shorten it (no silent truncation).")
    return crits, sents


def _highlight(sents, cited):
    out = []
    for i, s in sents:
        if i in cited:
            out.append(f"<li value='{i}' style='background:#fce7f3;border-left:4px solid #f472b6;border-radius:6px'><b>[{i}]</b> {html.escape(s)}</li>")
        else:
            out.append(f"<li value='{i}'><b>[{i}]</b> {html.escape(s)}</li>")
    return "<ol start='0' style='list-style:none;padding-left:0.4em'>" + "".join(out) + "</ol>"


def screen(note, criteria_text, trial_name, mode_label, attn_backend, k, progress=gr.Progress()):
    if STATE.get("status") != "ready":
        raise gr.Error("The model is still loading. Wait for the status line to say it is ready.")
    crits, sents = _validate(note, criteria_text)
    mode = MODES[mode_label]
    if mode == "speculative" and STATE["draft"] is None:
        raise gr.Error("Speculative mode needs the draft model, which is not loaded. Choose another mode.")
    tok, model, draft = STATE["tok"], STATE["model"], STATE["draft"]
    tgt_dec = ExplicitQwen2Decoder(model, attn_backend) if mode != "hf_cache" else HFCacheDecoder(model)
    dr_dec = ExplicitQwen2Decoder(draft, attn_backend) if mode == "speculative" else None
    results, t0 = [], time.time()
    for j, (ctype, ctext) in enumerate(progress.tqdm(crits, desc="Screening criteria")):
        try:
            r = predict_one(tok, model, note, ctype, ctext, mode=mode, attn_backend=attn_backend,
                            draft_model=draft, k=int(k), decoder=tgt_dec, draft_decoder=dr_dec)
        except Exception as e:      # one bad criterion must not kill the whole run
            r = {"criterion_type": ctype, "criterion_text": ctext, "label": "invalid output", "evidence": [],
                 "evidence_text": [], "explanation": f"Error while running the model: {type(e).__name__}: {e}",
                 "raw_output": "", "valid_json": False, "parse_notes": [traceback.format_exc(limit=1)], "decode": {}}
        results.append(r)
    total = time.time() - t0
    verdict = aggregate_trial(results)
    df = pd.DataFrame([{"#": i + 1, "type": r["criterion_type"], "criterion": r["criterion_text"], "prediction": r["label"],
                        "evidence IDs": ", ".join(map(str, r["evidence"])) or "-",
                        "evidence text": " | ".join(r["evidence_text"]) or "-"} for i, r in enumerate(results)])
    cited = set(i for r in results for i in r["evidence"])
    icon = {"likely ineligible": "🔴", "potentially eligible": "🟢"}.get(verdict["verdict"], "🟡")
    missing = [r["criterion_text"] for r in results if r["label"] == "not enough information"]
    unsupported = [r["criterion_text"] for r in results if r["label"] not in ("not enough information", "invalid output") and not r["evidence"]]
    md = (f"### {icon} Preliminary outcome: **{verdict['verdict']}**\n"
          f"{verdict['n_criteria']} criteria · {verdict['n_blocking']} blocking · {verdict['n_unknown']} undetermined · "
          f"{total:.1f}s with *{mode_label}*  \n"
          f"<small>Rule-based aggregation of criterion predictions, not a ground-truth label. Rule: {verdict['rule']}</small>\n\n"
          "**Missing information** (record does not say enough):\n" + ("\n".join(f"- {m}" for m in missing) or "- none") +
          ("\n\n**Decisions without cited evidence** (check these):\n" + "\n".join(f"- {u}" for u in unsupported) if unsupported else ""))
    details = "".join(
        f"<details><summary><b>{i + 1}. [{r['criterion_type']}] {html.escape(r['label'])}</b> — {html.escape(r['criterion_text'][:120])}</summary>"
        f"<p>{html.escape(r['explanation'])}</p><p><small>raw model output: <code>{html.escape(r['raw_output'][:300])}</code>"
        f"{' · notes: ' + html.escape('; '.join(r['parse_notes'])) if r['parse_notes'] else ''}</small></p></details>"
        for i, r in enumerate(results))
    prov = provenance(note, crits, {"model": STATE["model_id"], "adapter": STATE["adapter"], "draft": STATE["draft_id"],
                                     "decoding": f"greedy, {mode}", "attention_backend": attn_backend,
                                     "lookahead_k": int(k) if mode == "speculative" else None, "device": STATE["device"],
                                     "dtype": STATE["dtype"], "runtime_s": round(total, 2)})
    tmp = Path(tempfile.mkdtemp())
    hp = tmp / "screening_report.html"; jp = tmp / "screening_report.json"
    hp.write_text(html_report(trial_name, sents, results, verdict, prov), encoding="utf-8")
    jp.write_text(json.dumps({"trial": trial_name, "verdict": verdict, "results": results, "patient_sentences": sents,
                              "provenance": prov, "notice": DISCLAIMER}, indent=1, default=str), encoding="utf-8")
    return md, df, _highlight(sents, cited), details, [str(hp), str(jp)]


def load_example(name):
    ex = next(e for e in EXAMPLES if e["name"] == name)
    return ex["note"], ex["criteria"], ex["trial_id"] or "Example trial"


# ------------------------------------------------------------------ live benchmark
def bench(note, criteria_text, max_new, k, which, progress=gr.Progress()):
    if STATE.get("status") != "ready":
        raise gr.Error("Model still loading.")
    tok, model, draft = STATE["tok"], STATE["model"], STATE["draft"]
    max_new = int(max_new)
    if which == "Free-text summary of the note (EOS ignored)":
        if not note or len(note.strip()) < 20:
            raise gr.Error("Paste a patient note first (Screening tab or below).")
        msgs = [{"role": "user", "content": "Summarise this patient's history in detail.\n\n" + note}]
        ids = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), add_special_tokens=False).input_ids
        eos = ()
    else:
        crits, sents = _validate(note, criteria_text)
        ctype, ctext = crits[0]
        numbered = "\n".join(f"{i}. {s}" for i, s in sents)
        ids = prompt_ids(tok, numbered, ctype, ctext)
        eos = eos_ids(tok)
    ids = torch.tensor([ids], device=model.device)
    rows, ref = [], None
    runs = [("No cache (HF, reference)", lambda: greedy_nocache(HFCacheDecoder(model), ids, max_new, eos)),
            ("No cache (explicit)", lambda: greedy_nocache(ExplicitQwen2Decoder(model, "torch_sdpa"), ids, max_new, eos)),
            ("KV cache (explicit, ours)", lambda: greedy_cached(ExplicitQwen2Decoder(model, "torch_sdpa"), ids, max_new, eos)),
            ("KV cache (explicit + tiled attention)", lambda: greedy_cached(ExplicitQwen2Decoder(model, "tiled"), ids, max_new, eos)),
            ("KV cache (HF DynamicCache)", lambda: greedy_cached(HFCacheDecoder(model), ids, max_new, eos))]
    if draft is not None:
        runs.append((f"Speculative k={int(k)} (ours)", lambda: speculative_greedy(
            ExplicitQwen2Decoder(model, "torch_sdpa"), ExplicitQwen2Decoder(draft, "torch_sdpa"), ids, max_new, k=int(k), eos_ids=eos)))
    for name, fn in progress.tqdm(runs, desc="Running decoders"):
        st = fn()
        toks = st.tokens
        ref = toks if ref is None else ref
        d = st.as_dict()
        rows.append({"method": name, "new tokens": len(toks),
                     "prefill ms": round(d.get("prefill_s", 0) * 1000, 1),
                     "total ms": round(d["total_s"] * 1000, 1),
                     "tokens/s": round(len(toks) / d["total_s"], 1) if d["total_s"] else None,
                     "peak GPU MB": round(d["peak_mem_mb"], 0) if d.get("peak_mem_mb") else "n/a (CPU)",
                     "same tokens as reference": "yes" if first_mismatch(ref, toks) is None else f"no (first diff at {first_mismatch(ref, toks)})",
                     "draft acceptance": round(d["acceptance_rate"], 2) if "acceptance_rate" in d else "-"})
    df = pd.DataFrame(rows)
    base = df.loc[0, "total ms"]
    df["speedup vs no-cache"] = (base / df["total ms"]).round(2)
    text = tok.decode([t for t in ref if t not in eos], skip_special_tokens=True)
    return df, df[["method", "tokens/s"]], f"**Prompt tokens:** {ids.shape[1]} · **Reference output:**\n\n```\n{text[:1500]}\n```"


def attention_check(T, D, block):
    T, D, block = int(T), int(D), int(block)
    dev = STATE.get("device", "cpu")
    torch.manual_seed(0)
    rows = []
    for causal, tq in ((True, T), (True, 1), (False, T)):
        q = torch.randn(1, 4, tq, D, device=dev, dtype=torch.float64)
        k_, v_ = torch.randn(1, 4, T, D, device=dev, dtype=torch.float64), torch.randn(1, 4, T, D, device=dev, dtype=torch.float64)
        truth = reference_attention(q, k_, v_, causal)
        for name, fn in (("reference", reference_attention), ("tiled online-softmax", lambda a, b, c, cz: tiled_attention(a, b, c, cz, block_q=block, block_k=block)),
                         ("PyTorch SDPA", torch_sdpa)):
            qq, kk, vv = q.float(), k_.float(), v_.float()
            fn(qq, kk, vv, causal)
            if dev.startswith("cuda"): torch.cuda.synchronize()
            t0 = time.perf_counter(); out = fn(qq, kk, vv, causal)
            if dev.startswith("cuda"): torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000
            err = (out.double() - truth).abs()
            rows.append({"mask": ("causal" if causal else "none") + (" (decode, 1 query)" if tq == 1 else ""),
                         "method": name, "max abs error vs float64": f"{float(err.max()):.2e}",
                         "mean abs error": f"{float(err.mean()):.2e}", "time ms (fp32)": round(ms, 2)})
    return pd.DataFrame(rows)


def saved_logs():
    out = []
    for f in sorted((ROOT / "results").glob("*.json")):
        out.append(f"- `{f.name}` ({f.stat().st_size // 1024} KB)")
    return "Benchmark logs from the full GPU runs (also used in the report):\n" + ("\n".join(out) or "- none yet")


# ------------------------------------------------------------------ UI
# ---- pastel theme: pink + lilac + light blue (light mode only) ---------------------
THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.pink, secondary_hue=gr.themes.colors.violet,
    neutral_hue=gr.themes.colors.slate, radius_size=gr.themes.sizes.radius_lg,
    font=[gr.themes.GoogleFont("Nunito"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="linear-gradient(135deg, #fdf2f8 0%, #f5f3ff 50%, #eff6ff 100%)",
    body_text_color="#2e1065",
    block_background_fill="rgba(255,255,255,0.88)",
    block_border_color="#ead7f7", block_border_width="1px",
    block_label_background_fill="#fce7f3", block_label_text_color="#831843",
    block_title_text_color="#5b21b6",
    input_background_fill="#ffffff", input_border_color="#e9d5ff",
    input_border_color_focus="#f472b6",
    button_primary_background_fill="linear-gradient(90deg, #f472b6, #a78bfa)",
    button_primary_background_fill_hover="linear-gradient(90deg, #ec4899, #8b5cf6)",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#e0f2fe", button_secondary_background_fill_hover="#bae6fd",
    button_secondary_text_color="#075985", button_secondary_border_color="#bae6fd",
    slider_color="#f472b6", checkbox_label_background_fill_selected="#fce7f3",
    table_even_background_fill="#fdf4ff", table_odd_background_fill="#ffffff",
    shadow_drop="0 4px 18px rgba(167,139,250,0.15)",
)

CSS = """
.gradio-container {max-width: 1280px !important; margin: auto;}
#hero {text-align:center; padding: 18px 12px 6px;}
#hero h1 {font-size: 2.4rem; margin: 0; background: linear-gradient(90deg,#db2777,#7c3aed,#2563eb);
  -webkit-background-clip: text; background-clip: text; color: transparent; font-weight: 800;}
#hero p {color:#6b21a8; margin: 6px 0 0; font-size: 1.05rem;}
.notice {background:#fff1f7; border:2px solid #f9a8d4; color:#831843; border-radius:14px;
  padding:10px 16px; font-weight:600; text-align:center;}
.status {background:#eff6ff; border:1px solid #bfdbfe; border-radius:12px; padding:8px 14px; color:#1e3a8a;}
.tab-nav button, button[role="tab"] {font-weight:700 !important; color:#6d28d9 !important;}
button[role="tab"][aria-selected="true"] {color:#db2777 !important; border-color:#f472b6 !important;}
details {background:#faf5ff; border:1px solid #e9d5ff; border-radius:10px; padding:6px 10px; margin:6px 0;}
details summary {cursor:pointer; color:#5b21b6;}
ol li {padding:2px 4px; border-radius:6px;}
footer {display:none !important;}
"""
# always light mode, even when the visitor's browser prefers dark
FORCE_LIGHT = """() => { document.body.classList.remove('dark');
  const u = new URL(window.location); if (u.searchParams.get('__theme') !== 'light') {
    u.searchParams.set('__theme', 'light'); window.location.replace(u); } }"""

with gr.Blocks(title="TrialScreen: clinical trial eligibility screening (educational)") as demo:
    gr.HTML("<div id='hero'><h1>🩺 TrialScreen</h1><p>Criterion-level clinical trial eligibility screening with evidence citations</p></div>")
    gr.HTML(f"<div class='notice'>⚠ {DISCLAIMER}</div>")
    status = gr.Markdown(status_md(), elem_classes="status")
    with gr.Tab("Screen a patient"):
        with gr.Row():
            with gr.Column(scale=1):
                ex = gr.Dropdown([e["name"] for e in EXAMPLES], label="Load an example (validation split or hand-written)", value=None)
                trial = gr.Textbox(label="Trial name or ID", placeholder="e.g. NCT01234567")
                note = gr.Textbox(label="Patient record", lines=10,
                                  placeholder="Paste a clinical note. Numbered sentences ('0. ...') are kept; otherwise sentences are split and numbered automatically.")
                crit = gr.Textbox(label="Eligibility criteria", lines=10,
                                  placeholder="Inclusion criteria:\n- ...\nExclusion criteria:\n- ...")
                with gr.Row():
                    mode = gr.Dropdown(list(MODES), value=list(MODES)[0], label="Decoding mode")
                    attn = gr.Dropdown(["reference", "tiled", "torch_sdpa"], value="torch_sdpa", label="Attention backend (explicit decoder)")
                    kk = gr.Slider(1, 8, value=4, step=1, label="Draft lookahead k (speculative)")
                run = gr.Button("Screen patient", variant="primary")
            with gr.Column(scale=1):
                verdict = gr.Markdown()
                table = gr.Dataframe(label="Criterion-level results", wrap=True, interactive=False)
                files = gr.File(label="Download report (HTML + JSON with provenance)", file_count="multiple")
        with gr.Row():
            with gr.Column():
                gr.Markdown("**Patient record with cited sentences highlighted**")
                note_html = gr.HTML()
            with gr.Column():
                gr.Markdown("**Explanations per criterion** (built only from the cited sentences)")
                det = gr.HTML()
        ex.change(load_example, ex, [note, crit, trial])
        run.click(screen, [note, crit, trial, mode, attn, kk], [verdict, table, note_html, det, files])
    with gr.Tab("Inference benchmark"):
        gr.Markdown("Runs every decoding path **live** on the same prompt (from the Screening tab inputs) with greedy decoding, "
                    "checks that tokens match the uncached reference, and times them. Single runs: use the repeated, "
                    "warmed-up GPU benchmark scripts for report numbers.")
        with gr.Row():
            which = gr.Radio(["Screening prompt (first criterion)", "Free-text summary of the note (EOS ignored)"],
                             value="Free-text summary of the note (EOS ignored)", label="Workload")
            mx = gr.Slider(8, 256, value=64, step=8, label="Max new tokens")
            kb = gr.Slider(1, 8, value=4, step=1, label="Speculative lookahead k")
        bbtn = gr.Button("Run comparison", variant="primary")
        btab = gr.Dataframe(label="Decoding comparison", interactive=False, wrap=True)
        bplot = gr.BarPlot(x="method", y="tokens/s", label="Throughput (tokens/s)")
        bout = gr.Markdown()
        bbtn.click(bench, [note, crit, mx, kb, which], [btab, bplot, bout])
        gr.Markdown("### Attention check: reference vs tiled online-softmax vs PyTorch SDPA")
        with gr.Row():
            aT = gr.Slider(16, 2048, value=256, step=16, label="Sequence length T")
            aD = gr.Dropdown([32, 64, 128], value=64, label="Head dim")
            aB = gr.Dropdown([16, 32, 64, 128], value=64, label="Tile size")
        abtn = gr.Button("Run attention check")
        atab = gr.Dataframe(interactive=False)
        abtn.click(attention_check, [aT, aD, aB], atab)
        logs = gr.Markdown()
    with gr.Tab("About"):
        gr.Markdown(open(ROOT / "app" / "ABOUT.md").read())
    demo.load(lambda: (status_md(), saved_logs()), None, [status, logs])

if __name__ == "__main__":
    _load()
    demo.queue(default_concurrency_limit=1).launch(server_name=os.environ.get("TS_HOST", "0.0.0.0"),
                                                   server_port=int(os.environ.get("TS_PORT", 7860)),
                                                   share=os.environ.get("TS_SHARE") == "1", css=CSS,
                                                   theme=THEME, js=FORCE_LIGHT)
