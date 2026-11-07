# TrialScreen

Criterion-level clinical trial eligibility screening with evidence citations, built for
Generative AI Assignment 2. A fine-tuned open-weight decoder (Qwen2.5-1.5B-Instruct + LoRA) reads a
patient note and one inclusion/exclusion criterion and returns the expert label category and the
supporting sentence IDs. Around it: an explicit KV-cached decoder, tiled online-softmax attention,
greedy speculative decoding and a browser application.

> **Educational screening prototype. Not a clinical enrollment decision tool.**

| Assignment task | Where |
|---|---|
| 1. Research, model choice, LoRA fine-tuning | `report/main.tex`, `src/trialscreen/train.py`, `scripts/02_train_lora.py`, `configs/` |
| 2. Labels + evidence, metrics | `src/trialscreen/{parsing,metrics,explain}.py`, `scripts/03_evaluate.py` |
| 3. KV-cached decoding | `src/trialscreen/decoding.py`, `scripts/04_bench_kv.py` |
| 4. Tiled online-softmax attention | `src/trialscreen/attention.py`, `scripts/05_attention.py` |
| 5. Speculative decoding | `src/trialscreen/speculative.py`, `scripts/06_bench_speculative.py` |
| 6. Browser application | `app/app.py` |
| Split manifest | `data/splits/split_manifest.json` (+ `train/validation/test.jsonl`) |
| Tests | `tests/` (51 tests; log in `results/cpu_smoke/pytest_output.txt`) |
| Report | `report/main.tex` (IEEE, LaTeX), tables/figures generated into `report/tables`, `report/figures` |

## 1. What is already done and verified (CPU)

* The fixed split is already generated: 37/8/8 patients, 602/206/206 usable annotations
  (annotation 883 has an empty criterion text in the released file and is excluded, see manifest).
* `pytest` passes (51 tests): split reproducibility and leakage, metrics, parsing, attention vs a
  float64 reference, cached vs uncached greedy equality, speculative == greedy for full/partial/zero
  acceptance and EOS handling.
* Our hand-written decoder was checked on the real Qwen2.5-0.5B-Instruct:
  `results/cpu_smoke/real_qwen0.5b_explicit_decoder_check.json` (max logit difference 2e-4, identical tokens).

Everything that needs a GPU (training, held-out results, T4 benchmarks) is produced by **one notebook**.

## 2. Run the full pipeline on a free Kaggle GPU (about 60 to 90 minutes)

1. Put this folder on GitHub (Codespaces: upload the zip, `unzip`, `git init`, commit, push).
2. On Kaggle: **New Notebook → File → Import notebook →** `notebooks/kaggle_run_all.ipynb`.
3. Right panel: **Accelerator = GPU T4 x1** (or x2), **Internet = On**.
4. In the first code cell set `REPO = "https://github.com/<you>/trialscreen.git"`.
5. Add a Kaggle secret `HF_TOKEN` (Add-ons → Secrets) holding a Hugging Face **write** token, and set
   `HF_REPO` in step 9 of the notebook.
6. **Run All.** It runs, in order:

| Step | Command | Output |
|---|---|---|
| data + checks | `python scripts/01_prepare_data.py` and `pytest` | `data/splits/*` |
| fine-tune | `python scripts/02_train_lora.py` | `checkpoints/.../best_adapter`, `results/train_*.json` |
| before/after | `python scripts/03_evaluate.py --split validation` | `results/metrics_*_validation.json` |
| final test (once) | `python scripts/03_evaluate.py --split test --final` | `results/metrics_*_test.json`, predictions |
| Task 3 | `python scripts/04_bench_kv.py` | `results/bench_kv.json` |
| Task 4 | `python scripts/05_attention.py` | `results/attention.json` |
| Task 5 | `python scripts/06_bench_speculative.py` | `results/bench_speculative.json` |
| report assets | `python scripts/07_make_report_assets.py` | `report/tables/*.tex`, `report/figures/*.png`, `results/error_cases.md` |
| publish adapter | `python scripts/08_push_adapter.py --repo <you>/qwen2.5-1.5b-trialscreen-lora` | adapter on the HF Hub |

7. Download `/kaggle/working/trialscreen_outputs.zip`, copy `results/`, `report/tables`, `report/figures`
   into your repository and commit them (these are the benchmark logs the assignment asks for).

The test split is only evaluated by the command with `--final`; never tune anything after running it.

## 3. Run the application (no notebook needed)

**Option A: free Hugging Face Space (recommended for graders, always online).**
1. huggingface.co → New Space → SDK *Gradio*, hardware *CPU basic (free)*.
2. Upload the repository files; replace the Space's `README.md` with `deploy/space_README.md`.
3. Space → Settings → Variables: `TS_ADAPTER = <you>/qwen2.5-1.5b-trialscreen-lora`.
4. The Space builds and opens in the browser. On free CPU a screening of ~20 criteria takes a few minutes;
   the progress bar shows each criterion.

**Option B: GitHub Codespaces or any machine.**
```bash
pip install -r requirements.txt
TS_ADAPTER=<you>/qwen2.5-1.5b-trialscreen-lora python app/app.py
```
Open the forwarded port 7860 (Codespaces opens it automatically). Use a 4-core/16 GB Codespace;
`TS_DTYPE=bfloat16` halves memory on CPU. With a GPU the app uses fp16/bf16 automatically.

**Using the app.** *Screen a patient*: load an example (validation patients or a hand-written case) or
paste a note and criteria under `Inclusion criteria:` / `Exclusion criteria:` headings, choose a decoding
mode, press **Screen patient**, inspect labels, highlighted evidence, explanations and missing information,
and download the HTML/JSON report. *Inference benchmark*: runs no-cache, explicit KV cache, tiled attention,
HF cache and speculative decoding live on the same prompt, checks the tokens are identical and times them;
plus a live attention correctness check.

## 4. Run tests locally (CPU, about 10 s)
```bash
pip install -r requirements.txt && python -m pytest -q tests
```

## 5. Design notes
* **Labels**: inclusion {included, not included, not enough information, not applicable}; exclusion
  {excluded, not excluded, not enough information, not applicable}. Never collapsed to yes/no.
* **No GPT-4 leakage**: GPT-4 columns are dropped on load (and a test checks it). Explanations are built
  only from the predicted label and the cited sentences, because the dataset's free-text rationales are GPT-4 outputs.
* **No silent truncation**: the longest example is 610 tokens; longer inputs raise an error instead.
* **Trial-level outcome** in the app is an explicit rule over criterion predictions, not a labelled outcome.

## 6. AI assistance
Built with help from Claude (Anthropic) for code drafting, tests and report drafting. See the AI-use
appendix in the report. The author reviewed, ran and is responsible for all code and results.

## 7. Licences
Code: MIT. Dataset: ncbi/TrialGPT-Criterion-Annotations (US public domain notice). Models: Qwen2.5 (Apache-2.0).
