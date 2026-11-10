# TrialScreen

**Criterion-level clinical trial eligibility screening with a small open language model.**

TrialScreen reads a patient note and a trial's inclusion and exclusion criteria and, for every criterion,
predicts one of the expert label categories and cites the patient sentences that support the decision.
The model is Qwen2.5-1.5B-Instruct adapted with LoRA on the TrialGPT Criterion Annotations dataset.
Around it the project implements an explicit KV-cached decoder, tiled attention with an online softmax,
greedy speculative decoding and a browser application.

> **Educational screening prototype. It is not a clinical enrollment decision tool.**

| | Link |
|---|---|
| Report (IEEE format) | [`report/main.pdf`](report/main.pdf), LaTeX source [`report/main.tex`](report/main.tex) |
| Fine-tuned LoRA adapter | https://huggingface.co/eman-huda/qwen2.5-1.5b-trialscreen-lora |
| Dataset | https://huggingface.co/datasets/ncbi/TrialGPT-Criterion-Annotations |
| Split manifest | [`data/splits/split_manifest.json`](data/splits/split_manifest.json) |
| Benchmark and evaluation logs | [`results/`](results) |
| How to use the app | [`docs/APP_GUIDE.md`](docs/APP_GUIDE.md) |

## Results (NVIDIA T4, greedy decoding)

**Criterion classification and evidence, held-out test split (8 patients, 206 annotations)**

| Model | Accuracy | Macro-F1 | Inclusion acc. | Exclusion acc. | Evidence F1 | Valid JSON |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct (base) | 0.165 | 0.170 | 0.400 | 0.031 | 0.158 | 100% |
| + LoRA (this work) | **0.743** | **0.534** | **0.800** | **0.710** | **0.571** | 100% |

**Inference**

| Component | Result |
|---|---|
| KV cache | identical greedy tokens to the uncached reference (fp16 differences only at exact logit ties, all equal in fp32); up to **10.1x** faster (1,024-token prompt, 256 generated) |
| Tiled online-softmax attention | max fp32 error 1.1e-6 vs an fp64 reference; **41 MB vs 2.39 GB** at 4,096 tokens; slower than the fused PyTorch kernel (expected) |
| Speculative decoding (0.5B draft) | token-identical to greedy on all 20 screening prompts; 66 to 79% draft acceptance; 0.61 to 0.86x the speed of greedy decoding at batch size 1 (explained in the report) |

## Repository layout

```
src/trialscreen/
  data.py          dataset loading, fixed 37/8/8 patient split, sentence numbering, prompts
  train.py         LoRA fine-tuning loop (loss on answer tokens only)
  modeling.py      model loading, prompt tokenisation, criterion-level prediction
  parsing.py       strict JSON parsing of model answers
  metrics.py       accuracy, macro-F1, per-class scores, confusion matrices, evidence P/R/F1
  explain.py       extractive explanations and the rule-based trial outcome
  decoding.py      hand-written Qwen2 decoder with an explicit KV cache; cached and uncached greedy loops
  attention.py     reference attention and tiled online-softmax attention
  speculative.py   greedy speculative decoding with cache rollback
  report_export.py HTML report with provenance
app/app.py         Gradio browser application
scripts/           01 data -> 02 train -> 03 evaluate -> 04 KV cache -> 05 attention -> 06 speculative
                   -> 07 report tables/figures -> 08 upload adapter
notebooks/         kaggle_run_all.ipynb (full pipeline), launch_app_kaggle.ipynb, launch_app_colab.ipynb
tests/             51 unit tests (split, metrics, attention, KV cache, speculative decoding)
results/           all metrics, predictions, benchmark logs and the error-analysis worksheet
report/            IEEE LaTeX report, tables and figures
```

## Run the application

Hugging Face Spaces that run a Python server now require a paid plan, so the app is served from a free GPU
notebook and shared through a public Gradio link. The person using the app only needs the link.

**Kaggle:** New Notebook → File → Import Notebook → `notebooks/launch_app_kaggle.ipynb`.
Settings: GPU T4, Internet On. Run All, then open the printed `https://….gradio.live` link.

**Google Colab:** File → Upload notebook → `notebooks/launch_app_colab.ipynb`.
Runtime → Change runtime type → T4 GPU. Run all, then open the printed link.

**Any machine with Python 3.10+:**
```bash
pip install -r requirements.txt
TS_ADAPTER=eman-huda/qwen2.5-1.5b-trialscreen-lora python app/app.py     # http://127.0.0.1:7860
```
Add `TS_SHARE=1` for a public link. On CPU, `TS_DTYPE=bfloat16` halves memory.

Step-by-step usage: [`docs/APP_GUIDE.md`](docs/APP_GUIDE.md).

## Reproduce the experiments

Everything that needs a GPU runs in one notebook on a free Kaggle T4 (about 75 minutes).

1. Kaggle → New Notebook → File → Import Notebook → `notebooks/kaggle_run_all.ipynb`.
2. Settings: GPU T4, Internet On. Optional: add a Kaggle secret `HF_TOKEN` to upload the adapter.
3. Save Version → Save & Run All (Commit). Download `trialscreen_outputs.zip` from the version's Output tab.

The notebook runs, in order:

| Step | Command | Output |
|---|---|---|
| Data and checks | `python scripts/01_prepare_data.py`, `pytest tests` | `data/splits/` |
| Fine-tuning | `python scripts/02_train_lora.py` | adapter, `results/train_*.json` |
| Evaluation (validation) | `python scripts/03_evaluate.py --split validation` | `results/metrics_*_validation.json` |
| Evaluation (test, once) | `python scripts/03_evaluate.py --split test --final` | `results/metrics_*_test.json` |
| KV cache | `python scripts/04_bench_kv.py` | `results/bench_kv.json` |
| Attention | `python scripts/05_attention.py` | `results/attention.json` |
| Speculative decoding | `python scripts/06_bench_speculative.py` | `results/bench_speculative.json` |
| Report assets | `python scripts/07_make_report_assets.py` | `report/tables/`, `report/figures/`, `results/error_cases.md` |

The test split can only be evaluated with `--final`; no choice was tuned on it.

**Unit tests (CPU, about 10 seconds):** `pip install -r requirements.txt && python -m pytest -q tests`

## Design decisions

* **Data integrity.** Patient IDs are sorted and shuffled with `numpy.random.default_rng(42)`; 37 patients
  train, 8 validate, 8 test. GPT-4 columns are removed on load and a test checks they are absent.
  Annotation 883 has an empty criterion text and is excluded (recorded in the manifest).
* **Labels.** Inclusion: included, not included, not enough information, not applicable. Exclusion:
  excluded, not excluded, not enough information, not applicable. They are never collapsed to yes/no.
* **Output.** `{"evidence": [sentence IDs], "label": "..."}`, evidence first. Explanations are built only
  from the predicted label and the cited sentences, so they always point to real text in the record.
* **No silent truncation.** The longest example is 610 tokens; longer inputs raise an error.
* **Trial outcome.** A rule over criterion predictions, not a labelled result: any "excluded" or
  "not included" gives *likely ineligible*; otherwise any "not enough information" gives *undetermined*;
  otherwise *potentially eligible*.

## Limitations

Eight test patients give wide uncertainty, and the "excluded" class does not occur in the test split.
Patients are synthetic and notes are short. The model under-predicts "not enough information" for exclusion
criteria. The custom decoders support batch size 1. See the report for details.

## Acknowledgements

Dataset and task from TrialGPT (Jin et al., *Nature Communications*, 2024). Claude (Anthropic) was used as a
programming and writing assistant, as declared in the report's appendix; all experiments were run and all
results verified by the author.

## 6. Hugging Face Model Link
https://huggingface.co/eman-huda/qwen2.5-1.5b-trialscreen-lora

## Licence

Code: MIT. Dataset: see its Hugging Face page. Qwen2.5 models: Apache-2.0.
