"""Upload the best LoRA adapter (~70 MB) and training logs to the Hugging Face Hub so the
app and graders can download it. Needs a free HF account and a write token:
    huggingface-cli login     (or set HF_TOKEN)
    python scripts/08_push_adapter.py --repo <your-username>/qwen2.5-1.5b-trialscreen-lora"""
import argparse
from _common import ROOT
from huggingface_hub import HfApi

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--adapter", default=str(ROOT / "checkpoints" / "qwen2.5-1.5b-trialscreen-lora" / "best_adapter"))
args = ap.parse_args()
api = HfApi()
api.create_repo(args.repo, repo_type="model", exist_ok=True)
card = f"""---
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
tags: [lora, clinical-trials, eligibility, educational]
---
# TrialScreen LoRA adapter (educational)

LoRA adapter for Qwen2.5-1.5B-Instruct trained on the 37-patient training split of
ncbi/TrialGPT-Criterion-Annotations (seed-42 patient split). Predicts a criterion-level eligibility
label and supporting sentence IDs as JSON. **Research prototype, not for clinical use.**
"""
(ROOT / "checkpoints" / "MODEL_CARD.md").write_text(card)
api.upload_folder(folder_path=args.adapter, repo_id=args.repo, repo_type="model")
api.upload_file(path_or_fileobj=str(ROOT / "checkpoints" / "MODEL_CARD.md"), path_in_repo="README.md", repo_id=args.repo)
for f in ("train_meta.json", "train_history.json", "train_log.txt"):
    p = ROOT / "results" / f
    if p.exists():
        api.upload_file(path_or_fileobj=str(p), path_in_repo=f"logs/{f}", repo_id=args.repo)
print(f"done: https://huggingface.co/{args.repo}")
