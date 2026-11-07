import argparse, statistics, torch
from _common import ROOT
from trialscreen.data import load_jsonl


def base_args(desc):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=str(ROOT / "checkpoints" / "qwen2.5-1.5b-trialscreen-lora" / "best_adapter"),
                    help="LoRA adapter merged into the target before benchmarking ('none' to skip)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="tiny settings for CPU smoke tests")
    return ap


def long_prompt_ids(tok, n_tokens, instruction="Summarise this patient's history in detail."):
    """Build a chat prompt of ~n_tokens from VALIDATION notes (never test)."""
    notes = [r.note for r in load_jsonl(ROOT / "data" / "splits" / "validation.jsonl")]
    seen, body = set(), ""
    for n in notes:
        if n in seen:
            continue
        seen.add(n)
        body += n + "\n"
        if len(tok(body).input_ids) > n_tokens:
            break
    while len(tok(body).input_ids) < n_tokens:      # tiny models / very long targets
        body += body
    msgs = [{"role": "user", "content": f"{instruction}\n\n{body}"}]
    ids = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
              add_special_tokens=False).input_ids
    # trim the note body from the front to hit the target length exactly, keep the template tail
    if len(ids) > n_tokens:
        ids = ids[:8] + ids[len(ids) - (n_tokens - 8):]
    return torch.tensor([ids])


def median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None
