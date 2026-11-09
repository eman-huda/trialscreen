"""Model loading, prompt tokenisation and criterion-level prediction."""
from __future__ import annotations
import time
import torch
from .data import build_messages, target_json, split_sentences
from .parsing import parse_output
from .decoding import ExplicitQwen2Decoder, HFCacheDecoder, greedy_cached, greedy_nocache
from .explain import explain

TARGET_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DRAFT_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def pick_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def pick_dtype(device):
    """bf16 only on GPUs with native support (compute capability >= 8, e.g. A100/L4).
    On a T4 (7.5) torch.cuda.is_bf16_supported() can return True because bf16 is
    *emulated*, which is several times slower than fp16, so we use fp16 there."""
    if device == "cuda":
        major, _ = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32


def load_model(model_id=TARGET_ID, adapter=None, device=None, dtype=None, merge=True):
    """Load tokenizer + causal LM. If `adapter` (local dir or Hub id) is given, apply the
    LoRA adapter and, by default, merge it into the base weights so the explicit decoder
    sees ordinary Linear layers."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    device = device or pick_device()
    dtype = dtype or pick_dtype(device)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        if merge:
            model = model.merge_and_unload()
        model.eval()
    return tok, model


def eos_ids(tok):
    ids = {tok.eos_token_id}
    for t in ("<|im_end|>", "<|endoftext|>"):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0:
            ids.add(i)
    return tuple(i for i in ids if i is not None)


def prompt_ids(tok, note_numbered, criterion_type, criterion_text):
    msgs = build_messages(note_numbered, criterion_type, criterion_text)
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(text, add_special_tokens=False).input_ids


def train_example(tok, rec, max_len=1536):
    """prompt ids + target ids; loss only on target tokens (labels -100 on the prompt).
    Truncation is never silent: an example longer than max_len raises, because cutting
    the note could remove annotated evidence sentences."""
    p = prompt_ids(tok, rec.note, rec.criterion_type, rec.criterion_text)
    t = tok(target_json(rec.label, rec.evidence) + "<|im_end|>", add_special_tokens=False).input_ids
    if len(p) + len(t) > max_len:
        raise ValueError(f"annotation {rec.annotation_id}: {len(p) + len(t)} tokens > max_len {max_len}")
    return {"input_ids": p + t, "labels": [-100] * len(p) + t, "prompt_len": len(p), "target_len": len(t)}


def make_decoder(model, mode="hf_cache", attn_backend="reference"):
    if mode.startswith("explicit"):
        return ExplicitQwen2Decoder(model, attn_backend=attn_backend)
    return HFCacheDecoder(model)


@torch.no_grad()
def predict_one(tok, model, note, criterion_type, criterion_text, mode="hf_cache",
                attn_backend="reference", max_new_tokens=48, draft_model=None, k=4, decoder=None,
                draft_decoder=None):
    """mode: hf_cache | explicit_cache | explicit_nocache | speculative"""
    sents = split_sentences(note)
    numbered = "\n".join(f"{i}. {s}" for i, s in sents)
    ids = torch.tensor([prompt_ids(tok, numbered, criterion_type, criterion_text)], device=model.device)
    eos = eos_ids(tok)
    t0 = time.perf_counter()
    extra = {}
    if mode == "speculative":
        from .speculative import speculative_greedy
        tgt = decoder or ExplicitQwen2Decoder(model, attn_backend=attn_backend)
        dr = draft_decoder or ExplicitQwen2Decoder(draft_model, attn_backend=attn_backend)
        st = speculative_greedy(tgt, dr, ids, max_new_tokens, k=k, eos_ids=eos)
        extra = st.as_dict()
    else:
        dec = decoder or make_decoder(model, mode, attn_backend)
        fn = greedy_nocache if mode.endswith("nocache") else greedy_cached
        st = fn(dec, ids, max_new_tokens, eos_ids=eos)
        extra = st.as_dict()
    text = tok.decode([t for t in st.tokens if t not in eos], skip_special_tokens=True)
    parsed = parse_output(text, criterion_type, len(sents))
    sent_map = dict(sents)
    return {
        "criterion_type": criterion_type, "criterion_text": criterion_text,
        "label": parsed["label"], "evidence": parsed["evidence"],
        "evidence_text": [sent_map[i] for i in parsed["evidence"]],
        "explanation": explain(criterion_type, criterion_text, parsed["label"], parsed["evidence"], sent_map),
        "raw_output": text, "valid_json": parsed["valid_json"], "parse_notes": parsed["notes"],
        "prompt_tokens": ids.shape[1], "latency_s": time.perf_counter() - t0, "decode": extra,
    }
