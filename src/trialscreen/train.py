"""LoRA supervised fine-tuning with a plain PyTorch loop (transparent and easy to defend).

Objective: next-token cross-entropy on the assistant answer only
    L = - (1/|A|) * sum_{t in A} log p_theta(y_t | y_<t, prompt)
where A are the target JSON tokens; prompt tokens are masked with label -100.
Only the low-rank matrices B (d x r) and A (r x k) of each adapted projection are
trained: W' = W + (alpha / r) * B A  (Hu et al., 2022).
"""
from __future__ import annotations
import json, math, random, time
from pathlib import Path
import torch
from .modeling import train_example


def collate(batch, pad_id):
    L = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, b in enumerate(batch):              # right padding
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"]); lab[i, :n] = torch.tensor(b["labels"]); att[i, :n] = 1
    return ids, lab, att


@torch.no_grad()
def eval_loss(model, examples, pad_id, bs, device):
    model.eval(); tot, n = 0.0, 0
    for i in range(0, len(examples), bs):
        ids, lab, att = collate(examples[i:i + bs], pad_id)
        ids, lab, att = ids.to(device), lab.to(device), att.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            out = model(input_ids=ids, attention_mask=att, labels=lab)
        ntok = int((lab[:, 1:] != -100).sum())
        tot += float(out.loss) * ntok; n += ntok
    model.train()
    return tot / max(n, 1)


def train_lora(cfg: dict, train_recs, val_recs, out_dir, log=print, val_eval_fn=None):
    from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(cfg["seed"]); random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(cfg["model_id"])
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    # fp32 master weights + fp16 autocast: T4/P100 have no bf16, and fp16 LoRA params
    # cannot be unscaled by GradScaler. 1.5B params * 4 B = ~6.2 GB, fits a 16 GB T4.
    model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], dtype=torch.float32).to(device)
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    lcfg = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
                      target_modules=cfg["target_modules"], task_type="CAUSAL_LM", bias="none")
    model = get_peft_model(model, lcfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"trainable params {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")

    tr = [train_example(tok, r, cfg["max_len"]) for r in train_recs]
    va = [train_example(tok, r, cfg["max_len"]) for r in val_recs]
    stats = {"train_max_tokens": max(len(e["input_ids"]) for e in tr),
             "val_max_tokens": max(len(e["input_ids"]) for e in va), "truncated_examples": 0}
    log(f"token stats {stats}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps_per_epoch = math.ceil(len(tr) / (cfg["batch_size"] * cfg["grad_accum"]))
    total_steps = steps_per_epoch * cfg["epochs"]
    sched = get_cosine_schedule_with_warmup(opt, int(cfg["warmup_ratio"] * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    history = {"step": [], "train_loss": [], "lr": [], "epoch_end": []}
    best, step = None, 0
    model.train(); t0 = time.time()
    for epoch in range(cfg["epochs"]):
        random.shuffle(tr)
        for b0 in range(0, len(tr), cfg["batch_size"] * cfg["grad_accum"]):
            chunk = tr[b0:b0 + cfg["batch_size"] * cfg["grad_accum"]]
            run = 0.0
            for m0 in range(0, len(chunk), cfg["batch_size"]):
                ids, lab, att = collate(chunk[m0:m0 + cfg["batch_size"]], pad_id)
                ids, lab, att = ids.to(device), lab.to(device), att.to(device)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                    loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
                scaler.scale(loss * len(ids) / len(chunk)).backward()
                run += loss.detach().item() * len(ids) / len(chunk)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg["max_grad_norm"])
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
            step += 1
            history["step"].append(step); history["train_loss"].append(run); history["lr"].append(sched.get_last_lr()[0])
            if step % cfg["log_every"] == 0:
                log(f"epoch {epoch + 1} step {step}/{total_steps} loss {run:.4f} lr {sched.get_last_lr()[0]:.2e} "
                    f"elapsed {time.time() - t0:.0f}s")
        vloss = eval_loss(model, va, pad_id, cfg["batch_size"], device)
        rec = {"epoch": epoch + 1, "step": step, "val_loss": vloss}
        if val_eval_fn is not None:            # label/evidence metrics on the VALIDATION split only
            rec.update(val_eval_fn(model, tok))
        history["epoch_end"].append(rec)
        log(f"== epoch {epoch + 1} {rec}")
        score = rec.get(cfg["select_metric"], -vloss)
        if best is None or score > best[0]:
            best = (score, epoch + 1)
            model.save_pretrained(out_dir / "best_adapter"); tok.save_pretrained(out_dir / "best_adapter")
            log(f"saved best adapter (epoch {epoch + 1}, {cfg['select_metric']}={score:.4f})")
    model.save_pretrained(out_dir / "last_adapter")
    meta = {"config": cfg, "token_stats": stats, "trainable_params": trainable, "total_params": total,
            "best_epoch": best[1], "best_score": best[0], "train_minutes": (time.time() - t0) / 60,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None}
    json.dump(history, open(out_dir / "history.json", "w"), indent=1)
    json.dump(meta, open(out_dir / "train_meta.json", "w"), indent=1)
    return out_dir / "best_adapter", history, meta
