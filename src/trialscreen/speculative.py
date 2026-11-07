"""Task 5: greedy speculative decoding (Leviathan et al. 2023; Chen et al. 2023), greedy case.

Invariant kept between rounds (both models):
    seq = prompt + generated tokens.
    target cache holds K/V for seq[:-1]   (everything except the newest token)
    draft cache holds K/V for a prefix of seq[:-1]; missing tokens are fed before drafting.

One round with lookahead k:
  1. Draft: feed the tokens the draft has not processed yet (ends with seq[-1]) and take
     argmax; then feed each drafted token to get the next, producing d_1..d_k.
  2. Verify: ONE target forward on [seq[-1], d_1, ..., d_k] (k+1 tokens). Its argmaxes
     p_0..p_k are what plain greedy target decoding would produce after seq, seq+d_1, ...
  3. Accept the longest prefix with d_i == p_{i-1}; say n tokens. Append d_1..d_n and then
     p_n (the target's correction on the first mismatch, or a free "bonus" token if all
     k drafts were accepted). Every emitted token is a target argmax, so the output equals
     greedy target decoding token for token (up to floating-point ties, see report).
  4. Roll back: crop the target cache to len(seq_new) - 1 and the draft cache to at most
     that length, discarding K/V written for rejected draft tokens.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
import torch
from .decoding import sync, reset_peak, peak_mem_mb


@dataclass
class SpecStats:
    tokens: list = field(default_factory=list)
    rounds: int = 0
    drafted: int = 0
    accepted: int = 0
    accepted_per_round: list = field(default_factory=list)
    target_forwards: int = 0
    draft_forwards: int = 0
    prefill_s: float = 0.0
    total_s: float = 0.0
    peak_mem_mb: float | None = None

    @property
    def acceptance_rate(self):
        return self.accepted / self.drafted if self.drafted else 0.0

    @property
    def tokens_per_round(self):
        return len(self.tokens) / self.rounds if self.rounds else 0.0

    def as_dict(self):
        return {"new_tokens": len(self.tokens), "rounds": self.rounds, "drafted": self.drafted,
                "accepted": self.accepted, "acceptance_rate": self.acceptance_rate,
                "tokens_per_round": self.tokens_per_round, "target_forwards": self.target_forwards,
                "draft_forwards": self.draft_forwards, "prefill_s": self.prefill_s, "total_s": self.total_s,
                "tokens_per_s": len(self.tokens) / self.total_s if self.total_s else None,
                "peak_mem_mb": self.peak_mem_mb}


@torch.no_grad()
def speculative_greedy(target, draft, prompt_ids: torch.Tensor, max_new_tokens: int, k: int = 4, eos_ids=()):
    device = prompt_ids.device
    target.reset(); draft.reset(); reset_peak(device)
    st = SpecStats()
    sync(device); t0 = time.perf_counter()
    # prefill both models on the prompt; the target's last logit gives the first token
    logits = target.forward(prompt_ids); st.target_forwards += 1
    draft.forward(prompt_ids[:, :-1]) if prompt_ids.shape[1] > 1 else None
    st.draft_forwards += 1
    seq = prompt_ids[0].tolist()
    first = int(logits[0, -1].argmax())
    seq.append(first); st.tokens.append(first)
    # the target already holds K/V for the whole prompt, i.e. seq[:-1]: invariant holds
    sync(device); st.prefill_s = time.perf_counter() - t0

    while len(st.tokens) < max_new_tokens and st.tokens[-1] not in eos_ids:
        k_eff = min(k, max_new_tokens - len(st.tokens))
        # ---- 1. draft k tokens
        pending = seq[draft.cache_len:]                      # tokens draft has not processed (ends with seq[-1])
        dl = draft.forward(torch.tensor([pending], device=device)); st.draft_forwards += 1
        drafts = [int(dl[0, -1].argmax())]
        while len(drafts) < k_eff and drafts[-1] not in eos_ids:
            dl = draft.forward(torch.tensor([[drafts[-1]]], device=device)); st.draft_forwards += 1
            drafts.append(int(dl[0, -1].argmax()))
        # ---- 2. verify with ONE target pass over [seq[-1], d_1..d_k]
        base = len(seq) - 1                                  # == target.cache_len
        assert target.cache_len == base, (target.cache_len, base)
        tl = target.forward(torch.tensor([[seq[-1]] + drafts], device=device)); st.target_forwards += 1
        preds = tl[0].argmax(-1).tolist()                    # p_0..p_k
        # ---- 3. accept the longest matching prefix, then add the target's own next token
        n = 0
        while n < len(drafts) and drafts[n] == preds[n]:
            n += 1
        new = drafts[:n] + [preds[n]]
        st.rounds += 1; st.drafted += len(drafts); st.accepted += n; st.accepted_per_round.append(n)
        for tok in new:
            seq.append(tok); st.tokens.append(tok)
            if tok in eos_ids or len(st.tokens) >= max_new_tokens:
                break
        # ---- 4. roll caches back to seq[:-1]
        target.crop(len(seq) - 1)
        draft.crop(min(draft.cache_len, len(seq) - 1))
    sync(device); st.total_s = time.perf_counter() - t0
    st.peak_mem_mb = peak_mem_mb(device)
    st.tokens = st.tokens[:max_new_tokens]
    return st
