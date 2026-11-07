"""Task 4: reference scaled dot-product attention and tiled online-softmax attention.

Shapes: q [B, H, Tq, D], k and v [B, H, Tk, D] (grouped-query attention is handled by the
caller, which repeats K/V heads). With a KV cache, Tq <= Tk and the queries are the LAST Tq
positions of the sequence, so query i sits at absolute position (Tk - Tq + i) and may attend
to key j iff j <= Tk - Tq + i ("bottom-right aligned" causal mask).

Both functions accumulate in float32 (float64 for float64 inputs) and cast back to the input dtype.
"""
from __future__ import annotations
import math
import torch


def causal_mask(tq: int, tk: int, device=None) -> torch.Tensor:
    """Boolean [tq, tk]; True = may attend."""
    offset = tk - tq
    qi = torch.arange(tq, device=device).unsqueeze(1) + offset
    kj = torch.arange(tk, device=device).unsqueeze(0)
    return kj <= qi


def reference_attention(q, k, v, causal: bool = True, *, scale: float | None = None):
    """Materialises the full [Tq, Tk] score matrix.
    softmax(x)_j = exp(x_j - max x) / sum_l exp(x_l - max x)  (max subtraction for stability)."""
    d = q.shape[-1]
    scale = scale if scale is not None else 1.0 / math.sqrt(d)
    acc = torch.float64 if q.dtype == torch.float64 else torch.float32
    qf, kf, vf = q.to(acc), k.to(acc), v.to(acc)
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale            # [B,H,Tq,Tk]
    if causal:
        m = causal_mask(q.shape[-2], k.shape[-2], q.device)
        scores = scores.masked_fill(~m, float("-inf"))
    row_max = scores.amax(dim=-1, keepdim=True)
    row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))  # fully-masked rows
    p = torch.exp(scores - row_max)
    denom = p.sum(dim=-1, keepdim=True)
    out = torch.matmul(p, vf) / denom.clamp_min(1e-30)
    return out.to(q.dtype)


def tiled_attention(q, k, v, causal: bool = True, *, scale: float | None = None,
                    block_q: int = 64, block_k: int = 64):
    """Blocked attention with online softmax (Milakov & Gimelshein 2018; FlashAttention, Dao et al. 2022).

    For each query tile we stream over key/value tiles keeping, per query row:
        m  = running maximum of the scores seen so far
        l  = running sum of exp(score - m)
        acc = running sum of exp(score - m) * v
    When a new tile raises the maximum from m to m_new, previous terms are rescaled by
    exp(m - m_new). At the end out = acc / l. Only [block_q, block_k] scores exist at a time,
    so memory is O(Tq*D + block_q*block_k) instead of O(Tq*Tk).
    Key tiles that lie entirely in the causal future of a query tile are skipped.
    """
    B, H, Tq, D = q.shape
    Tk = k.shape[-2]
    scale = scale if scale is not None else 1.0 / math.sqrt(D)
    offset = Tk - Tq
    acc_t = torch.float64 if q.dtype == torch.float64 else torch.float32
    out = torch.empty(B, H, Tq, D, dtype=acc_t, device=q.device)
    for qs in range(0, Tq, block_q):
        qe = min(qs + block_q, Tq)
        qb = q[:, :, qs:qe].to(acc_t) * scale                                 # [B,H,bq,D]
        m = torch.full((B, H, qe - qs, 1), float("-inf"), device=q.device, dtype=acc_t)
        l = torch.zeros((B, H, qe - qs, 1), device=q.device, dtype=acc_t)
        acc = torch.zeros((B, H, qe - qs, D), device=q.device, dtype=acc_t)
        last_visible = (qe - 1) + offset   # largest key index any row in this tile may see
        for ks in range(0, Tk, block_k):
            if causal and ks > last_visible:
                break                       # whole tile is in the future: skip
            ke = min(ks + block_k, Tk)
            kb, vb = k[:, :, ks:ke].to(acc_t), v[:, :, ks:ke].to(acc_t)
            s = torch.matmul(qb, kb.transpose(-1, -2))                      # [B,H,bq,bk]
            if causal and ke - 1 > qs + offset:   # tile crosses the diagonal: apply mask
                qi = torch.arange(qs, qe, device=q.device).unsqueeze(1) + offset
                kj = torch.arange(ks, ke, device=q.device).unsqueeze(0)
                s = s.masked_fill(kj > qi, float("-inf"))
            m_new = torch.maximum(m, s.amax(dim=-1, keepdim=True))
            m_safe = torch.where(torch.isfinite(m_new), m_new, torch.zeros_like(m_new))
            alpha = torch.exp(torch.where(torch.isfinite(m), m - m_safe, torch.full_like(m, float("-inf"))))
            p = torch.exp(s - m_safe)
            l = alpha * l + p.sum(dim=-1, keepdim=True)
            acc = alpha * acc + torch.matmul(p, vb)
            m = m_new
        out[:, :, qs:qe] = acc / l.clamp_min(1e-30)
    return out.to(q.dtype)


def torch_sdpa(q, k, v, causal: bool = True, *, scale: float | None = None):
    """PyTorch fused/optimised backend, used only as a comparison point."""
    import torch.nn.functional as F
    if causal and q.shape[-2] != k.shape[-2]:
        mask = causal_mask(q.shape[-2], k.shape[-2], q.device)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)


BACKENDS = {"reference": reference_attention, "tiled": tiled_attention, "torch_sdpa": torch_sdpa}
