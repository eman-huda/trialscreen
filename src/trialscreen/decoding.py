"""Task 3: explicit KV-cached autoregressive decoding.

Two decoders share one interface (forward(new_tokens) -> logits for those tokens,
cache_len, crop(n), reset()):

* ExplicitQwen2Decoder: our own forward pass for the Qwen2 architecture (RMSNorm, RoPE,
  grouped-query attention, SwiGLU MLP) that reads the weights of a loaded Hugging Face
  model and keeps its OWN pre-allocated key/value cache. Positional handling, causal
  masking and cache growth are written out explicitly here; attention can use the
  reference, tiled online-softmax (Task 4) or PyTorch SDPA backend.
* HFCacheDecoder: drives the Hugging Face model forward pass with an explicit
  DynamicCache, passing position ids / cache positions ourselves. Used as an
  independent cross-check and as the fast path for bulk evaluation.

greedy_nocache() is the reference path without cache reuse: every step recomputes the
whole sequence from scratch. greedy_cached() prefills once and then feeds one token per step.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
import torch
import torch.nn.functional as F
from .attention import BACKENDS


# ----------------------------------------------------------------------------- utils
def sync(device):
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize()


def reset_peak(device):
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()


def peak_mem_mb(device):
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        return torch.cuda.max_memory_allocated() / 2**20
    return None


# ----------------------------------------------------------------------------- explicit cache
class ExplicitKVCache:
    """Pre-allocated per-layer K/V buffers of shape [B, n_kv_heads, max_len, head_dim].

    `length` is the number of valid positions. Appending writes new keys/values at
    [length, length+T) and returns a view over [0, length+T). Cropping (needed when
    speculative drafts are rejected) only moves `length` back: stale entries beyond it
    are simply overwritten later, so no memory is freed or copied.
    Cache size = 2 * layers * kv_heads * head_dim * max_len * bytes_per_element.
    """

    def __init__(self, n_layers, batch, n_kv, max_len, head_dim, dtype, device):
        self.k = [torch.zeros(batch, n_kv, max_len, head_dim, dtype=dtype, device=device) for _ in range(n_layers)]
        self.v = [torch.zeros(batch, n_kv, max_len, head_dim, dtype=dtype, device=device) for _ in range(n_layers)]
        self.length = 0
        self.max_len = max_len

    def update(self, layer, k_new, v_new, start):
        T = k_new.shape[2]
        if start + T > self.max_len:
            raise RuntimeError(f"KV cache overflow: need {start + T} > max_len {self.max_len}")
        self.k[layer][:, :, start:start + T] = k_new
        self.v[layer][:, :, start:start + T] = v_new
        return self.k[layer][:, :, :start + T], self.v[layer][:, :, :start + T]

    def crop(self, n):
        self.length = min(self.length, n)

    def nbytes_used(self):
        per_pos = sum(t[:, :, :1].numel() * t.element_size() for t in self.k + self.v)
        return per_pos * self.length


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class ExplicitQwen2Decoder:
    """Hand-written Qwen2 forward pass with an explicit KV cache (batch size 1)."""

    def __init__(self, hf_model, attn_backend="reference", max_len=2048):
        cfg = hf_model.config
        self.cfg = cfg
        self.m = hf_model.model if hasattr(hf_model, "model") else hf_model
        self.lm_head = hf_model.lm_head
        self.n_layers = cfg.num_hidden_layers
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.eps = cfg.rms_norm_eps
        theta = getattr(cfg, "rope_theta", None)
        if theta is None:   # newer transformers keep it in rope_parameters
            theta = (getattr(cfg, "rope_parameters", None) or {}).get("rope_theta", 10000.0)
        self.theta = float(theta)
        self.attn = BACKENDS[attn_backend]
        self.attn_backend = attn_backend
        self.device = self.lm_head.weight.device
        self.dtype = self.lm_head.weight.dtype
        self.max_len = max_len
        # RoPE inverse frequencies: theta^(-2i/d), i = 0..d/2-1
        self.inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.head_dim, 2, device=self.device,
                                                            dtype=torch.float32) / self.head_dim))
        self.reset()

    # -- interface ------------------------------------------------------------
    def reset(self):
        self.cache = ExplicitKVCache(self.n_layers, 1, self.n_kv, self.max_len, self.head_dim,
                                     self.dtype, self.device)

    @property
    def cache_len(self):
        return self.cache.length

    def crop(self, n):
        self.cache.crop(n)

    @torch.no_grad()
    def forward(self, ids: torch.Tensor, use_cache: bool = True, logits_to_keep: int = 0):
        """ids [1, T] are the NEW tokens. Their absolute positions are
        cache_len .. cache_len+T-1. Returns logits [1, T, vocab] (float32), or only the last
        `logits_to_keep` positions when it is > 0 (greedy decoding needs just the last one; the
        full [T, 151936] float32 logits tensor would cost ~0.6 MB per position).
        use_cache=False runs on a temporary empty cache (no reuse)."""
        if not use_cache:
            saved = self.cache
            self.reset()
        start = self.cache.length
        T = ids.shape[1]
        pos = torch.arange(start, start + T, device=self.device, dtype=torch.float32)
        freqs = torch.outer(pos, self.inv_freq)                     # [T, d/2]
        emb = torch.cat((freqs, freqs), dim=-1)                     # [T, d]
        cos, sin = emb.cos().to(self.dtype)[None, None], emb.sin().to(self.dtype)[None, None]

        h = self.m.embed_tokens(ids)
        for i, layer in enumerate(self.m.layers):
            x = self._rmsnorm(h, layer.input_layernorm.weight)
            a = layer.self_attn
            q = a.q_proj(x).view(1, T, self.n_heads, self.head_dim).transpose(1, 2)
            k = a.k_proj(x).view(1, T, self.n_kv, self.head_dim).transpose(1, 2)
            v = a.v_proj(x).view(1, T, self.n_kv, self.head_dim).transpose(1, 2)
            # rotary position embedding applied BEFORE caching: cached keys keep their positions
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
            k_all, v_all = self.cache.update(i, k, v, start)        # [1, n_kv, start+T, d]
            rep = self.n_heads // self.n_kv                          # grouped-query attention
            k_all = k_all.repeat_interleave(rep, dim=1)
            v_all = v_all.repeat_interleave(rep, dim=1)
            # causal: query at absolute pos start+t sees keys 0..start+t (bottom-right aligned)
            o = self.attn(q, k_all, v_all, causal=True)
            o = o.transpose(1, 2).reshape(1, T, self.n_heads * self.head_dim)
            h = h + a.o_proj(o)
            x = self._rmsnorm(h, layer.post_attention_layernorm.weight)
            mlp = layer.mlp
            h = h + mlp.down_proj(F.silu(mlp.gate_proj(x)) * mlp.up_proj(x))
        if logits_to_keep:
            h = h[:, -logits_to_keep:]
        h = self._rmsnorm(h, self.m.norm.weight)
        logits = self.lm_head(h).float()
        self.cache.length = start + T
        if not use_cache:
            self.cache = saved
        return logits

    def _rmsnorm(self, x, w):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return w * xf.to(x.dtype)


class HFCacheDecoder:
    """Same interface on top of the Hugging Face forward pass with an explicit DynamicCache.
    We pass position_ids and cache_position ourselves; the library handles masking inside
    the model given the cache length."""

    def __init__(self, hf_model):
        self.model = hf_model
        self.device = next(hf_model.parameters()).device
        self.reset()

    def reset(self):
        from transformers import DynamicCache
        self.cache = DynamicCache()
        self._len = 0

    @property
    def cache_len(self):
        return self._len

    def crop(self, n):
        if n < self._len:
            # negative argument = number of tokens to remove; works across transformers versions
            self.cache.crop(n - self._len)
            self._len = n

    @torch.no_grad()
    def forward(self, ids, use_cache: bool = True, logits_to_keep: int = 0):
        if not use_cache:
            out = self.model(input_ids=ids, use_cache=False, logits_to_keep=logits_to_keep)
            return out.logits.float()
        T = ids.shape[1]
        pos = torch.arange(self._len, self._len + T, device=self.device)
        out = self.model(input_ids=ids, past_key_values=self.cache, use_cache=True,
                         position_ids=pos[None], cache_position=pos, logits_to_keep=logits_to_keep)
        self._len += T
        return out.logits.float()


# ----------------------------------------------------------------------------- greedy loops
@dataclass
class GenStats:
    tokens: list = field(default_factory=list)
    prompt_len: int = 0
    prefill_s: float = 0.0
    decode_s: float = 0.0
    total_s: float = 0.0
    peak_mem_mb: float | None = None
    cache_mb: float | None = None

    @property
    def n_new(self):
        return len(self.tokens)

    @property
    def decode_tok_s(self):
        return (self.n_new - 1) / self.decode_s if self.decode_s > 0 and self.n_new > 1 else None

    @property
    def tok_s(self):
        return self.n_new / self.total_s if self.total_s > 0 else None

    def as_dict(self):
        return {"prompt_len": self.prompt_len, "new_tokens": self.n_new, "prefill_s": self.prefill_s,
                "decode_s": self.decode_s, "total_s": self.total_s, "tokens_per_s": self.tok_s,
                "decode_tokens_per_s": self.decode_tok_s, "peak_mem_mb": self.peak_mem_mb,
                "cache_mb": self.cache_mb}


@torch.no_grad()
def greedy_cached(dec, prompt_ids: torch.Tensor, max_new_tokens: int, eos_ids=()):
    """Prefill the prompt once, then feed exactly one new token per step, reusing the cache."""
    device = prompt_ids.device
    dec.reset(); reset_peak(device)
    st = GenStats(prompt_len=prompt_ids.shape[1])
    sync(device); t0 = time.perf_counter()
    logits = dec.forward(prompt_ids, logits_to_keep=1)   # prefill: whole prompt, cache grows to P
    nxt = int(logits[0, -1].argmax())
    sync(device); t1 = time.perf_counter()
    st.tokens.append(nxt)
    while len(st.tokens) < max_new_tokens and nxt not in eos_ids:
        logits = dec.forward(torch.tensor([[nxt]], device=device))   # decode: 1 token, cache +1
        nxt = int(logits[0, -1].argmax())
        st.tokens.append(nxt)
    sync(device); t2 = time.perf_counter()
    st.prefill_s, st.decode_s, st.total_s = t1 - t0, t2 - t1, t2 - t0
    st.peak_mem_mb = peak_mem_mb(device)
    if isinstance(dec, ExplicitQwen2Decoder):
        st.cache_mb = dec.cache.nbytes_used() / 2**20
    return st


@torch.no_grad()
def greedy_nocache(dec, prompt_ids: torch.Tensor, max_new_tokens: int, eos_ids=()):
    """Reference path: every step reruns the full sequence (prompt + generated) with no reuse.
    'prefill' here is the first step, so the comparison with the cached path is like for like."""
    device = prompt_ids.device
    dec.reset(); reset_peak(device)
    st = GenStats(prompt_len=prompt_ids.shape[1])
    seq = prompt_ids
    sync(device); t0 = time.perf_counter(); t1 = None
    while True:
        logits = dec.forward(seq, use_cache=False, logits_to_keep=1)
        nxt = int(logits[0, -1].argmax())
        if t1 is None:
            sync(device); t1 = time.perf_counter()
        st.tokens.append(nxt)
        if len(st.tokens) >= max_new_tokens or nxt in eos_ids:
            break
        seq = torch.cat([seq, torch.tensor([[nxt]], device=device)], dim=1)
    sync(device); t2 = time.perf_counter()
    st.prefill_s, st.decode_s, st.total_s = t1 - t0, t2 - t1, t2 - t0
    st.peak_mem_mb = peak_mem_mb(device)
    return st


def first_mismatch(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))
