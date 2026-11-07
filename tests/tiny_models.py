"""Tiny randomly initialised Qwen2 models for fast CPU correctness tests."""
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM


def tiny_qwen2(seed=0, layers=2, hidden=64, heads=4, kv=2, vocab=257, dtype=torch.float32):
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=vocab, hidden_size=hidden, intermediate_size=hidden * 2,
                      num_hidden_layers=layers, num_attention_heads=heads, num_key_value_heads=kv,
                      max_position_embeddings=1024, rope_theta=10000.0, tie_word_embeddings=True)
    m = Qwen2ForCausalLM(cfg).to(dtype).eval()
    # default init (std 0.02) gives near-uniform logits and many argmax ties; a larger init
    # makes greedy decoding non-degenerate and well separated for exact-equality tests
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() == 2:
                p.normal_(0, 0.5)
    return m
