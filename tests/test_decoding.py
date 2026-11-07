import sys, pathlib, copy, pytest, torch
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tests"))
from tiny_models import tiny_qwen2
from trialscreen.decoding import ExplicitQwen2Decoder, HFCacheDecoder, greedy_cached, greedy_nocache
from trialscreen.speculative import speculative_greedy

TARGET = tiny_qwen2(seed=0)


def prompt(n, seed=1):
    return torch.randint(0, 257, (1, n), generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("backend", ["reference", "tiled", "torch_sdpa"])
def test_explicit_forward_matches_hf_logits(backend):
    ids = prompt(23)
    with torch.no_grad():
        ref = TARGET(ids).logits.float()
    out = ExplicitQwen2Decoder(TARGET, backend, max_len=128).forward(ids)
    assert (out - ref).abs().max() < 1e-3


def test_incremental_logits_match_full_recompute():
    """Feeding tokens one by one through the cache gives the same logits as one full pass."""
    ids = prompt(20)
    dec = ExplicitQwen2Decoder(TARGET, "reference", max_len=64)
    full = dec.forward(ids, use_cache=False)
    dec.reset()
    dec.forward(ids[:, :12])
    steps = [dec.forward(ids[:, i:i + 1]) for i in range(12, 20)]
    inc = torch.cat(steps, dim=1)
    assert dec.cache_len == 20
    assert (inc - full[:, 12:]).abs().max() < 1e-3


@pytest.mark.parametrize("P,N", [(5, 10), (31, 25), (80, 40)])
def test_greedy_equivalence_cached_vs_uncached(P, N):
    ids = prompt(P, seed=P)
    ref = greedy_nocache(HFCacheDecoder(TARGET), ids, N).tokens
    assert greedy_nocache(ExplicitQwen2Decoder(TARGET, "reference", 256), ids, N).tokens == ref
    assert greedy_cached(ExplicitQwen2Decoder(TARGET, "reference", 256), ids, N).tokens == ref
    assert greedy_cached(ExplicitQwen2Decoder(TARGET, "tiled", 256), ids, N).tokens == ref
    assert greedy_cached(HFCacheDecoder(TARGET), ids, N).tokens == ref


def test_cache_growth_and_crop():
    dec = ExplicitQwen2Decoder(TARGET, "reference", max_len=64)
    dec.forward(prompt(10)); assert dec.cache_len == 10
    dec.forward(prompt(1)); assert dec.cache_len == 11
    dec.crop(7); assert dec.cache_len == 7
    with pytest.raises(RuntimeError):
        dec.forward(prompt(60))


def _noisy_copy(model, scale):
    m = copy.deepcopy(model)
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn(p.shape, generator=g) * scale)
    return m


@pytest.mark.parametrize("draft_kind", ["identical", "noisy", "unrelated"])
@pytest.mark.parametrize("k", [1, 3, 5])
@pytest.mark.parametrize("dec_cls", ["explicit", "hf"])
def test_speculative_equals_greedy(draft_kind, k, dec_cls):
    draft_model = {"identical": TARGET, "noisy": _noisy_copy(TARGET, 0.02),
                   "unrelated": tiny_qwen2(seed=7, layers=1, hidden=32, heads=2, kv=1)}[draft_kind]
    mk = (lambda m: ExplicitQwen2Decoder(m, "reference", 256)) if dec_cls == "explicit" else HFCacheDecoder
    ids = prompt(17, seed=5)
    ref = greedy_cached(mk(TARGET), ids, 40).tokens
    st = speculative_greedy(mk(TARGET), mk(draft_model), ids, 40, k=k)
    assert st.tokens == ref
    if draft_kind == "identical":
        assert st.acceptance_rate == 1.0
    if draft_kind == "noisy":
        assert 0.0 < st.acceptance_rate < 1.0   # exercises partial acceptance + rejection


def test_speculative_respects_eos():
    ids = prompt(9, seed=11)
    ref = greedy_cached(ExplicitQwen2Decoder(TARGET, "reference", 256), ids, 30).tokens
    eos = ref[6]
    want = greedy_cached(ExplicitQwen2Decoder(TARGET, "reference", 256), ids, 30, eos_ids=(eos,)).tokens
    got = speculative_greedy(ExplicitQwen2Decoder(TARGET, "reference", 256), ExplicitQwen2Decoder(TARGET, "reference", 256),
                             ids, 30, k=4, eos_ids=(eos,)).tokens
    assert got == want and got[-1] == eos
