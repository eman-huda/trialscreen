import sys, pathlib, pytest, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from trialscreen.attention import reference_attention, tiled_attention, torch_sdpa, causal_mask

SHAPES = [(1, 1, 4, 4, 8, 2, 2), (2, 3, 17, 17, 16, 4, 5), (1, 2, 1, 33, 16, 8, 8), (1, 2, 5, 40, 16, 3, 7),
          (2, 2, 65, 65, 32, 64, 64), (1, 1, 100, 100, 8, 128, 128)]


@pytest.mark.parametrize("B,H,Tq,Tk,D,bq,bk", SHAPES)
@pytest.mark.parametrize("causal", [True, False])
def test_tiled_matches_float64_reference(B, H, Tq, Tk, D, bq, bk, causal):
    torch.manual_seed(0)
    q, k, v = (torch.randn(B, H, n, D, dtype=torch.float64) for n in (Tq, Tk, Tk))
    truth = reference_attention(q, k, v, causal)
    ref32 = reference_attention(q.float(), k.float(), v.float(), causal)
    til32 = tiled_attention(q.float(), k.float(), v.float(), causal, block_q=bq, block_k=bk)
    sdpa = torch_sdpa(q.float(), k.float(), v.float(), causal)
    assert (til32.double() - truth).abs().max() < 1e-5
    assert (ref32.double() - truth).abs().max() < 1e-5
    assert (sdpa.double() - truth).abs().max() < 1e-5


def test_bottom_right_causal_mask():
    m = causal_mask(2, 4)
    assert m.tolist() == [[True, True, True, False], [True, True, True, True]]


def test_softmax_stability_large_scores():
    q = torch.full((1, 1, 3, 4), 1e3); k = torch.full((1, 1, 3, 4), 1e3); v = torch.randn(1, 1, 3, 4)
    for fn in (reference_attention, lambda *a: tiled_attention(*a, block_q=2, block_k=2)):
        out = fn(q, k, v, True)
        assert torch.isfinite(out).all()


def test_keyword_only_tiles():
    q = torch.randn(1, 1, 4, 4)
    with pytest.raises(TypeError):
        tiled_attention(q, q, q, True, 0.5, 2, 2)
