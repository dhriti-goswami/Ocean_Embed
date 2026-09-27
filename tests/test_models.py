import pytest
import torch

from oceanembed.models import build_model


@pytest.mark.parametrize("variant", ["unet", "oceanembed"])
@pytest.mark.parametrize("uncertainty", [False, True])
def test_output_shapes(variant, uncertainty):
    m = build_model(variant, in_ch=28, n_depth=35, grid_hw=(72, 88), out_vars=1, uncertainty=uncertainty)
    y = m(torch.randn(2, 28, 72, 88))
    assert y.shape == (2, 35 * (2 if uncertainty else 1), 72, 88)


def test_embedding_is_eighth_resolution():
    m = build_model("oceanembed", 28, 35, (72, 88), 1)
    z, _ = m.encode(torch.randn(1, 28, 72, 88))
    assert z.shape == (1, 128, 9, 11)


def test_salinity_output_channels():
    m = build_model("oceanembed", 28, 35, (72, 88), out_vars=2, uncertainty=True)
    assert m(torch.randn(1, 28, 72, 88)).shape[1] == 35 * 2 + 35
