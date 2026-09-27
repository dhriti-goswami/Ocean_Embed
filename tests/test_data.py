import numpy as np

from oceanembed.data import DataConfig, OceanData


def test_history_channels_are_past_days_only(cube):
    d0 = OceanData(DataConfig(path=cube, history_days=0))
    d2 = OceanData(DataConfig(path=cube, history_days=2))
    n_sat = len(DataConfig(path=cube).inputs)
    assert d2.n_inputs == d0.n_inputs + 2 * n_sat           # 8 fields x 2 extra days
    # channel block "t-1" on day k must equal "today" block on day k-1 (no future leakage)
    t1 = slice(n_sat, 2 * n_sat)
    today = slice(2 * n_sat, 3 * n_sat)
    assert np.allclose(d2.X[10, t1], d2.X[9, today])


def test_splits_are_contiguous_disjoint_and_ordered(cube):
    d = OceanData(DataConfig(path=cube))
    tr, va, te = d.idx["train"], d.idx["val"], d.idx["test"]
    assert len(set(tr) & set(va)) == len(set(va) & set(te)) == len(set(tr) & set(te)) == 0
    assert tr.max() < va.min() and va.max() < te.min()
    assert len(tr) + len(va) + len(te) == len(d.times)


def test_ascat_swath_channels_are_not_inputs(cube):
    d = OceanData(DataConfig(path=cube))
    assert not any("ascat" in c for c in d.channel_names)


def test_sst_converted_from_kelvin(cube):
    d = OceanData(DataConfig(path=cube, history_days=0))
    k = d.channel_names.index("analysed_sst")
    assert 20 < d.x_mean[k] < 35


def test_seabed_levels_masked_per_depth(cube):
    d = OceanData(DataConfig(path=cube))
    assert d.mask3d[0].sum() >= d.mask3d[-1].sum()          # fewer valid cells at depth
    assert not np.isnan(d.T[:, d.mask3d]).any()


def test_padding_divisible_for_unet_and_vit(cube):
    d = OceanData(DataConfig(path=cube))
    assert d.Hp % 8 == 0 and d.Wp % 8 == 0
    x, y, m3, mld, k = d.dataset("train")[0]
    assert x.shape[-2:] == y.shape[-2:] == m3.shape[-2:] == (d.Hp, d.Wp)
