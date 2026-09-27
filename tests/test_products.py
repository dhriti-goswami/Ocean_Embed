import numpy as np

from oceanembed.data.synthetic import GLORYS_DEPTHS as Z
from oceanembed.physics.products import all_products, isotherm_depth


def _profile():
    # mixed to 30 m at ~29 C, 26 C at 60 m, 20 C at 120 m, 8 C at 900 m
    return np.interp(Z, [0, 30, 60, 120, 900], [29, 28.9, 26, 20, 8])[None, :, None, None]


def test_d26_and_tchp_on_idealized_profile():
    p = all_products(_profile(), Z)
    assert abs(p["D26_m"].item() - 60) < 1.5
    assert 50 < p["TCHP_kJcm2"].item() < 60                 # rho*cp*~132 K m / 1e7 ~ 54
    assert 25 < p["MLD_m"].item() < 45


def test_isotherm_zero_when_surface_colder():
    T = np.full((1, len(Z), 1, 1), 24.0)
    assert isotherm_depth(T, Z, 26.0).item() == 0.0


def test_isotherm_nan_when_never_crossed():
    T = np.full((1, len(Z), 1, 1), 28.0)
    assert np.isnan(isotherm_depth(T, Z, 26.0).item())


def test_land_stays_nan():
    T = np.full((1, len(Z), 1, 1), np.nan)
    assert all(np.isnan(v).all() for v in all_products(T, Z).values())
