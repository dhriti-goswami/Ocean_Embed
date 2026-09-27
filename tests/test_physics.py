import numpy as np
import torch

from oceanembed.data.synthetic import GLORYS_DEPTHS as Z
from oceanembed.physics.losses import PhysicsLoss, density
from oceanembed.physics.products import all_products


def _loss(mode="cyclone"):
    D = len(Z)
    norm = dict(t_mean=np.zeros(D), t_std=np.ones(D))
    return PhysicsLoss(Z, norm, np.full(D, 1025.0), False, None, mode=mode)


def _T(values_at):
    return torch.tensor(np.interp(Z, *values_at), dtype=torch.float32)[None, :, None, None]


def test_soft_quantities_track_exact_products():
    T = _T(([0, 30, 60, 120, 900], [29, 28.9, 26, 20, 8]))
    m3 = torch.ones_like(T, dtype=torch.bool)
    soft = _loss().soft_quantities(T, m3)
    hard = all_products(T.numpy(), Z)
    assert abs(soft["D26"].item() - hard["D26_m"].item()) < 6
    assert abs(soft["TCHP"].item() - hard["TCHP_kJcm2"].item()) < 6


def test_stable_profile_has_zero_stability_penalty():
    T = _T(([0, 900], [29, 6]))
    m3 = torch.ones_like(T, dtype=torch.bool)
    assert _loss().stability_term(T, None, m3).item() == 0.0


def test_deep_inversion_is_penalized_but_shallow_is_not():
    loss = _loss()
    m3 = torch.ones((1, len(Z), 1, 1), dtype=torch.bool)
    shallow = _T(([0, 20, 40, 900], [28, 29, 27, 6]))          # warm layer at 20 m: allowed
    deep = _T(([0, 300, 400, 900], [29, 10, 12, 6]))           # warming at 300-400 m: not
    assert loss.stability_term(shallow, None, m3).item() == 0.0
    assert loss.stability_term(deep, None, m3).item() > 0.0


def test_cyclone_terms_are_zero_when_prediction_equals_target():
    loss = _loss()
    T = _T(([0, 30, 60, 120, 900], [29, 28.9, 26, 20, 8]))
    m3 = torch.ones_like(T, dtype=torch.bool)
    q = loss.quantity_terms(T, T.clone(), m3)
    assert all(v.item() < 1e-10 for v in q.values())


def test_density_increases_as_water_cools_and_salts():
    assert density(torch.tensor(10.0)) > density(torch.tensor(25.0))
    assert density(torch.tensor(20.0), torch.tensor(36.0)) > density(torch.tensor(20.0), torch.tensor(33.0))
