"""Unit tests for the scheduling predict-then-optimize case study."""

import numpy as np

from scheduling.data import make_dataset
from scheduling.evaluate import mean_regret, regret
from scheduling.loss import spo_plus_loss_and_grad
from scheduling.optimizer import schedule_cost, solve_assignment
from scheduling.train import fit_spo_plus, fit_two_stage


def test_assignment_is_a_permutation():
    cost = np.array([[3.0, 1.0], [2.0, 5.0]])
    omega = solve_assignment(cost)
    assert omega.sum() == 2.0
    assert np.allclose(omega.sum(axis=0), 1.0)
    assert np.allclose(omega.sum(axis=1), 1.0)
    assert schedule_cost(cost, omega) == 3.0  # picks (0,1)=1 and (1,0)=2


def test_spo_plus_is_zero_at_perfect_prediction():
    c = np.array([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0], [2.0, 3.0, 1.0]])
    loss, grad = spo_plus_loss_and_grad(c.copy(), c)
    assert abs(loss) < 1e-9
    assert np.allclose(grad, 0.0)


def test_spo_plus_loss_is_nonnegative():
    c = np.array([[1.0, 2.0], [2.0, 1.0]])
    chat = np.array([[2.0, 1.0], [1.0, 2.0]])
    loss, grad = spo_plus_loss_and_grad(chat, c)
    assert loss >= -1e-9
    assert grad.shape == c.shape


def test_regret_is_nonnegative():
    ds = make_dataset(5, n_jobs=6, seed=7)
    w = np.zeros(ds[0].X.shape[1])
    for ins in ds:
        assert regret(ins, w) >= -1e-9


def test_training_is_finite_and_well_behaved():
    train = make_dataset(60, n_jobs=6, seed=1)
    test = make_dataset(20, n_jobs=6, seed=2)
    w_two = fit_two_stage(train)
    w_spo, losses = fit_spo_plus(train, n_epochs=20, lr=0.05)
    assert np.isfinite(w_two).all()
    assert np.isfinite(w_spo).all()
    assert np.isfinite(losses).all()
    r = mean_regret(test, w_spo)
    assert np.isfinite(r) and r >= -1e-9
