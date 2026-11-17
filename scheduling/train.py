"""Training strategies: two-stage (MSE) and decision-focused (SPO+)."""

import numpy as np

from .data import Instance
from .loss import spo_plus_loss_and_grad


def predict_runtimes(instance: Instance, w: np.ndarray) -> np.ndarray:
    """Linear predictor ``p_hat = X w`` (misspecified: the truth is nonlinear)."""
    return instance.X @ w


def predicted_cost(instance: Instance, p_hat: np.ndarray) -> np.ndarray:
    """Cost matrix built from predicted runtimes."""
    return instance.cost_matrix(p_hat)


def true_cost(instance: Instance) -> np.ndarray:
    """Cost matrix built from the (observed) true runtimes."""
    return instance.cost_matrix(instance.p)


def fit_two_stage(train: list[Instance], ridge: float = 1e-3) -> np.ndarray:
    """Two-stage: fit runtimes from features with least squares, ignore the task."""
    X = np.vstack([ins.X for ins in train])
    p = np.concatenate([ins.p for ins in train])
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + ridge * np.eye(d), X.T @ p)


def fit_spo_plus(
    train: list[Instance],
    n_epochs: int = 40,
    lr: float = 0.1,
    ridge: float = 1e-3,
    clip_norm: float = 1.0,
):
    """Decision-focused training: full-batch GD on the SPO+ loss w.r.t. ``w``.

    Returns ``(w, losses)`` where ``losses`` is the mean SPO+ loss per epoch.
    The gradient is clipped so ``w`` stays bounded (SPO+ is only piecewise
    linear and not strongly convex).
    """
    d = train[0].X.shape[1]
    w = np.zeros(d)
    losses = []
    for _ in range(n_epochs):
        dw = np.zeros(d)
        total = 0.0
        for ins in train:
            chat = predicted_cost(ins, predict_runtimes(ins, w))
            c = true_cost(ins)
            loss, dchat = spo_plus_loss_and_grad(chat, c)
            total += loss
            # chain rule: chat[j, t] = q[t] * p_hat[j] + A[j, t], p_hat = X w
            dp = np.sum(dchat * ins.q[None, :], axis=1)  # (n,)
            dw += ins.X.T @ dp                            # (d,)
        dw /= len(train)
        dw += ridge * w
        nrm = np.linalg.norm(dw)
        if nrm > clip_norm:
            dw *= clip_norm / nrm
        w -= lr * dw
        losses.append(total / len(train))
    return w, losses
