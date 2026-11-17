"""Synthetic data for the job-scheduling predict-then-optimize case.

Scenario
--------
A cloud provider schedules a batch of ``n`` jobs into the ``n`` time slots of
one day.  Every slot ``t`` has a known electricity price ``q[t]``.  Every job
``j`` has an *uncertain* runtime ``p[j]`` that is only revealed after the job
runs, but is correlated with observable features ``X[j]`` (job category, input
size, memory demand, ...).  Placing job ``j`` in slot ``t`` also incurs a known
setup / data-transfer cost ``A[j, t]``.

The true cost of a schedule (a permutation ``sigma``) is

    C(sigma) = sum_j ( q[sigma(j)] * p[j] + A[j, sigma(j)] )

so the decision is a linear assignment problem with cost matrix

    cost[j, t] = q[t] * p[j] + A[j, t].

The runtime is deliberately a *nonlinear* function of the features while the
predictor is linear, i.e. the predictor is misspecified.  This is the regime
where decision-focused (SPO+) learning can beat plain two-stage MSE prediction.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Instance:
    """One scheduling instance: a batch of jobs to assign to time slots."""

    X: np.ndarray  # (n, d) job features
    p: np.ndarray  # (n,)   true runtimes (revealed after training, used as label)
    q: np.ndarray  # (n,)   slot prices (known)
    A: np.ndarray  # (n, n) known job-slot setup costs

    def cost_matrix(self, runtimes: np.ndarray) -> np.ndarray:
        """Build the (n, n) assignment cost from runtimes and slot prices."""
        return np.outer(runtimes, self.q) + self.A


def _true_runtime(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Nonlinear ground truth: linear term + curvature + an interaction."""
    d = X.shape[1]
    w = np.linspace(1.0, 2.0, d)
    linear = X @ w
    nonlinear = 0.9 * (X[:, 0] - 0.5) ** 2 + 1.1 * X[:, 1] * (1.0 - X[:, 2])
    noise = 0.05 * rng.standard_normal(X.shape[0])
    return np.clip(3.0 + linear + nonlinear + noise, 0.5, None)


def make_dataset(
    n_instances: int,
    n_jobs: int = 8,
    d_features: int = 5,
    seed: int = 0,
    price_spread: float = 1.0,
) -> list["Instance"]:
    """Generate a list of i.i.d. scheduling instances."""
    rng = np.random.default_rng(seed)
    instances = []
    for _ in range(n_instances):
        X = rng.uniform(0.0, 1.0, size=(n_jobs, d_features))
        p = _true_runtime(X, rng)
        # Slot prices: an increasing "peak-hour" profile with random jitter.
        base = np.linspace(0.5, 1.5, n_jobs)
        q = base + price_spread * rng.uniform(0.0, 0.5, size=n_jobs)
        q = np.clip(q, 0.1, None)
        # Known setup costs make the problem a genuinely 2-D assignment.
        A = rng.uniform(0.0, 2.0, size=(n_jobs, n_jobs))
        instances.append(Instance(X=X, p=p, q=q, A=A))
    return instances
