"""Optimization oracle: min-cost job-to-slot assignment (Hungarian algorithm).

The feasible set is the Birkhoff polytope (doubly stochastic matrices); a linear
objective attains its optimum at a permutation matrix, so the Hungarian solver
returns the optimal schedule.  This is the ``omega*(c)`` oracle used by SPO+.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def solve_assignment(cost: np.ndarray) -> np.ndarray:
    """Return the 0/1 assignment matrix ``omega`` minimizing ``<cost, omega>``.

    ``cost`` has shape (n_jobs, n_slots); the returned matrix has the same shape
    with exactly one 1 per row and column when the instance is square.
    """
    row_ind, col_ind = linear_sum_assignment(cost)
    omega = np.zeros_like(cost, dtype=float)
    omega[row_ind, col_ind] = 1.0
    return omega


def schedule_cost(cost: np.ndarray, omega: np.ndarray) -> float:
    """Total cost incurred by an assignment matrix under a (true) cost."""
    return float(np.sum(cost * omega))
