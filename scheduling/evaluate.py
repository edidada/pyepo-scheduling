"""Decision-quality evaluation: normalized regret on held-out instances."""

import numpy as np

from .data import Instance
from .optimizer import schedule_cost, solve_assignment
from .train import predicted_cost, predict_runtimes, true_cost


def regret(instance: Instance, w: np.ndarray) -> float:
    """Excess true cost of the schedule chosen from predicted vs true runtimes."""
    c = true_cost(instance)
    chat = predicted_cost(instance, predict_runtimes(instance, w))
    chosen = solve_assignment(chat)
    best = solve_assignment(c)
    return schedule_cost(c, chosen) - schedule_cost(c, best)


def mean_regret(instances: list[Instance], w: np.ndarray) -> float:
    """Mean regret normalized by the optimal cost (comparable across instances)."""
    vals = []
    for ins in instances:
        c = true_cost(ins)
        opt = schedule_cost(c, solve_assignment(c))
        vals.append(regret(ins, w) / max(opt, 1e-9))
    return float(np.mean(vals))
