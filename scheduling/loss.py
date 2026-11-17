"""SPO+ loss and its subgradient for the scheduling assignment problem.

Reference: Elmachtoub & Grigas, "Smart Predict-then-Optimize" (2022), and the
PyEPO implementation ``pyepo.func.SPOPlus``.  We implement the optimization
oracle with the Hungarian solver, so the project needs no Gurobi/PyEPO runtime
dependency; a ``pyepo`` optModel can be dropped in as the oracle instead.

For a convex feasible set S, let ``omega*(v) = argmin_{omega in S} <v, omega>``.
Given the true cost ``c`` and predicted cost ``chat``, the SPO+ loss is

    L(chat, c) = -<2 chat - c, omega*(2 chat - c)>
                 + 2 <chat, omega*(c)> - <c, omega*(c)>

a convex upper bound on the decision regret, with subgradient

    dL/dchat = 2 * (omega*(c) - omega*(2 chat - c)).
"""

from .optimizer import schedule_cost, solve_assignment


def spo_plus_loss_and_grad(chat, c):
    """Return ``(loss_value, subgradient dL/dchat)`` for one instance."""
    sol_true = solve_assignment(c)              # omega*(c)
    sol_spo = solve_assignment(2.0 * chat - c)  # omega*(2 chat - c)
    perturbed = 2.0 * chat - c
    loss = (
        -schedule_cost(perturbed, sol_spo)
        + 2.0 * schedule_cost(chat, sol_true)
        - schedule_cost(c, sol_true)
    )
    grad = 2.0 * (sol_true - sol_spo)
    return float(loss), grad
