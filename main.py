"""Run the full predict-then-optimize scheduling case study.

No local build is needed: the GitHub Actions workflow
``.github/workflows/ci.yml`` installs dependencies, byte-compiles the sources,
runs the tests and executes this script.  See README.md for the scenario.
"""

import numpy as np

from scheduling.data import make_dataset
from scheduling.evaluate import mean_regret
from scheduling.train import fit_spo_plus, fit_two_stage


def main() -> None:
    n_jobs = 8
    train = make_dataset(200, n_jobs=n_jobs, seed=0)
    test = make_dataset(100, n_jobs=n_jobs, seed=1234)

    w_zero = np.zeros(train[0].X.shape[1])
    w_two = fit_two_stage(train)
    w_spo, losses = fit_spo_plus(train, n_epochs=40, lr=0.1)

    rows = [
        ("zero predictor (baseline)", mean_regret(test, w_zero)),
        ("two-stage (MSE)", mean_regret(test, w_two)),
        ("decision-focused (SPO+)", mean_regret(test, w_spo)),
    ]

    print("=== Job-scheduling predict-then-optimize case study ===")
    print(f"scenario       : schedule {n_jobs} jobs into {n_jobs} priced time slots")
    print(f"train / test   : {len(train)} / {len(test)} scheduling instances")
    print(f"SPO+ train loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print("-" * 58)
    print(f"{'method':<30}{'normalized regret':>20}")
    for name, r in rows:
        print(f"{name:<30}{r:>20.4f}")
    print("-" * 58)
    best = min(rows[1:], key=lambda x: x[1])
    print(f"lowest regret among trained models: {best[0]}")
    assert np.isfinite([r for _, r in rows]).all(), "non-finite regret"
    assert all(r >= -1e-9 for _, r in rows), "negative regret"


if __name__ == "__main__":
    main()
