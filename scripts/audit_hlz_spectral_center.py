"""Audita propriedades do agregador spectral-center usado no Robust GMV."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.estimators import hlz_spectral_reweighting, weighted_center_and_scatter


def main() -> None:
    rng = np.random.default_rng(123)
    clean = rng.normal(size=(9, 5))
    outlier = np.array([[12.0, -9.0, 8.0, 0.0, 4.0]])
    vectors = np.vstack([clean, outlier])

    uniform_weights = np.full(len(vectors), 1.0 / len(vectors))
    _, uniform_scatter = weighted_center_and_scatter(vectors, uniform_weights)
    uniform_radius = float(np.linalg.eigvalsh(uniform_scatter).max())

    _, hlz_weights, hlz_radius = hlz_spectral_reweighting(vectors, epsilon=1.0 / 3.0)
    cap = 1.0 / (len(vectors) * (1.0 - 1.0 / 3.0))

    print(f"uniform_weighted_radius={uniform_radius:.6f}")
    print(f"hlz_weighted_radius={hlz_radius:.6f}")
    print(f"max_weight={hlz_weights.max():.6f}")
    print(f"weight_cap={cap:.6f}")

    if hlz_weights.max() > cap + 1e-10:
        raise SystemExit(1)
    if hlz_radius > uniform_radius * 1.05:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
