"""Audita propriedades do agregador spectral-center usado no Robust GMV."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.estimators import capped_simplex_weights, hlz_spectral_center


def spectral_radius(vectors: np.ndarray, center: np.ndarray, epsilon: float = 1.0 / 3.0) -> float:
    centered = vectors - center
    scatter_uniform = centered.T @ centered / len(centered)
    eigenvalues, eigenvectors = np.linalg.eigh((scatter_uniform + scatter_uniform.T) / 2.0)
    direction = eigenvectors[:, int(np.argmax(eigenvalues))]
    weights = capped_simplex_weights((centered @ direction) ** 2, epsilon=epsilon)
    scatter = np.einsum("i,ij,ik->jk", weights, centered, centered)
    return float(np.linalg.eigvalsh((scatter + scatter.T) / 2.0).max())


def main() -> None:
    rng = np.random.default_rng(123)
    clean = rng.normal(size=(9, 5))
    outlier = np.array([[12.0, -9.0, 8.0, 0.0, 4.0]])
    vectors = np.vstack([clean, outlier])

    mean_center = vectors.mean(axis=0)
    median_center = np.median(vectors, axis=0)
    spectral_center = hlz_spectral_center(vectors, epsilon=1.0 / 3.0)

    mean_radius = spectral_radius(vectors, mean_center)
    median_radius = spectral_radius(vectors, median_center)
    spectral_radius_value = spectral_radius(vectors, spectral_center)

    print(f"mean_radius={mean_radius:.6f}")
    print(f"median_radius={median_radius:.6f}")
    print(f"spectral_center_radius={spectral_radius_value:.6f}")

    if spectral_radius_value > median_radius * 1.05:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

