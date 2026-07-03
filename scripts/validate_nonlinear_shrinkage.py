"""Valida o nonlinear shrinkage contra uma matriz de referencia.

Uso:
    python scripts/validate_nonlinear_shrinkage.py
    python scripts/validate_nonlinear_shrinkage.py --reference-csv reference_lw2017.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.estimators import nonlinear_shrinkage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--returns-csv", default=ROOT / "data" / "processed" / "returns.csv")
    parser.add_argument("--reference-csv", help="Matriz oficial Ledoit-Wolf 2017 para comparar.")
    parser.add_argument("--window-size", type=int, default=252)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    returns = pd.read_csv(args.returns_csv, index_col=0).iloc[: args.window_size]
    estimate = nonlinear_shrinkage(returns)
    eigenvalues = np.linalg.eigvalsh(estimate)

    print(f"shape={estimate.shape}")
    print(f"symmetry_error={np.max(np.abs(estimate - estimate.T)):.6e}")
    print(f"min_eigenvalue={eigenvalues.min():.6e}")
    print(f"max_eigenvalue={eigenvalues.max():.6e}")

    if args.reference_csv:
        reference = pd.read_csv(args.reference_csv, index_col=0).to_numpy(dtype=float)
        difference = np.linalg.norm(estimate - reference, ord="fro")
        denominator = max(np.linalg.norm(reference, ord="fro"), 1e-16)
        relative_error = difference / denominator
        print(f"relative_frobenius_error={relative_error:.6e}")
        if relative_error > args.tolerance:
            raise SystemExit(1)


if __name__ == "__main__":
    main()

