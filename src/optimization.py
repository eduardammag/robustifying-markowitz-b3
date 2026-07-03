"""Otimizacao das carteiras e rotinas matematicas auxiliares."""

from __future__ import annotations

import numpy as np

from .estimators import estimate_largest_eigenvalue, prepare_robust_blocks, robust_covariance_action_from_blocks


def project_to_simplex(weights: np.ndarray) -> np.ndarray:
    """Projeta pesos no simplex: pesos >= 0 e soma igual a 1."""

    values = np.asarray(weights, dtype=float)
    if values.ndim != 1:
        raise ValueError("weights deve ser um vetor.")

    sorted_values = np.sort(values)[::-1]
    cumulative = np.cumsum(sorted_values)
    rho = np.nonzero(sorted_values * np.arange(1, len(values) + 1) > cumulative - 1)[0]
    if len(rho) == 0:
        return np.full_like(values, 1.0 / len(values))
    rho = rho[-1]
    theta = (cumulative[rho] - 1) / (rho + 1)
    return np.maximum(values - theta, 0.0)


def project_budget_hyperplane(weights: np.ndarray) -> np.ndarray:
    """Projeta no conjunto afim {w: soma(w) = 1}, permitindo pesos negativos."""

    values = np.asarray(weights, dtype=float)
    n_assets = len(values)
    return values - (values.sum() - 1.0) / n_assets


def normalize_budget(weights: np.ndarray) -> np.ndarray:
    """Normaliza pesos para satisfazer soma igual a 1."""

    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    if abs(total) < 1e-12:
        return np.full_like(weights, 1.0 / len(weights))
    return weights / total


def minimum_variance_weights(
    covariance: np.ndarray,
    long_only: bool = False,
    ridge: float = 1e-8,
) -> np.ndarray:
    """Calcula pesos GMV com ou sem restricao long-only."""

    covariance = np.asarray(covariance, dtype=float)
    n_assets = covariance.shape[0]

    if long_only:
        return projected_gradient_descent(covariance)

    regularized = covariance + ridge * np.eye(n_assets)
    inv_ones = np.linalg.solve(regularized, np.ones(n_assets))
    return normalize_budget(inv_ones)


def portfolio_gradient(covariance: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Gradiente do risco 1/2 w' Sigma w."""

    return np.asarray(covariance) @ np.asarray(weights)


def robust_gradient_update(
    weights: np.ndarray,
    gradient: np.ndarray,
    learning_rate: float,
    long_only: bool = True,
) -> np.ndarray:
    """Atualiza pesos usando gradiente robusto e projecao de restricao."""

    updated = np.asarray(weights) - learning_rate * np.asarray(gradient)
    return project_to_simplex(updated) if long_only else project_budget_hyperplane(updated)


def projected_gradient_descent(
    covariance: np.ndarray,
    learning_rate: float = 0.05,
    max_iter: int = 2_000,
    tolerance: float = 1e-10,
) -> np.ndarray:
    """Resolve GMV long-only por projected gradient descent."""

    covariance = np.asarray(covariance, dtype=float)
    n_assets = covariance.shape[0]
    weights = np.full(n_assets, 1.0 / n_assets)

    largest_eigenvalue = max(np.linalg.eigvalsh(covariance).max(), 1e-8)
    step = min(learning_rate, 1.0 / largest_eigenvalue)

    for _ in range(max_iter):
        previous = weights.copy()
        weights = project_to_simplex(weights - step * portfolio_gradient(covariance, weights))
        if np.linalg.norm(weights - previous, ord=1) < tolerance:
            break

    return weights


def robust_gmv_projected_gradient_descent(
    returns: np.ndarray,
    n_blocks: int = 10,
    trim_quantile: float = 0.98,
    epsilon: float = 1.0 / 3.0,
    max_iter: int = 40,
    tolerance: float = 1e-8,
    learning_rate: float | None = None,
    step_scale: float = 0.2,
) -> np.ndarray:
    """Carteira GMV robusta do paper via PGD sobre estimativas de Sigma w.

    Comeca em EW, estima a acao robusta da covariancia no peso corrente e
    projeta apenas na restricao de orcamento, sem impor long-only.
    """

    values = np.asarray(returns, dtype=float)
    n_assets = values.shape[1]
    weights = np.full(n_assets, 1.0 / n_assets)
    robust_blocks = prepare_robust_blocks(values, n_blocks=n_blocks, trim_quantile=trim_quantile)

    if learning_rate is None:
        learning_rate = step_scale / estimate_largest_eigenvalue(values)

    for _ in range(max_iter):
        previous = weights.copy()
        action = robust_covariance_action_from_blocks(robust_blocks, weights, epsilon=epsilon)
        weights = project_budget_hyperplane(weights - learning_rate * action)
        if np.linalg.norm(weights - previous, ord=1) < tolerance:
            break

    return weights
