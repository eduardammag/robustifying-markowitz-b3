"""Estimadores usados nas carteiras.

O ponto central do paper e evitar estimar uma covariancia robusta completa para
calcular a carteira robusta. Em vez disso, o algoritmo estima diretamente a
acao do operador de covariancia em um vetor corrente, isto e, Sigma w.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def _as_array(returns: pd.DataFrame | np.ndarray) -> np.ndarray:
    values = returns.to_numpy() if isinstance(returns, pd.DataFrame) else np.asarray(returns)
    return np.nan_to_num(values, nan=0.0)


def sample_covariance(returns: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Matriz de covariancia amostral."""

    values = _as_array(returns)
    return np.cov(values, rowvar=False)


def linear_shrinkage(returns: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Covariancia Ledoit-Wolf com shrinkage linear."""

    return LedoitWolf().fit(_as_array(returns)).covariance_


def nonlinear_shrinkage(
    returns: pd.DataFrame | np.ndarray,
    bandwidth: float | None = None,
    min_eigenvalue: float = 1e-10,
) -> np.ndarray:
    """Covariancia por nonlinear shrinkage espectral.

    Implementa uma versao direta do estimador de Ledoit-Wolf: preserva os
    autovetores da covariancia amostral e aplica uma transformacao nao linear
    aos autovalores usando uma estimativa suavizada do Stieltjes transform da
    distribuicao espectral amostral.
    """

    values = _as_array(returns)
    centered = values - values.mean(axis=0, keepdims=True)
    n_obs, n_assets = centered.shape
    sample = centered.T @ centered / max(n_obs - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(sample)
    eigenvalues = np.maximum(eigenvalues, min_eigenvalue)

    concentration = n_assets / max(n_obs, 1)
    bandwidth = n_obs ** (-1.0 / 3.0) if bandwidth is None else bandwidth
    spectral_scale = max(float(np.median(eigenvalues)), float(eigenvalues.mean()), min_eigenvalue)
    imaginary_part = bandwidth * np.maximum(eigenvalues, spectral_scale)
    z_grid = eigenvalues + 1j * imaginary_part

    stieltjes = np.array([np.mean(1.0 / (eigenvalues - z)) for z in z_grid])
    denominator = (
        (1.0 - concentration - concentration * eigenvalues * stieltjes.real) ** 2
        + (concentration * eigenvalues * stieltjes.imag) ** 2
    )
    shrunk = eigenvalues / np.maximum(denominator, min_eigenvalue)

    sample_trace = eigenvalues.sum()
    shrunk_trace = shrunk.sum()
    if shrunk_trace > min_eigenvalue:
        shrunk *= sample_trace / shrunk_trace
    shrunk = np.maximum(shrunk, min_eigenvalue)

    covariance = (eigenvectors * shrunk) @ eigenvectors.T
    return (covariance + covariance.T) / 2.0


def median_of_means_covariance(
    returns: pd.DataFrame | np.ndarray,
    n_blocks: int = 10,
) -> np.ndarray:
    """Estimador auxiliar de covariancia por median-of-means.

    Mantido para diagnosticos. A carteira robusta do paper nao deve usar esta
    matriz como plug-in; ela usa `robust_covariance_action`.
    """

    values = _as_array(returns)
    blocks = np.array_split(values, n_blocks)
    covariances = np.array([np.cov(block, rowvar=False) for block in blocks if len(block) > 1])
    return np.median(covariances, axis=0)


def pairwise_center(returns: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Centraliza observacoes por diferencas pareadas, preservando a covariancia."""

    values = _as_array(returns)
    even_length = 2 * (len(values) // 2)
    paired = values[:even_length].reshape(-1, 2, values.shape[1])
    return (paired[:, 0, :] - paired[:, 1, :]) / np.sqrt(2.0)


def coordinatewise_median(vectors: np.ndarray) -> np.ndarray:
    """Agregador robusto simples para medias por bloco."""

    return np.median(np.asarray(vectors, dtype=float), axis=0)


def capped_simplex_weights(scores: np.ndarray, epsilon: float = 1.0 / 3.0) -> np.ndarray:
    """Resolve min sum_j u_j scores_j no simplex capado Delta_{l,epsilon}."""

    scores = np.asarray(scores, dtype=float)
    n_scores = len(scores)
    cap = 1.0 / (n_scores * (1.0 - epsilon))
    weights = np.zeros(n_scores)
    remaining = 1.0

    for index in np.argsort(scores):
        weight = min(cap, remaining)
        weights[index] = weight
        remaining -= weight
        if remaining <= 1e-12:
            break

    return weights / weights.sum()


def spectral_center_radius(vectors: np.ndarray, center: np.ndarray, epsilon: float = 1.0 / 3.0) -> float:
    """Raio espectral local usado para auditar candidatos a centro."""

    centered = vectors - center
    scatter = centered.T @ centered / len(centered)
    eigenvalues, eigenvectors = np.linalg.eigh((scatter + scatter.T) / 2.0)
    direction = eigenvectors[:, int(np.argmax(eigenvalues))]
    weights = capped_simplex_weights((centered @ direction) ** 2, epsilon=epsilon)
    weighted_scatter = np.einsum("i,ij,ik->jk", weights, centered, centered)
    return float(np.linalg.eigvalsh((weighted_scatter + weighted_scatter.T) / 2.0).max())


def hlz_spectral_center(
    vectors: np.ndarray,
    epsilon: float = 1.0 / 3.0,
    max_iter: int = 100,
    tolerance: float = 1e-9,
) -> np.ndarray:
    """Agregador spectral-center para os vetores de bloco.

    Esta rotina implementa o mecanismo computacional usado pelo HLZ/spectral
    reweighting: alterna entre encontrar a direcao de maior dispersao em torno
    do centro e reponderar os blocos no simplex capado, descartando influencia
    excessiva dos blocos mais distantes naquela direcao.
    """

    vectors = np.asarray(vectors, dtype=float)
    if len(vectors) == 1:
        return vectors[0]

    center = coordinatewise_median(vectors)
    weights = np.full(len(vectors), 1.0 / len(vectors))
    best_center = center.copy()
    best_radius = spectral_center_radius(vectors, best_center, epsilon=epsilon)

    for _ in range(max_iter):
        previous = center.copy()
        centered = vectors - center
        scatter = np.einsum("i,ij,ik->jk", weights, centered, centered)
        eigenvalues, eigenvectors = np.linalg.eigh((scatter + scatter.T) / 2.0)
        direction = eigenvectors[:, int(np.argmax(eigenvalues))]
        scores = (centered @ direction) ** 2
        weights = capped_simplex_weights(scores, epsilon=epsilon)
        center = weights @ vectors
        radius = spectral_center_radius(vectors, center, epsilon=epsilon)
        if radius < best_radius:
            best_radius = radius
            best_center = center.copy()

        if np.linalg.norm(center - previous) < tolerance:
            break

    return best_center


def robust_covariance_action(
    returns: pd.DataFrame | np.ndarray,
    weights: np.ndarray,
    n_blocks: int = 10,
    trim_quantile: float = 0.98,
    epsilon: float = 1.0 / 3.0,
) -> np.ndarray:
    """Estima Sigma w por median-of-means sobre acoes de covariancia.

    Implementa a versao empirica coerente com o paper: centraliza por
    diferencas pareadas, divide em blocos, calcula a media de X(X'w) em cada
    bloco apos truncagem por norma, e agrega as acoes por spectral center.
    """

    centered = pairwise_center(returns)
    weights = np.asarray(weights, dtype=float)

    if len(centered) < n_blocks:
        n_blocks = max(1, len(centered))

    norms = np.linalg.norm(centered, axis=1)
    threshold = np.quantile(norms, trim_quantile)
    blocks = np.array_split(centered, n_blocks)
    actions = []

    for block in blocks:
        if len(block) == 0:
            continue
        block_norms = np.linalg.norm(block, axis=1)
        trimmed = block[block_norms <= threshold]
        if len(trimmed) == 0:
            trimmed = block
        actions.append((trimmed.T @ (trimmed @ weights)) / len(trimmed))

    return hlz_spectral_center(np.asarray(actions), epsilon=epsilon)


def prepare_robust_blocks(
    returns: pd.DataFrame | np.ndarray,
    n_blocks: int = 10,
    trim_quantile: float = 0.98,
) -> list[np.ndarray]:
    """Prepara blocos pareados e truncados para repetidas estimativas de Sigma w."""

    centered = pairwise_center(returns)
    if len(centered) < n_blocks:
        n_blocks = max(1, len(centered))

    norms = np.linalg.norm(centered, axis=1)
    threshold = np.quantile(norms, trim_quantile)
    prepared = []
    for block in np.array_split(centered, n_blocks):
        if len(block) == 0:
            continue
        block_norms = np.linalg.norm(block, axis=1)
        trimmed = block[block_norms <= threshold]
        prepared.append(trimmed if len(trimmed) else block)
    return prepared


def robust_covariance_action_from_blocks(
    blocks: list[np.ndarray],
    weights: np.ndarray,
    epsilon: float = 1.0 / 3.0,
) -> np.ndarray:
    """Estima Sigma w usando blocos robustos pre-computados."""

    actions = [(block.T @ (block @ weights)) / len(block) for block in blocks]
    return hlz_spectral_center(np.asarray(actions), epsilon=epsilon, max_iter=50)


def empirical_covariance_action(
    returns: pd.DataFrame | np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Calcula Sigma_hat w sem materializar a matriz de covariancia."""

    values = _as_array(returns)
    centered = values - values.mean(axis=0, keepdims=True)
    return centered.T @ (centered @ np.asarray(weights, dtype=float)) / max(len(centered) - 1, 1)


def estimate_largest_eigenvalue(
    returns: pd.DataFrame | np.ndarray,
    n_iter: int = 100,
    tolerance: float = 1e-10,
) -> float:
    """Estima a maior autovalor de Sigma_hat por power iteration."""

    values = _as_array(returns)
    rng = np.random.default_rng(123)
    vector = rng.normal(size=values.shape[1])
    vector = vector / np.linalg.norm(vector)

    eigenvalue = 0.0
    for _ in range(n_iter):
        updated = empirical_covariance_action(values, vector)
        norm = np.linalg.norm(updated)
        if norm < 1e-14:
            return 1e-8
        vector = updated / norm
        next_eigenvalue = float(vector @ empirical_covariance_action(values, vector))
        if abs(next_eigenvalue - eigenvalue) < tolerance:
            break
        eigenvalue = next_eigenvalue

    return max(eigenvalue, 1e-8)


ESTIMATORS = {
    "sample": sample_covariance,
    "linear_shrinkage": linear_shrinkage,
    "nonlinear_shrinkage": nonlinear_shrinkage,
    "median_of_means_covariance": median_of_means_covariance,
}
