"""Metricas de performance, risco e pesos."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def cumulative_wealth(returns: pd.DataFrame, initial_wealth: float = 1.0) -> pd.DataFrame:
    """Calcula riqueza acumulada."""

    return initial_wealth * (1.0 + returns).cumprod()


def annualized_volatility(returns: pd.DataFrame, periods_per_year: int = 252) -> pd.Series:
    """Volatilidade anualizada."""

    return returns.std() * np.sqrt(periods_per_year)


def annualized_return(returns: pd.DataFrame, periods_per_year: int = 252) -> pd.Series:
    """Retorno anualizado geometrico."""

    wealth = cumulative_wealth(returns)
    years = len(returns) / periods_per_year
    return wealth.iloc[-1] ** (1 / years) - 1


def sharpe_ratio(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> pd.Series:
    """Sharpe anualizado."""

    excess = returns - risk_free_rate / periods_per_year
    return excess.mean() / excess.std() * np.sqrt(periods_per_year)


def max_drawdown(wealth: pd.DataFrame) -> pd.Series:
    """Drawdown maximo."""

    drawdown = wealth / wealth.cummax() - 1.0
    return drawdown.min()


def calmar_ratio(returns: pd.DataFrame, periods_per_year: int = 252) -> pd.Series:
    """Calmar: retorno anualizado dividido pelo drawdown maximo absoluto."""

    wealth = cumulative_wealth(returns)
    return annualized_return(returns, periods_per_year) / max_drawdown(wealth).abs()


def mean_turnover(turnover: pd.DataFrame) -> pd.Series:
    """Turnover medio por carteira."""

    return turnover.mean()


def newey_west_covariance(moment_series: pd.DataFrame, lags: int | None = None) -> np.ndarray:
    """Estimador HAC/Newey-West da covariancia de longo prazo."""

    values = moment_series.dropna().to_numpy(dtype=float)
    n_obs = len(values)
    centered = values - values.mean(axis=0, keepdims=True)
    if lags is None:
        lags = int(np.floor(4 * (n_obs / 100) ** (2 / 9)))

    covariance = centered.T @ centered / n_obs
    for lag in range(1, min(lags, n_obs - 1) + 1):
        weight = 1.0 - lag / (lags + 1)
        gamma = centered[lag:].T @ centered[:-lag] / n_obs
        covariance += weight * (gamma + gamma.T)

    return covariance


def _sharpe_from_moments(moments: np.ndarray) -> float:
    mean_x, _, second_x, _ = moments
    variance_x = max(second_x - mean_x**2, 1e-16)
    return mean_x / np.sqrt(variance_x)


def _sharpe_difference_from_moments(moments: np.ndarray) -> float:
    mean_x, mean_y, second_x, second_y = moments
    variance_x = max(second_x - mean_x**2, 1e-16)
    variance_y = max(second_y - mean_y**2, 1e-16)
    return mean_x / np.sqrt(variance_x) - mean_y / np.sqrt(variance_y)


def _variance_difference_from_moments(moments: np.ndarray) -> float:
    mean_x, mean_y, second_x, second_y = moments
    return (second_x - mean_x**2) - (second_y - mean_y**2)


def _numerical_gradient(function, point: np.ndarray) -> np.ndarray:
    gradient = np.zeros_like(point, dtype=float)
    step = 1e-6 * np.maximum(np.abs(point), 1.0)
    for index in range(len(point)):
        upper = point.copy()
        lower = point.copy()
        upper[index] += step[index]
        lower[index] -= step[index]
        gradient[index] = (function(upper) - function(lower)) / (2 * step[index])
    return gradient


def hac_delta_pvalue(
    returns: pd.Series,
    benchmark: pd.Series,
    statistic: str,
) -> float:
    """p-value HAC para igualdade de Sharpe ou variancia contra benchmark."""

    aligned = pd.concat([returns, benchmark], axis=1).dropna()
    if len(aligned) < 10:
        return np.nan

    x = aligned.iloc[:, 0]
    y = aligned.iloc[:, 1]
    moments_ts = pd.DataFrame(
        {
            "x": x,
            "y": y,
            "x2": x**2,
            "y2": y**2,
        },
        index=aligned.index,
    )
    moments = moments_ts.mean().to_numpy()
    covariance = newey_west_covariance(moments_ts)

    if statistic == "sharpe":
        function = _sharpe_difference_from_moments
    elif statistic == "variance":
        function = _variance_difference_from_moments
    else:
        raise ValueError("statistic deve ser 'sharpe' ou 'variance'.")

    estimate = function(moments)
    gradient = _numerical_gradient(function, moments)
    standard_error = np.sqrt(max(float(gradient @ covariance @ gradient / len(moments_ts)), 1e-16))
    z_score = estimate / standard_error
    return float(2.0 * stats.norm.sf(abs(z_score)))


def turnover_pvalues(turnover: pd.DataFrame, benchmark_name: str = "GMV Robust") -> pd.Series:
    """p-values de t-test pareado para turnover contra a carteira robusta."""

    if benchmark_name not in turnover:
        return pd.Series(dtype=float)

    benchmark = turnover[benchmark_name]
    values = {}
    for column in turnover:
        if column == benchmark_name:
            values[column] = np.nan
            continue
        aligned = pd.concat([turnover[column], benchmark], axis=1).dropna()
        values[column] = stats.ttest_rel(aligned.iloc[:, 0], aligned.iloc[:, 1]).pvalue if len(aligned) > 1 else np.nan
    return pd.Series(values)


def return_pvalues(returns: pd.DataFrame, benchmark_name: str = "GMV Robust") -> pd.DataFrame:
    """p-values HAC de Sharpe e variancia contra a carteira robusta."""

    if benchmark_name not in returns:
        return pd.DataFrame(index=returns.columns)

    benchmark = returns[benchmark_name]
    rows = {}
    for column in returns:
        if column == benchmark_name:
            rows[column] = {
                "sharpe_pvalue_vs_robust": np.nan,
                "variance_pvalue_vs_robust": np.nan,
            }
            continue
        rows[column] = {
            "sharpe_pvalue_vs_robust": hac_delta_pvalue(returns[column], benchmark, "sharpe"),
            "variance_pvalue_vs_robust": hac_delta_pvalue(returns[column], benchmark, "variance"),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def weight_statistics(weights: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Resume concentracao e extremos dos pesos ao longo do tempo."""

    rows = []
    for name, history in weights.items():
        rows.append(
            {
                "portfolio": name,
                "min_weight": history.min(axis=1).mean(),
                "max_weight": history.max(axis=1).mean(),
                "weight_sd": history.std(axis=1).mean(),
                "mad_ew": (history.sub(1.0 / history.shape[1]).abs().mean(axis=1)).mean(),
                "max_minus_min": (history.max(axis=1) - history.min(axis=1)).mean(),
                "effective_n": (1 / (history.pow(2).sum(axis=1))).mean(),
            }
        )
    return pd.DataFrame(rows).set_index("portfolio")


def performance_table(
    returns: pd.DataFrame,
    turnover: pd.DataFrame | None = None,
    target_turnover: pd.DataFrame | None = None,
    weights: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Tabela final com metricas principais."""

    table = pd.DataFrame(
        {
            "ann_return": annualized_return(returns),
            "ann_volatility": annualized_volatility(returns),
            "sharpe": sharpe_ratio(returns),
            "max_drawdown": max_drawdown(cumulative_wealth(returns)),
            "calmar": calmar_ratio(returns),
        }
    )

    if turnover is not None:
        table["turnover"] = mean_turnover(turnover)
        table["turnover_pvalue_vs_robust"] = turnover_pvalues(turnover)
        if "Index" in table.index:
            table.loc["Index", "turnover"] = 0.0

    if target_turnover is not None:
        table["target_turnover"] = mean_turnover(target_turnover)
        table["target_turnover_pvalue_vs_robust"] = turnover_pvalues(target_turnover)
        if "Index" in table.index:
            table.loc["Index", "target_turnover"] = 0.0

    table = table.join(return_pvalues(returns), how="left")

    if weights is not None:
        table = table.join(weight_statistics(weights), how="left")

    return table
