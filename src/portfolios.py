"""Construcao das carteiras do experimento."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .estimators import (
    linear_shrinkage,
    nonlinear_shrinkage,
    sample_covariance,
)
from .optimization import minimum_variance_weights, robust_gmv_projected_gradient_descent


def equal_weight(returns: pd.DataFrame) -> pd.Series:
    """Carteira equally weighted."""

    n_assets = returns.shape[1]
    return pd.Series(1.0 / n_assets, index=returns.columns, name="EW")


def gmv(returns: pd.DataFrame) -> pd.Series:
    """GMV sem restricao de sinal."""

    weights = minimum_variance_weights(sample_covariance(returns), long_only=False)
    return pd.Series(weights, index=returns.columns, name="GMV")


def gmv_long_only(returns: pd.DataFrame) -> pd.Series:
    """GMV com pesos long-only."""

    weights = minimum_variance_weights(sample_covariance(returns), long_only=True)
    return pd.Series(weights, index=returns.columns, name="GMV Long Only")


def gmv_linear_shrinkage(returns: pd.DataFrame) -> pd.Series:
    """GMV com covariancia Ledoit-Wolf, permitindo short sales."""

    weights = minimum_variance_weights(linear_shrinkage(returns), long_only=False)
    return pd.Series(weights, index=returns.columns, name="GMV Linear Shrinkage")


def gmv_nonlinear_shrinkage(returns: pd.DataFrame) -> pd.Series:
    """GMV com nonlinear shrinkage espectral, permitindo short sales."""

    weights = minimum_variance_weights(nonlinear_shrinkage(returns), long_only=False)
    return pd.Series(weights, index=returns.columns, name="GMV Nonlinear Shrinkage")


def gmv_robust(returns: pd.DataFrame) -> pd.Series:
    """GMV robusta via PGD estimando diretamente Sigma w."""

    weights = robust_gmv_projected_gradient_descent(returns.to_numpy())
    return pd.Series(weights, index=returns.columns, name="GMV Robust")


PORTFOLIO_BUILDERS = {
    "EW": equal_weight,
    "GMV": gmv,
    "GMV Long Only": gmv_long_only,
    "GMV Linear Shrinkage": gmv_linear_shrinkage,
    "GMV Nonlinear Shrinkage": gmv_nonlinear_shrinkage,
    "GMV Robust": gmv_robust,
}


def build_all_portfolios(returns: pd.DataFrame) -> pd.DataFrame:
    """Calcula todos os vetores de pesos para uma janela de retornos."""

    weights = {name: builder(returns) for name, builder in PORTFOLIO_BUILDERS.items()}
    return pd.DataFrame(weights).T.replace([np.inf, -np.inf], np.nan).fillna(0.0)
