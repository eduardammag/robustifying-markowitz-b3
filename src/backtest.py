"""Backtest com janelas moveis, rebalanceamento e custos."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .data import RollingWindow
from .portfolios import build_all_portfolios


@dataclass(frozen=True)
class BacktestResult:
    """Saidas principais do backtest."""

    gross_returns: pd.DataFrame
    net_returns: pd.DataFrame
    weights: dict[str, pd.DataFrame]
    turnover: pd.DataFrame
    target_turnover: pd.DataFrame


def normalize_weights(weights: pd.Series) -> pd.Series:
    """Normaliza pesos apos drift de mercado."""

    total = weights.sum()
    if abs(total) < 1e-12:
        return pd.Series(1.0 / len(weights), index=weights.index)
    return weights / total


def drift_weights(weights: pd.Series, period_returns: pd.DataFrame) -> pd.Series:
    """Calcula pesos imediatamente antes do proximo rebalanceamento."""

    if period_returns.empty:
        return weights
    gross_asset_returns = (1.0 + period_returns).prod(axis=0)
    return normalize_weights(weights * gross_asset_returns)


def compute_turnover(new_weights: pd.Series, pre_rebalance_weights: pd.Series | None) -> float:
    """Turnover do paper: soma absoluta dos trades no rebalanceamento."""

    if pre_rebalance_weights is None:
        pre_rebalance_weights = new_weights
    pre_rebalance_weights = pre_rebalance_weights.reindex(new_weights.index).fillna(0.0)
    return float((new_weights - pre_rebalance_weights).abs().sum())


def compute_target_turnover(new_weights: pd.Series, old_target_weights: pd.Series | None) -> float:
    """Target turnover: mudanca entre pesos-alvo, sem drift de precos."""

    if old_target_weights is None:
        old_target_weights = new_weights
    old_target_weights = old_target_weights.reindex(new_weights.index).fillna(0.0)
    return float((new_weights - old_target_weights).abs().sum())


def run_rolling_backtest(
    returns: pd.DataFrame,
    windows: list[RollingWindow],
    transaction_cost: float = 0.001,
) -> BacktestResult:
    """Executa backtest mensal com retorno bruto e liquido."""

    gross_records: list[pd.Series] = []
    net_records: list[pd.Series] = []
    turnover_records: list[pd.Series] = []
    target_turnover_records: list[pd.Series] = []
    weight_history: dict[str, list[pd.Series]] = {}
    previous_pre_rebalance_weights: dict[str, pd.Series] = {}
    previous_target_weights: dict[str, pd.Series] = {}

    for index, window in enumerate(windows):
        weights = build_all_portfolios(window.train_returns)
        start = returns.index.searchsorted(window.rebalance_date)
        end = returns.index.searchsorted(windows[index + 1].rebalance_date) if index + 1 < len(windows) else len(returns)
        holding_returns = returns.iloc[start:end]

        if holding_returns.empty:
            continue

        gross_period = {}
        net_period = {}
        turnover_period = {}
        target_turnover_period = {}

        for name, portfolio_weights in weights.iterrows():
            aligned_weights = portfolio_weights.reindex(returns.columns).fillna(0.0)
            gross = holding_returns @ aligned_weights
            turnover = compute_turnover(aligned_weights, previous_pre_rebalance_weights.get(name))
            target_turnover = compute_target_turnover(aligned_weights, previous_target_weights.get(name))
            cost = transaction_cost * turnover
            net = gross.copy()
            net.iloc[0] -= cost

            gross_period[name] = gross
            net_period[name] = net
            turnover_period[name] = turnover
            target_turnover_period[name] = target_turnover
            previous_pre_rebalance_weights[name] = drift_weights(aligned_weights, holding_returns)
            previous_target_weights[name] = aligned_weights
            weight_history.setdefault(name, []).append(aligned_weights.rename(window.rebalance_date))

        gross_records.append(pd.DataFrame(gross_period, index=holding_returns.index))
        net_records.append(pd.DataFrame(net_period, index=holding_returns.index))
        turnover_records.append(pd.Series(turnover_period, name=window.rebalance_date))
        target_turnover_records.append(pd.Series(target_turnover_period, name=window.rebalance_date))

    gross_returns = pd.concat(gross_records).sort_index()
    net_returns = pd.concat(net_records).sort_index()
    turnover = pd.DataFrame(turnover_records)
    target_turnover = pd.DataFrame(target_turnover_records)
    weights_out = {name: pd.DataFrame(history) for name, history in weight_history.items()}

    return BacktestResult(gross_returns, net_returns, weights_out, turnover, target_turnover)
