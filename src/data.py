"""Download, limpeza, retornos e janelas moveis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RollingWindow:
    """Uma janela de treino seguida por uma data de rebalanceamento."""

    train_returns: pd.DataFrame
    rebalance_date: pd.Timestamp


def download_prices(
    tickers: list[str],
    start: str,
    end: str | None = None,
    output_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Baixa precos ajustados via yfinance.

    Tickers brasileiros normalmente usam o sufixo `.SA`, como `PETR4.SA`.
    """

    import yfinance as yf

    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        yf.set_tz_cache_location(str(cache_dir))

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError("Download retornou vazio. Verifique tickers, datas e conexao.")

    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
    prices.columns = [str(column).removesuffix(".SA") for column in prices.columns]
    prices = clean_prices(prices)

    if prices.shape[1] < 1:
        raise RuntimeError("Poucos ativos validos apos limpeza dos precos.")

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(output_path)

    return prices


def load_prices(path: str | Path) -> pd.DataFrame:
    """Carrega precos de um CSV local."""

    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    return clean_prices(prices)


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Limpa precos, ordena datas e remove ativos sem historico utilizavel."""

    clean = prices.copy()
    clean.index = pd.to_datetime(clean.index)
    clean = clean.sort_index()
    clean = clean.apply(pd.to_numeric, errors="coerce")
    clean = clean.dropna(axis=1, how="all")
    clean = clean.ffill().dropna(how="all")
    return clean.loc[:, clean.notna().mean() >= 0.95].dropna()


def compute_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    """Calcula retornos logaritmicos ou simples."""

    if method == "log":
        returns = np.log(prices / prices.shift(1))
    elif method == "simple":
        returns = prices.pct_change()
    else:
        raise ValueError("method deve ser 'log' ou 'simple'.")

    return returns.replace([np.inf, -np.inf], np.nan).dropna()


def build_rolling_windows(
    returns: pd.DataFrame,
    window_size: int = 252,
    rebalance_frequency: str = "ME",
) -> list[RollingWindow]:
    """Constroi janelas moveis com rebalanceamento mensal."""

    rebalance_dates = returns.resample(rebalance_frequency).last().index
    windows: list[RollingWindow] = []

    for date in rebalance_dates:
        position = returns.index.searchsorted(date)
        start = position - window_size
        if start < 0 or position >= len(returns):
            continue
        windows.append(RollingWindow(returns.iloc[start:position], returns.index[position]))

    return windows


def make_synthetic_prices(
    n_assets: int = 8,
    n_days: int = 900,
    seed: int = 42,
) -> pd.DataFrame:
    """Gera precos sinteticos para testar o pipeline sem download."""

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n_days)
    factors = rng.normal(0.00025, 0.008, size=(n_days, 2))
    loadings = rng.normal(0.6, 0.2, size=(2, n_assets))
    idiosyncratic = rng.normal(0.0001, 0.012, size=(n_days, n_assets))
    returns = factors @ loadings + idiosyncratic
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    columns = [f"ATIVO{i + 1}" for i in range(n_assets)]
    return pd.DataFrame(prices, index=dates, columns=columns)
