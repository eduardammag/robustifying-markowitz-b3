"""Geracao de figuras e tabelas."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .metrics import cumulative_wealth


def _prepare_output(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def plot_cumulative_wealth(returns: pd.DataFrame, output_dir: str | Path) -> Path:
    """Salva grafico de riqueza acumulada."""

    output = _prepare_output(output_dir)
    wealth = cumulative_wealth(returns)
    axis = wealth.plot(figsize=(11, 6), linewidth=1.8)
    axis.set_title("Riqueza acumulada")
    axis.set_xlabel("")
    axis.set_ylabel("Riqueza")
    axis.grid(True, alpha=0.25)
    plt.tight_layout()
    path = output / "cumulative_wealth.png"
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def plot_turnover(turnover: pd.DataFrame, output_dir: str | Path) -> Path:
    """Salva grafico de turnover por rebalanceamento."""

    output = _prepare_output(output_dir)
    axis = turnover.plot(figsize=(11, 5), linewidth=1.5)
    axis.set_title("Turnover")
    axis.set_xlabel("")
    axis.set_ylabel("Turnover")
    axis.grid(True, alpha=0.25)
    plt.tight_layout()
    path = output / "turnover.png"
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def plot_weights(weights: dict[str, pd.DataFrame], output_dir: str | Path) -> list[Path]:
    """Salva pesos ao longo do tempo para cada carteira."""

    output = _prepare_output(output_dir)
    paths: list[Path] = []
    for name, history in weights.items():
        axis = history.plot(figsize=(11, 6), linewidth=1.2, legend=False)
        axis.set_title(f"Pesos - {name}")
        axis.set_xlabel("")
        axis.set_ylabel("Peso")
        axis.grid(True, alpha=0.25)
        plt.tight_layout()
        path = output / f"weights_{name.lower().replace(' ', '_')}.png"
        plt.savefig(path, dpi=160)
        plt.close()
        paths.append(path)
    return paths


def plot_weight_distribution(weights: dict[str, pd.DataFrame], output_dir: str | Path) -> Path:
    """Salva distribuicao dos pesos por carteira."""

    output = _prepare_output(output_dir)
    records = []
    for name, history in weights.items():
        stacked = history.stack().rename("weight").reset_index()
        stacked["portfolio"] = name
        records.append(stacked[["portfolio", "weight"]])
    data = pd.concat(records, ignore_index=True)

    plt.figure(figsize=(11, 6))
    sns.boxplot(data=data, x="portfolio", y="weight")
    plt.xticks(rotation=30, ha="right")
    plt.title("Distribuicao dos pesos")
    plt.tight_layout()
    path = output / "weight_distribution.png"
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def save_tables(tables: dict[str, pd.DataFrame], output_dir: str | Path) -> list[Path]:
    """Salva tabelas em CSV."""

    output = _prepare_output(output_dir)
    paths = []
    for name, table in tables.items():
        path = output / f"{name}.csv"
        table.to_csv(path, index_label="portfolio")
        paths.append(path)
    return paths
