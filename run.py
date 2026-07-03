"""Executa todo o pipeline do projeto."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.backtest import run_rolling_backtest
from src.data import (
    build_rolling_windows,
    compute_returns,
    download_prices,
    load_prices,
    make_synthetic_prices,
)
from src.metrics import performance_table
from src.plots import (
    plot_cumulative_wealth,
    plot_turnover,
    plot_weight_distribution,
    plot_weights,
    save_tables,
)
from src.universe import UNIVERSES, read_tickers_file, yahoo_tickers


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
FROZEN_TICKERS = RAW_DIR / "tickers_ibovespa_frozen_20260702.txt"
INDEX_PRICES = RAW_DIR / "ibovespa_index.csv"
PROCESSED_RETURNS = ROOT / "data" / "processed" / "returns.csv"
PROCESSED_INDEX_RETURNS = ROOT / "data" / "processed" / "index_returns.csv"
RESULTS_DIR = ROOT / "data" / "results"
FIGURES_DIR = ROOT / "outputs" / "figures"
TABLES_DIR = ROOT / "outputs" / "tables"


def parse_args() -> argparse.Namespace:
    """Define os parametros de execucao do pipeline."""

    parser = argparse.ArgumentParser(description="Pipeline Markowitz robusto para ativos B3.")
    parser.add_argument("--universe", default="ibovespa", choices=sorted(UNIVERSES))
    parser.add_argument("--tickers-file", help="Arquivo com um ticker B3 por linha.")
    parser.add_argument("--index-ticker", default="^BVSP", help="Ticker do benchmark de indice.")
    parser.add_argument("--start", default="2015-01-01", help="Data inicial dos precos.")
    parser.add_argument("--end", default=None, help="Data final dos precos.")
    parser.add_argument("--return-method", default="simple", choices=["simple", "log"])
    parser.add_argument("--window-size", type=int, help="Janela unica, mantida por compatibilidade.")
    parser.add_argument("--window-sizes", default="252,500", help="Janelas separadas por virgula.")
    parser.add_argument("--transaction-cost", type=float, default=0.005)
    parser.add_argument(
        "--offline-synthetic",
        action="store_true",
        help="Usa dados sinteticos apenas para testar o pipeline sem internet.",
    )
    return parser.parse_args()


def parse_window_sizes(args: argparse.Namespace) -> list[int]:
    """Interpreta as janelas de estimacao solicitadas."""

    if args.window_size is not None:
        return [args.window_size]
    return [int(value.strip()) for value in args.window_sizes.split(",") if value.strip()]


def select_tickers(args: argparse.Namespace) -> list[str]:
    """Seleciona a lista de ativos a partir do universo ou de arquivo externo."""

    if args.tickers_file:
        return read_tickers_file(args.tickers_file)
    if args.universe == "ibovespa" and FROZEN_TICKERS.exists():
        return read_tickers_file(str(FROZEN_TICKERS))
    return UNIVERSES[args.universe]


def load_or_download_index(args: argparse.Namespace):
    """Carrega ou baixa o indice benchmark."""

    if args.offline_synthetic:
        return None
    if INDEX_PRICES.exists():
        return load_prices(INDEX_PRICES)
    return download_prices(
        [args.index_ticker],
        start=args.start,
        end=args.end,
        output_path=INDEX_PRICES,
        cache_dir=RAW_DIR / ".yfinance-cache",
    )


def main() -> None:
    """Roda preparacao, backtest, metricas e exportacao."""

    args = parse_args()

    for path in [RAW_DIR, PROCESSED_RETURNS.parent, RESULTS_DIR, FIGURES_DIR, TABLES_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    raw_prices = RAW_DIR / ("synthetic_prices.csv" if args.offline_synthetic else f"{args.universe}_prices.csv")

    if args.offline_synthetic:
        prices = make_synthetic_prices()
        prices.to_csv(raw_prices)
    elif raw_prices.exists():
        prices = load_prices(raw_prices)
    else:
        tickers = select_tickers(args)
        prices = download_prices(
            yahoo_tickers(tickers),
            start=args.start,
            end=args.end,
            output_path=raw_prices,
            cache_dir=RAW_DIR / ".yfinance-cache",
        )

    returns = compute_returns(prices, method=args.return_method)
    returns.to_csv(PROCESSED_RETURNS)
    index_prices = load_or_download_index(args)
    index_returns = None
    if index_prices is not None:
        index_returns = compute_returns(index_prices, method=args.return_method).iloc[:, 0].rename("Index")
        index_returns.to_csv(PROCESSED_INDEX_RETURNS)

    for window_size in parse_window_sizes(args):
        windows = build_rolling_windows(returns, window_size=window_size, rebalance_frequency="ME")
        result = run_rolling_backtest(returns, windows, transaction_cost=args.transaction_cost)
        gross_returns = result.gross_returns.copy()
        net_returns = result.net_returns.copy()
        if index_returns is not None:
            aligned_index = index_returns.reindex(gross_returns.index).dropna()
            gross_returns.loc[aligned_index.index, "Index"] = aligned_index
            net_returns.loc[aligned_index.index, "Index"] = aligned_index
            gross_returns = gross_returns.dropna(subset=["Index"])
            net_returns = net_returns.reindex(gross_returns.index)

        net_metrics = performance_table(net_returns, result.turnover, result.target_turnover, result.weights)
        gross_metrics = performance_table(gross_returns, result.turnover, result.target_turnover, result.weights)

        window_results_dir = RESULTS_DIR / f"window_{window_size}"
        window_figures_dir = FIGURES_DIR / f"window_{window_size}"
        window_results_dir.mkdir(parents=True, exist_ok=True)
        window_figures_dir.mkdir(parents=True, exist_ok=True)

        net_returns.to_csv(window_results_dir / "net_returns.csv")
        gross_returns.to_csv(window_results_dir / "gross_returns.csv")
        result.turnover.to_csv(window_results_dir / "turnover.csv")
        result.target_turnover.to_csv(window_results_dir / "target_turnover.csv")

        save_tables(
            {
                f"metrics_net_T{window_size}": net_metrics,
                f"metrics_gross_T{window_size}": gross_metrics,
            },
            TABLES_DIR,
        )

        plot_cumulative_wealth(net_returns, window_figures_dir)
        plot_turnover(result.turnover, window_figures_dir)
        plot_weights(result.weights, window_figures_dir)
        plot_weight_distribution(result.weights, window_figures_dir)

    print("Pipeline concluido.")
    print(f"Universo: {args.universe}")
    print(f"Precos brutos: {raw_prices}")
    print(f"Retornos processados: {PROCESSED_RETURNS}")
    print(f"Janelas: {parse_window_sizes(args)}")
    print(f"Custo de transacao: {args.transaction_cost}")
    print(f"Tabelas: {TABLES_DIR}")
    print(f"Figuras: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
