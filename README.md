# robustifying-markowitz-b3

Estrutura enxuta para reproduzir experimentos de carteiras Markowitz robustas com ativos da B3.

O projeto separa o pipeline em sete módulos principa `src/data.pwnload, limpeza, retornos e janelas moveis.
- `src/estimators.ptimadores de covariancia classicos, shrinkage e robustos.
- `src/optimization.pimizacao, PGD e projecoes de pesos.
- `src/portfolios.pnstrucao das carteiras.
- `src/backtest.plling window, rebalanceamento e custos.
- `src/metrics.ptricas de performance, risco e pesos.
- `src/plots.pguras e tabelas finais.

## Estrutura

```text
robustifying-markowitz-b3/
├── README.md
├── requirements.txt
├── run.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── results/
├── src/
│   ├── data.py
│   ├── estimators.py
│   ├── optimization.py
│   ├── portfolios.py
│   ├── backtest.py
│   ├── metrics.py
│   └── plots.py
└── outputs/
    ├── figures/
    └── tables/
```

## Uso

Instale as dependenci``bash
pip install -r requirements.txt
```

Execute o pipeline com o universo Iboves``bash
python run.py
```

Por padrao, essa chamada executa as duas janelas do paper, `T=252` e `T=500`,
com custo proporcional de transacao de 50 bps (`0.005`). Os resultados ficam  `data/results/window_252/`
- `data/results/window_500/`
- `outputs/tables/metrics_net_T252.csv`
- `outputs/tables/metrics_net_T500.csv`

Por padrao, `run.py` usa o universo congelado em
`data/raw/tickers_ibovespa_frozen_20260702.txt`, baixa ou carrega precos ajustados
via Yahoo Finance e salva em `data/raw/ibovespa_prices.csv`. A carteira teorica da
B3 muda periodicamente; por isso o experimento principal fica fixado neste arquivo.
Para usar outra lista congelada, ro``bash
python run.py --tickers-file data/raw/tickers_ibovespa.txt
```

Para testar o pipeline sem intern``bash
python run.py --offline-synthetic
```

O pipeline usa retornos simples por padrao, pois riqueza acumulada, custos de transacao
e drift dos pesos entre rebalanceamentos sao calculados em retornos simples. Para rodar
com log-retornos na etapa de estimacao, use `--return-method log` com cautela ao comparar
metricas de riqueza.

## Coerencia com o paper

A carteira `GMV Robust` segue a logica central do arti vez de estimar e inverter uma
covariancia robusta completa, ela usa PGD a partir da carteira EW e estima diretamente a
acao `Sigma w` por blocos median-of-means. A agregacao dos vetores de bloco usa um
procedimento spectral-center/Hrotina alterna entre a direcao de maior dispersao e
pesos no simplex capado `Delta_{l,epsilon}`, reduzindo a influencia dos blocos extremos.
A projecao robusta e feita apenas na restricao orcamentaria `sum(w) = 1`, permitindo pesos
negativos, como no paper.

O benchmark `Index` usa o ticker `^BVSP` por padrao e e salvo em
`data/raw/ibovespa_index.csv`. Ele entra nas tabelas de performance com turnover zero,
como o indice nos experimentos do artigo.

O turnover reportado em `outputs/tables/metrics_*.csv` sepa `turnovema absoluta dos trades para rebalancear a carteira apos o drift de precos.
- `target_turnovedanca entre pesos-alvo, limpando o efeito dos precos.

`GMV Nonlinear Shrinkage` usa uma implementacao espectral interna inspirada em
Ledoit-Wo autovetores da covariancia amostral sao preservados e os autovalores
sao encolhidos de forma nao linear por meio de uma estimativa suavizada do Stieltjes
transform. Assim, ele deixa de ser uma proxy OAS e passa a representar o benchmark
nonlinear shrinkage do horse race.

As tabelas `metrics_gross.csv` e `metrics_net.csv` tambem incluem p-values contra
`GMV Robus `turnover_pvalue_vs_robustest pareado para turnover.
- `target_turnover_pvalue_vs_robustest pareado para target turnover.
- `sharpe_pvalue_vs_robusste HAC/Newey-West por delta method para Sharpe.
- `variance_pvalue_vs_robusste HAC/Newey-West por delta method para variancia.

## Validacao

O reposititorio inclui scripts de auditor``bash
python scripts/validate_nonlinear_shrinkage.py
python scripts/audit_hlz_spectral_center.py
```

Para validar numericamente o nonlinear shrinkage contra a implementacao original de
Ledoit-Wolf, exporte a matriz de referencia oficial para CSV e ro``bash
python scripts/validate_nonlinear_shrinkage.py --reference-csv caminho/para/reference_lw2017.csv
```

Sem uma matriz oficial de referencia, o script valida apenas propriedades internas
necessarias, como simetria e autovalores positivos.

## Fluxo do pipeline

1. Baixa ou carrega dados.
2. Limpa precos e calcula retornos.
3. Constroi janelas moveis.
4. Estima covariancias.
5. Constroi carteiras.
6. Executa backtest com custos.
7. Calcula metricas e salva tabelas/figuras.

