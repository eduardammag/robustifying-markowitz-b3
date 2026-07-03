"""Universos de ativos brasileiros usados no experimento."""

from __future__ import annotations


# Ponto de partida pratico para o Ibovespa via Yahoo Finance.
# A carteira teorica da B3 muda periodicamente; para reproducao estrita,
# congele a lista usada no paper em um CSV e passe via --tickers-file.
IBOVESPA_TICKERS = [
    "ABEV3",
    "ALOS3",
    "ASAI3",
    "AURE3",
    "AZUL4",
    "B3SA3",
    "BBAS3",
    "BBDC3",
    "BBDC4",
    "BBSE3",
    "BEEF3",
    "BPAC11",
    "BRAP4",
    "BRAV3",
    "BRFS3",
    "BRKM5",
    "CCRO3",
    "CMIG4",
    "CMIN3",
    "COGN3",
    "CPFE3",
    "CPLE6",
    "CRFB3",
    "CSAN3",
    "CSNA3",
    "CYRE3",
    "ELET3",
    "ELET6",
    "EMBR3",
    "ENGI11",
    "EQTL3",
    "FLRY3",
    "GGBR4",
    "GOAU4",
    "HAPV3",
    "HYPE3",
    "IGTI11",
    "IRBR3",
    "ITSA4",
    "ITUB4",
    "JBSS3",
    "KLBN11",
    "LREN3",
    "MGLU3",
    "MRFG3",
    "MRVE3",
    "MULT3",
    "NTCO3",
    "PCAR3",
    "PETR3",
    "PETR4",
    "PETZ3",
    "PRIO3",
    "RADL3",
    "RAIL3",
    "RAIZ4",
    "RDOR3",
    "RECV3",
    "RENT3",
    "SANB11",
    "SBSP3",
    "SLCE3",
    "SMTO3",
    "SUZB3",
    "TAEE11",
    "TIMS3",
    "TOTS3",
    "UGPA3",
    "USIM5",
    "VALE3",
    "VAMO3",
    "VBBR3",
    "VIVA3",
    "VIVT3",
    "WEGE3",
    "YDUQ3",
]


UNIVERSES = {
    "ibovespa": IBOVESPA_TICKERS,
}


def yahoo_tickers(tickers: list[str]) -> list[str]:
    """Converte tickers B3 para o formato usado pelo Yahoo Finance."""

    return [ticker if ticker.endswith(".SA") else f"{ticker}.SA" for ticker in tickers]


def read_tickers_file(path: str) -> list[str]:
    """Le um arquivo simples com um ticker por linha."""

    with open(path, encoding="utf-8") as file:
        return [
            line.strip().upper().removesuffix(".SA")
            for line in file
            if line.strip() and not line.strip().startswith("#")
        ]

