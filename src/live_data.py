"""Fetch recent market data from Yahoo Finance for live predictions."""

import pandas as pd
import yfinance as yf

# Maps our column names to Yahoo Finance tickers. SPX uses the ^GSPC index;
# USO/SLV/GLD are the ETFs the training data was built from.
TICKERS = {
    "SPX": "^GSPC",
    "USO": "USO",
    "SLV": "SLV",
    "EUR/USD": "EURUSD=X",
    "GLD": "GLD",
}


def fetch_history(days: int = 90) -> pd.DataFrame:
    """Download recent daily closes, shaped like the training CSV.

    Returns a DataFrame with Date, SPX, GLD, USO, SLV, EUR/USD -- oldest
    first. Days where any series is missing (e.g. FX-only trading days)
    are dropped so all columns align.
    """
    raw = yf.download(
        list(TICKERS.values()),
        period=f"{days}d",
        interval="1d",
        progress=False,
        auto_adjust=True,
    )
    closes = raw["Close"].rename(columns={v: k for k, v in TICKERS.items()})
    closes = closes.dropna()
    df = closes.reset_index()
    df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    return df[["Date", "SPX", "GLD", "USO", "SLV", "EUR/USD"]]


def fetch_latest(days: int = 90):
    """Return (today, history) for a live prediction.

    today   -- dict of the most recent SPX/USO/SLV/EUR-USD closes
    history -- all prior rows (used for lag/rolling features), oldest first
    """
    df = fetch_history(days)
    if len(df) < 2:
        raise RuntimeError("Not enough market data returned from Yahoo Finance")
    latest = df.iloc[-1]
    today = {c: float(latest[c]) for c in ["SPX", "USO", "SLV", "EUR/USD"]}
    history = df.iloc[:-1].reset_index(drop=True)
    return today, history
