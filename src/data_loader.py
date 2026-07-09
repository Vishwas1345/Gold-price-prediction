"""Load and clean the historical gold price dataset."""

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "gld_price_data.csv"
PRICE_COLUMNS = ["SPX", "GLD", "USO", "SLV", "EUR/USD"]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the daily price CSV, sorted chronologically.

    Chronological order is required everywhere downstream: lag/rolling
    features and the time-based train/test split both assume it.
    """
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.dropna(subset=PRICE_COLUMNS)
    df = df.drop_duplicates(subset="Date", keep="last")
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def time_based_split(df: pd.DataFrame, test_fraction: float = 0.2):
    """Split chronologically: the most recent `test_fraction` of rows is the test set.

    A random split leaks future information into training for time series
    data (neighboring days land on both sides of the split), which is what
    inflated this project's original accuracy number.
    """
    cut = int(len(df) * (1 - test_fraction))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()
