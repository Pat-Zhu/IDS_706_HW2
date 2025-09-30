from __future__ import annotations
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

FEATURE_FOR_MODEL = [
    "Open",
    "High",
    "Low",
    "Volume",
    "MA5",
    "MA10",
    "RSI14",
    "BB_Upper",
    "BB_Lower",
]


def load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    if df.empty:
        raise ValueError("Downloaded dataframe is empty.")
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="first")]
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    return df


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index over the closing price."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _compute_bbands(close: pd.Series, window: int = 20, k: float = 2.0):
    """Bollinger Bands (mid, upper, lower)."""
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = mid + k * std
    lower = mid - k * std
    return mid, upper, lower


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add MA5/MA10, RSI(14), and Bollinger Bands."""
    d = df.copy()
    d["MA5"] = d["Close"].rolling(5).mean()
    d["MA10"] = d["Close"].rolling(10).mean()
    d["RSI14"] = _compute_rsi(d["Close"], period=14)
    d["BB_Mid"], d["BB_Upper"], d["BB_Lower"] = _compute_bbands(d["Close"], window=20, k=2.0)
    # NEW: tests expect Return right after add_indicators
    d["Return"] = d["Close"].pct_change()
    return d


def define_target(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["Tomorrow"] = d["Close"].shift(-1)
    d["Target"] = (d["Tomorrow"] > d["Close"]).astype(int)
    d["Return"] = d["Close"].pct_change()
    return d.dropna()


def oversold_bounce_filter(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["RSI14"] < 30) & (df["Close"] > df["MA10"])]


def yearly_stats(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.assign(Year=df.index.year)
        .groupby("Year")
        .agg(
            mean_return=("Return", "mean"),
            days=("Return", "count"),
            up_ratio=("Target", "mean"),
            avg_volume=("Volume", "mean"),
        )
    )


def monthly_stats(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("ME").agg(mean_return=("Return", "mean"), trading_days=("Return", "count"))


def get_xy(df: pd.DataFrame):
    return df[FEATURE_FOR_MODEL], df["Target"]


def train_logreg(X: pd.DataFrame, y: pd.Series, random_state: int = 42):
    if y.nunique() < 2:
        raise ValueError("Need at least two classes in y for classification.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    return model, float(acc)


def run_pipeline(ticker="AAPL", start="2020-01-01", end="2025-01-01") -> dict:
    df = load_data(ticker, start, end)
    df = add_indicators(df)
    df = define_target(df)
    X, y = get_xy(df)
    model, acc = train_logreg(X, y)
    bounce = oversold_bounce_filter(df)
    y_stats = yearly_stats(df)
    m_stats = monthly_stats(df)
    return {
        "accuracy": acc,
        "rows": int(len(df)),
        "signals": int(len(bounce)),
        "yearly_index": list(y_stats.index.astype(int)),
        "monthly_rows": int(len(m_stats)),
    }
