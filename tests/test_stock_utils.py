import pandas as pd
import numpy as np
import pytest
from src.stock_utils import (
    add_indicators, define_target, oversold_bounce_filter,
    yearly_stats, monthly_stats, get_xy, train_logreg, load_data
)

def make_sample_ohlcv(n=70, seed=0):
    rng = pd.date_range("2024-01-01", periods=n, freq="B")
    rs = np.random.RandomState(seed)
    close = np.cumsum(rs.randn(n)) + 100
    high = close + rs.rand(n)
    low = close - rs.rand(n)
    open_ = close + rs.randn(n) * 0.2
    vol = rs.randint(1_000_000, 2_000_000, size=n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close, "Volume": vol}, index=rng)

def test_add_indicators_and_target_shapes():
    df = make_sample_ohlcv(80)
    df2 = add_indicators(df)
    assert {"MA5","MA10","RSI14","BB_Upper","BB_Lower","Return"}.issubset(df2.columns)
    df3 = define_target(df2)
    assert "Target" in df3.columns
    assert set(df3["Target"].unique()) <= {0,1}

def test_filter_and_grouping():
    df = define_target(add_indicators(make_sample_ohlcv(120)))
    b = oversold_bounce_filter(df)
    assert isinstance(b, pd.DataFrame)
    ys = yearly_stats(df)
    ms = monthly_stats(df)
    assert {"mean_return","days","up_ratio","avg_volume"}.issubset(ys.columns)
    assert {"mean_return","trading_days"}.issubset(ms.columns)

def test_train_logreg_works():
    df = define_target(add_indicators(make_sample_ohlcv(90)))
    X, y = get_xy(df)
    model, acc = train_logreg(X, y)
    assert 0.0 <= acc <= 1.0

def test_load_data_empty(monkeypatch):
    import yfinance as yf
    def fake_download(*args, **kwargs):
        return pd.DataFrame()
    monkeypatch.setattr(yf, "download", fake_download)
    with pytest.raises(ValueError):
        load_data("AAPL", "2020-01-01", "2020-02-01")
