# stock_analysis.py
# Author: Patrick Zhu
# Description: Stock classification using Yahoo Finance + Logistic Regression with technical indicators

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 1) Import Dataset
print("Fetching data from Yahoo Finance...")
data = yf.download("AAPL", start="2020-01-01", end="2025-01-01")
data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

# 2) Inspect the Data  
print("\n--- Head ---\n", data.head())
print("\n--- Info ---")
data.info()  
print("\n--- Describe (numeric) ---\n", data.describe().T)

# Checking Missing value
print("\n--- Missing values per column ---\n", data.isna().sum())
dup_idx = data.index.duplicated().sum()
print(f"\n--- Duplicate index rows (by Date) --- {dup_idx}")

# 3) Feature processing: Technical Indicators
data["MA5"] = data["Close"].rolling(window=5).mean()
data["MA10"] = data["Close"].rolling(window=10).mean()

delta = data["Close"].diff()
gain = delta.where(delta > 0, 0.0)
loss = -delta.where(delta < 0, 0.0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
data["RSI14"] = 100 - (100 / (1 + rs))

data["BB_Mid"] = data["Close"].rolling(window=20).mean()
data["BB_Std"] = data["Close"].rolling(window=20).std()
data["BB_Upper"] = data["BB_Mid"] + (2 * data["BB_Std"])
data["BB_Lower"] = data["BB_Mid"] - (2 * data["BB_Std"])

# 4) Define Target (Tomorrow Up/Down)
data["Tomorrow"] = data["Close"].shift(-1)
data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)

# Data Modification
data["Return"] = data["Close"].pct_change()
data = data.dropna()

# 5) Basic Filtering 
filter_mask = (data["RSI14"] < 30) & (data["Close"] > data["MA10"])
filtered = data.loc[filter_mask]
print(f"\n--- Filter result: RSI<30 & Close>MA10 --- {filtered.shape[0]} rows")
print(filtered[["Close", "MA10", "RSI14"]].head())

# 6) Grouping / Resample
# 6a. Group by year
yearly_stats = (
    data.assign(Year=data.index.year)
        .groupby("Year")
        .agg(
            mean_return=("Return", "mean"),
            days=("Return", "count"),
            up_ratio=("Target", "mean"),
            avg_volume=("Volume", "mean"),
        )
)
print("\n--- Yearly stats (groupby) ---\n", yearly_stats)

# 6b. By month
monthly_stats = (
    data.resample("M")
        .agg(mean_return=("Return", "mean"), trading_days=("Return", "count"))
)
print("\n--- Monthly stats (resample) ---\n", monthly_stats.head())

# 7) ML Model
features = ["Open", "High", "Low", "Volume", "MA5", "MA10", "RSI14", "BB_Upper", "BB_Lower"]
X = data[features]
y = data["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
print("\n--- Machine Learning: Logistic Regression ---")
print("Accuracy:", log_reg.score(X_test, y_test))

# 8) Visualization
plt.figure(figsize=(12, 6))
plt.plot(data.index, data["Close"], label="Close Price")
plt.plot(data.index, data["MA5"], label="MA5")
plt.plot(data.index, data["MA10"], label="MA10")
plt.fill_between(data.index, data["BB_Upper"], data["BB_Lower"], color='gray', alpha=0.3, label="Bollinger Bands")
plt.title("Apple Stock with Technical Indicators")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.show()
