# stock_analysis.py
# Author: Patrick Zhu
# Description: Stock classification using Yahoo Finance + Logistic Regression with technical indicators

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 1. Import Dataset

print("Fetching data from Yahoo Finance...")
data = yf.download("AAPL", start="2020-01-01", end="2025-01-01")

data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

print("\n--- Head ---\n", data.head())

# 2. Feature Engineering: Technical Indicators

# Moving Average Calculation
data["MA5"] = data["Close"].rolling(window=5).mean()
data["MA10"] = data["Close"].rolling(window=10).mean()

# RSI calculation
delta = data["Close"].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
data["RSI14"] = 100 - (100 / (1 + rs))

# Bollinger Bands Calculation
data["BB_Mid"] = data["Close"].rolling(window=20).mean()
data["BB_Std"] = data["Close"].rolling(window=20).std()
data["BB_Upper"] = data["BB_Mid"] + (2 * data["BB_Std"])
data["BB_Lower"] = data["BB_Mid"] - (2 * data["BB_Std"])

# 3. Define Target (Tomorrow Up/Down)

data["Tomorrow"] = data["Close"].shift(-1)
data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)

# Delete NAs
data = data.dropna()

# 4. Features & ML Model

features = ["Open", "High", "Low", "Volume", "MA5", "MA10", "RSI14", "BB_Upper", "BB_Lower"]
X = data[features]
y = data["Target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

print("\n--- Machine Learning: Logistic Regression ---")
print("Accuracy:", log_reg.score(X_test, y_test))

# 5. Visualization saved as Figure 1

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
