# IDS_706_HW2
## 1. Project Overview
This project downloads daily Apple (AAPL) stock data from the public **Yahoo Finance** API and performs:
- **Data inspection** with Pandas (`head`, `info`, `describe`, missing/duplicate checks)
- **Feature engineering** (MA5, MA10, RSI(14), Bollinger Bands)
- **Filtering & grouping** examples (oversold RSI filter; yearly and monthly summaries)
- **Machine learning**: Logistic Regression to classify whether tomorrow’s close is higher than today’s  
- **Visualization**: Close price with moving averages and Bollinger Bands

## 2. Dataset
- **Source:** Yahoo Finance (public API) using `yfinance` package in python
- **Ticker:** `AAPL`
- **Period requested:** `2020-01-01` to `2025-01-01`  

## 3. Setup instruction

### Requirements
- Python
- Packages: `pandas`, `yfinance`, `scikit-learn`, `matplotlib`

### Quickstart

```
# run
python stock_analysis.py
```
## 4. Data Analysis

`head()` to preview rows
`info()` for types and non-null counts
`describe()` for summary statistics
Checks for missing values and duplicate dates
There is no missing value or duplicate date

```
# Data inspection
data.head()
data.info()
data.describe().T
data.isna().sum()
data.index.duplicated().sum()
```
To capture trend, momentum, and volatility, I created some new features in the dataset:
Moving averages: `MA5`, `MA10`
RSI(14)
Bollinger Bands (20, 2): `BB_Mid`, `BB_Upper`, `BB_Lower`

**Example**
```
data["MA5"] = data["Close"].rolling(window=5).mean()
data["MA10"] = data["Close"].rolling(window=10).mean()
```

Then I defined the target following the research question: Will tomorrow’s close be higher than today’s?

```
data["Tomorrow"] = data["Close"].shift(-1)
data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)
```

Next, to demonstrate filtering, I looked for potential “oversold bounce” days following the condition of `RSI14 < 30` and `Close > MA10`.

**Example**

```
filter_mask = (data["RSI14"] < 30) & (data["Close"] > data["MA10"])
filtered = data.loc[filter_mask]
print(f"\n--- Filter result: RSI<30 & Close>MA10 --- {filtered.shape[0]} rows")
print(filtered[["Close", "MA10", "RSI14"]].head())
```
There are three days that are oversold bounce after the filtering, which are:                                    
`2023-08-22` `2023-09-25` `2024-01-12`

The next step will be grouping, I did two types of grouping, groupby year and sample by month

**Example**
```
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

monthly_stats = (
    data.resample("M")
        .agg(mean_return=("Return", "mean"), trading_days=("Return", "count"))
)
print("\n--- Monthly stats (resample) ---\n", monthly_stats.head())
```

After done with all the data inspection and manipulation, we can apply the logistic model:
I used 
`features = ["Open", "High", "Low", "Volume", "MA5", "MA10", "RSI14", "BB_Upper", "BB_Lower"]`
as my features for the logistic regression

Data were split 80/20 for training and testing

```
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
```

## 5. Data Analysis Outcomes/Takeaways
1) The dataset is clean (no missing values or duplicate dates).
2) Annual summaries show clear regime differences.
3) The strict RSI filter yields few signals (3 days), useful as a filtering example.
4) The logistic model achieves around 51% accuracy; improvements likely require feature scaling, richer features (lags, ranges, volatility) or using alternative models.

## 6. Data Visualization
I plotted Close, MA5/MA10, and Bollinger Bands to visually validate the indicators and observe volatility regimes.
The graph is a seperate file in this repository.

## 7. Week 3 Reproducible & Testable Setup

### 7.1 Repo Structure

```text
IDS_706_HW2/
├─ src/
│  ├─ __init__.py
│  ├─ stock_utils.py
│  └─ stock_analysis_cli.py
├─ tests/
│  ├─ test_stock_utils.py
│  └─ test_end_to_end.py
├─ requirements.txt
├─ Dockerfile
├─ .dockerignore
├─ pytest.ini
├─ Makefile
└─ README.md
```

### 7.2 Local (venv) — Install, Test, Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# tests
pytest -q

# CLI: prints a JSON summary (accuracy/rows/signals/…)
python -m src.stock_analysis_cli --ticker AAPL --start 2020-01-01 --end 2025-01-01
```



