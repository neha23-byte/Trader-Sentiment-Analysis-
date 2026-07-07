# %% [markdown]
# # Trader Performance vs Market Sentiment
# ### Hyperliquid trading behavior analyzed against the Bitcoin Fear & Greed Index
#
# This notebook covers:
# - **Part A** — data loading, cleaning, and merge
# - **Part B** — sentiment vs performance analysis, behavior shifts, trader segments
# - **Part C** — actionable strategy rules
# - **Bonus** — a simple next-day profitability classifier and a trader clustering exercise

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["figure.figsize"] = (9, 5)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

CHART_DIR = "charts"
import os
os.makedirs(CHART_DIR, exist_ok=True)

def savefig(name):
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/{name}.png", dpi=140, bbox_inches="tight")
    plt.show()

# %% [markdown]
# ## Part A.1 — Load data and document shape / quality

# %%
trades = pd.read_csv("data/historical_trader_data.csv", low_memory=False)
sentiment = pd.read_csv("data/fear_greed_index.csv")

print("=== Trader data ===")
print("Shape:", trades.shape)
print("\nColumns:", list(trades.columns))
print("\nMissing values per column:\n", trades.isna().sum())
print("\nFully-duplicated rows:", trades.duplicated().sum())

# %%
print("=== Sentiment data ===")
print("Shape:", sentiment.shape)
print("\nColumns:", list(sentiment.columns))
print("\nMissing values per column:\n", sentiment.isna().sum())
print("\nFully-duplicated rows:", sentiment.duplicated().sum())
print("\nClassification counts:\n", sentiment["classification"].value_counts())

# %% [markdown]
# **Notes on data quality**
# - The trader file has a single fully-blank row (all-NaN) which is dropped.
# - The trader file does **not** include an explicit `leverage` column (despite being
#   mentioned in the assignment as an example field). We use **trade size in USD**
#   and **position-size changes** as behavioral proxies for risk-taking instead, and
#   call this out explicitly rather than inventing a leverage figure.
# - `Crossed` loads with mixed types (string "TRUE"/"FALSE"); harmless for this analysis.

# %%
trades = trades.dropna(subset=["Account"]).copy()

# Parse timestamps
trades["Timestamp IST"] = pd.to_datetime(trades["Timestamp IST"], format="%d-%m-%Y %H:%M", errors="coerce")
print("Rows with unparseable timestamps:", trades["Timestamp IST"].isna().sum())
trades = trades.dropna(subset=["Timestamp IST"])
trades["date"] = trades["Timestamp IST"].dt.date

sentiment["date"] = pd.to_datetime(sentiment["date"]).dt.date

print("\nTrader data date range:", trades["date"].min(), "to", trades["date"].max())
print("Sentiment data date range:", sentiment["date"].min(), "to", sentiment["date"].max())
print("Unique trader accounts:", trades["Account"].nunique())
print("Unique coins traded:", trades["Coin"].nunique())

# %% [markdown]
# ## Part A.2 — Align datasets by date

# %%
daily_sentiment = sentiment[["date", "classification", "value"]].drop_duplicates(subset="date")

merged = trades.merge(daily_sentiment, on="date", how="left")
print("Trades with no matching sentiment day:", merged["classification"].isna().sum(), "/", len(merged))

# Collapse the 5-class classification into a simple Fear / Greed / Neutral bucket for
# the headline comparisons the assignment asks for, while keeping the full label too.
def simplify(label):
    if pd.isna(label):
        return np.nan
    if "Fear" in label:
        return "Fear"
    if "Greed" in label:
        return "Greed"
    return "Neutral"

merged["sentiment_simple"] = merged["classification"].apply(simplify)
merged = merged.dropna(subset=["sentiment_simple"])
print("Final merged trade rows:", len(merged))
print(merged["sentiment_simple"].value_counts())

# %% [markdown]
# ## Part A.3 — Build key metrics
#
# We build two metric tables:
# 1. **Per (account, day)** — the core panel used for most of the analysis
# 2. **Per day (all accounts pooled)** — used for the sentiment-level comparisons

# %%
merged["is_win"] = merged["Closed PnL"] > 0
merged["is_realized"] = merged["Closed PnL"] != 0  # only realized-PnL rows count for win rate

account_day = merged.groupby(["Account", "date", "sentiment_simple"]).agg(
    daily_pnl=("Closed PnL", "sum"),
    n_trades=("Closed PnL", "size"),
    n_realized=("is_realized", "sum"),
    n_wins=("is_win", "sum"),
    avg_trade_size_usd=("Size USD", "mean"),
    total_volume_usd=("Size USD", "sum"),
    n_buy=("Direction", lambda s: (s == "Buy").sum()),
    n_sell=("Direction", lambda s: (s == "Sell").sum()),
).reset_index()

account_day["win_rate"] = np.where(account_day["n_realized"] > 0,
                                    account_day["n_wins"] / account_day["n_realized"], np.nan)
account_day["long_short_ratio"] = np.where(account_day["n_sell"] > 0,
                                            account_day["n_buy"] / account_day["n_sell"],
                                            np.nan)

print(account_day.shape)
account_day.head()

# %%
daily_market = merged.groupby(["date", "sentiment_simple"]).agg(
    total_pnl=("Closed PnL", "sum"),
    n_trades=("Closed PnL", "size"),
    n_active_accounts=("Account", "nunique"),
    avg_trade_size_usd=("Size USD", "mean"),
    n_buy=("Direction", lambda s: (s == "Buy").sum()),
    n_sell=("Direction", lambda s: (s == "Sell").sum()),
).reset_index()
daily_market["long_short_ratio"] = daily_market["n_buy"] / daily_market["n_sell"]
daily_market["trades_per_account"] = daily_market["n_trades"] / daily_market["n_active_accounts"]

print(daily_market.shape)
daily_market.head()

# %% [markdown]
# A simple **drawdown proxy** per account-day: the worst single losing trade of the day,
# expressed as a share of that day's traded volume. This approximates how "painful" a bad
# day was, without needing intraday equity curves (not available in this fills-only data).

# %%
def worst_trade_share(g):
    losses = g.loc[g["Closed PnL"] < 0, "Closed PnL"]
    vol = g["Size USD"].sum()
    if vol == 0 or losses.empty:
        return 0.0
    return abs(losses.min()) / vol

dd_proxy = merged.groupby(["Account", "date", "sentiment_simple"]).apply(worst_trade_share).reset_index(name="drawdown_proxy")
account_day = account_day.merge(dd_proxy, on=["Account", "date", "sentiment_simple"])
print(account_day[["drawdown_proxy"]].describe())

# %% [markdown]
# ## Part B.1 — Does performance differ between Fear and Greed days?

# %%
perf_by_sentiment = account_day.groupby("sentiment_simple").agg(
    mean_daily_pnl=("daily_pnl", "mean"),
    median_daily_pnl=("daily_pnl", "median"),
    mean_win_rate=("win_rate", "mean"),
    mean_drawdown_proxy=("drawdown_proxy", "mean"),
    n_account_days=("daily_pnl", "size"),
).round(3)
print(perf_by_sentiment)

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
order = ["Extreme Fear" if False else "Fear", "Neutral", "Greed"]
order = [o for o in ["Fear", "Neutral", "Greed"] if o in perf_by_sentiment.index]

perf_by_sentiment.loc[order, "mean_daily_pnl"].plot(kind="bar", ax=axes[0], color=["#d64545","#c9a227","#3a8f4f"])
axes[0].set_title("Mean daily PnL per trader")
axes[0].set_ylabel("USD")

perf_by_sentiment.loc[order, "mean_win_rate"].plot(kind="bar", ax=axes[1], color=["#d64545","#c9a227","#3a8f4f"])
axes[1].set_title("Mean win rate per trader-day")
axes[1].set_ylabel("Win rate")

perf_by_sentiment.loc[order, "mean_drawdown_proxy"].plot(kind="bar", ax=axes[2], color=["#d64545","#c9a227","#3a8f4f"])
axes[2].set_title("Mean drawdown proxy per trader-day")
axes[2].set_ylabel("Worst-trade / day-volume")

savefig("01_performance_by_sentiment")

# %% [markdown]
# ## Part B.2 — Do traders change behavior with sentiment?
# (trade frequency, position size, long/short bias)

# %%
behavior_by_sentiment = account_day.groupby("sentiment_simple").agg(
    mean_trades_per_acct_day=("n_trades", "mean"),
    mean_trade_size_usd=("avg_trade_size_usd", "mean"),
    mean_long_short_ratio=("long_short_ratio", "mean"),
).round(3)
print(behavior_by_sentiment)

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
behavior_by_sentiment.loc[order, "mean_trades_per_acct_day"].plot(kind="bar", ax=axes[0], color=["#d64545","#c9a227","#3a8f4f"])
axes[0].set_title("Avg trades per account-day")

behavior_by_sentiment.loc[order, "mean_trade_size_usd"].plot(kind="bar", ax=axes[1], color=["#d64545","#c9a227","#3a8f4f"])
axes[1].set_title("Avg trade size (USD)")

behavior_by_sentiment.loc[order, "mean_long_short_ratio"].plot(kind="bar", ax=axes[2], color=["#d64545","#c9a227","#3a8f4f"])
axes[2].axhline(1.0, color="black", linewidth=1, linestyle="--")
axes[2].set_title("Long / Short ratio (buys per sell)")

savefig("02_behavior_by_sentiment")

# %% [markdown]
# ## Part B.3 — Trader segments
#
# We define three simple, reproducible segments using each account's full-year stats:
# - **Position size**: high vs low average trade size (median split)
# - **Activity**: frequent vs infrequent traders (median split on active days)
# - **Consistency**: consistent winners vs inconsistent, based on the *std-dev* of
#   daily PnL relative to its mean (coefficient of variation)

# %%
account_stats = account_day.groupby("Account").agg(
    total_pnl=("daily_pnl", "sum"),
    mean_daily_pnl=("daily_pnl", "mean"),
    std_daily_pnl=("daily_pnl", "std"),
    active_days=("date", "nunique"),
    avg_trade_size=("avg_trade_size_usd", "mean"),
    mean_win_rate=("win_rate", "mean"),
).reset_index()

account_stats["size_segment"] = np.where(
    account_stats["avg_trade_size"] >= account_stats["avg_trade_size"].median(), "High size", "Low size")
account_stats["activity_segment"] = np.where(
    account_stats["active_days"] >= account_stats["active_days"].median(), "Frequent", "Infrequent")
account_stats["cv"] = account_stats["std_daily_pnl"] / account_stats["mean_daily_pnl"].replace(0, np.nan).abs()
account_stats["consistency_segment"] = np.where(
    account_stats["cv"] <= account_stats["cv"].median(skipna=True), "Consistent", "Inconsistent")

print(account_stats[["size_segment","activity_segment","consistency_segment"]].apply(pd.Series.value_counts))

# %%
seg_perf = account_day.merge(
    account_stats[["Account","size_segment","activity_segment","consistency_segment"]], on="Account")

for seg_col, title in [("size_segment","Position size"), ("activity_segment","Trading activity"),
                        ("consistency_segment","Consistency")]:
    pivot = seg_perf.groupby([seg_col, "sentiment_simple"])["daily_pnl"].mean().unstack()[order]
    pivot.T.plot(kind="bar", figsize=(8,4.5))
    plt.title(f"Mean daily PnL by sentiment — segment: {title}")
    plt.ylabel("USD")
    plt.legend(title=seg_col)
    savefig(f"03_segment_{seg_col}")

# %% [markdown]
# ## Insights
#
# The three charts and tables above are read together to produce the written insights
# in `WRITEUP.md`. Nothing here is hard-coded — the numbers driving the write-up are
# regenerated every time this notebook is run.

# %% [markdown]
# ## Part C — Strategy ideas
# See `WRITEUP.md` for the two rule-of-thumb strategies derived from the segment results
# above (kept in a separate file so the notebook stays focused on the analysis).

# %% [markdown]
# ## Bonus 1 — Predict next-day profitability bucket
#
# For each account we build a simple day-level feature set (yesterday's sentiment,
# trade count, size, win rate) and try to predict whether **tomorrow** is a profitable
# day for that account. This is intentionally simple (logistic regression) — the point
# is to show the pipeline, not to claim a production-ready signal from ~1 year of data.

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler

panel = account_day.sort_values(["Account", "date"]).copy()
panel["sentiment_code"] = panel["sentiment_simple"].map({"Fear": -1, "Neutral": 0, "Greed": 1})

# next-day target, computed per account
panel["next_day_pnl"] = panel.groupby("Account")["daily_pnl"].shift(-1)
panel["target_profitable"] = (panel["next_day_pnl"] > 0).astype(int)

feature_cols = ["daily_pnl", "n_trades", "avg_trade_size_usd", "win_rate",
                "long_short_ratio", "drawdown_proxy", "sentiment_code"]
model_df = panel.dropna(subset=feature_cols + ["next_day_pnl"]).copy()
model_df["win_rate"] = model_df["win_rate"].fillna(0)
model_df["long_short_ratio"] = model_df["long_short_ratio"].replace([np.inf, -np.inf], np.nan)
model_df = model_df.dropna(subset=["long_short_ratio"])

print("Modeling rows:", len(model_df))
print("Positive rate:", model_df["target_profitable"].mean().round(3))

X = model_df[feature_cols]
y = model_df["target_profitable"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
scaler = StandardScaler().fit(X_train)
clf = LogisticRegression(max_iter=1000).fit(scaler.transform(X_train), y_train)

pred = clf.predict(scaler.transform(X_test))
proba = clf.predict_proba(scaler.transform(X_test))[:, 1]

print("Accuracy:", round(accuracy_score(y_test, pred), 3))
print("ROC AUC:", round(roc_auc_score(y_test, proba), 3))
print(classification_report(y_test, pred))

coef_table = pd.Series(clf.coef_[0], index=feature_cols).sort_values()
print("\nStandardized coefficients (direction of effect):")
print(coef_table.round(3))

# %%
coef_table.plot(kind="barh", color="#3a6ea5")
plt.title("Logistic regression coefficients — predicting next-day profitability")
plt.axvline(0, color="black", linewidth=1)
savefig("04_model_coefficients")

# %% [markdown]
# ## Bonus 2 — Clustering traders into behavioral archetypes

# %%
from sklearn.cluster import KMeans

cluster_features = account_stats[["mean_daily_pnl", "active_days", "avg_trade_size", "mean_win_rate"]].copy()
cluster_features = cluster_features.fillna(cluster_features.median())

scaler2 = StandardScaler()
X_clust = scaler2.fit_transform(cluster_features)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X_clust)
account_stats["cluster"] = kmeans.labels_

cluster_profile = account_stats.groupby("cluster")[
    ["mean_daily_pnl", "active_days", "avg_trade_size", "mean_win_rate"]].mean().round(2)
cluster_profile["n_accounts"] = account_stats["cluster"].value_counts().sort_index()
print(cluster_profile)

# %%
fig, ax = plt.subplots(figsize=(8,5.5))
scatter = ax.scatter(account_stats["avg_trade_size"], account_stats["mean_daily_pnl"],
                      c=account_stats["cluster"], cmap="tab10", s=60, alpha=0.8)
ax.set_xscale("log")
ax.set_xlabel("Avg trade size (USD, log scale)")
ax.set_ylabel("Mean daily PnL (USD)")
ax.set_title("Trader archetypes (KMeans, k=4)")
legend1 = ax.legend(*scatter.legend_elements(), title="Cluster")
ax.add_artist(legend1)
savefig("05_trader_clusters")

# %% [markdown]
# ## Save output tables for the write-up

# %%
os.makedirs("output_tables", exist_ok=True)
perf_by_sentiment.to_csv("output_tables/perf_by_sentiment.csv")
behavior_by_sentiment.to_csv("output_tables/behavior_by_sentiment.csv")
cluster_profile.to_csv("output_tables/cluster_profile.csv")
account_stats.to_csv("output_tables/account_stats.csv", index=False)
print("Saved summary tables to output_tables/")
