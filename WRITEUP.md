# Primetrade.ai Data Science Intern Assignment: Write-up
**Prepared by:** neha23-byte
**Topic:** Trader Performance vs. Market Sentiment on Hyperliquid

---

## 1. Methodology
This analysis evaluates how Bitcoin market sentiment (tracked via the Fear & Greed Index) interacts with empirical trader execution behavior and profitability metrics on Hyperliquid. 

### Data Preparation & Cleansing
* **Datasets Merged:** `historical_trader_data.csv` (211,224 execution rows) and `fear_greed_index_2.csv` (2,644 historical daily sentiment metrics).
* **Granularity & Alignment:** Both datasets were aligned chronologically at the daily level using normalized date fields derived from execution timestamps.
* **Simplification:** Granular classification buckets (`Extreme Fear`, `Fear`, `Neutral`, `Greed`, `Extreme Greed`) were mapped into three clean operational regimes: **Fear**, **Neutral**, and **Greed**.
* **Proxy Selection:** Because an explicit leverage feature was unavailable in the raw data schema, total trade size (`Size USD`) and single-account daily aggregated position sizes were utilized as empirical structural proxies for leverage and exposure depth.

---

## 2. Key Insights & Findings (Part B)

### Insight 1: Profitability Scaling vs. Consistency Regimes
* **The Data:** On **Fear** days, traders captured a substantially higher mean daily PnL of **$9,385.31** (with an average win rate of **89.2%**), whereas **Greed** days yielded a lower mean daily PnL of **$5,372.37** (win rate **87.3%**).
* **The Interpretation:** Volatility shocks during Fear regimes create wider mispricings that premium traders exploit for large, outsized absolute gains. However, looking at the median daily PnL (**$238.33** in Greed vs. **$142.99** in Fear), the typical everyday account actually secures more consistent, stable baseline gains during quiet, trending Greed environments.

### Insight 2: Aggressive Sentiment-Driven Behavioral Shifts
* **The Data:** Trade frequency surges dramatically during market stress. Traders average **106.43 trades per account-day** during **Fear** conditions, dropping by more than half to **50.37 trades per account-day** during **Greed**.
* **The Interpretation:** Fear triggers rapid position adjustments, scaling, and hyper-active intraday scalping. Conversely, Greed conditions foster patience, leading to lower-frequency, longer-duration trend positioning.

### Insight 3: The Long/Short Bias Reversal
* **The Data:** The empirical Long/Short ratio spikes to **1.645** during **Fear** conditions but collapses to an ultra-skewed **0.479** during **Greed**.
* **The Interpretation:** Hyperliquid traders exhibit a stark contrarian behavior profile. When the broader market panics (Fear), these accounts aggressively build long exposure to catch local bottoms. When the market reaches a euphoric state (Greed), they heavily tilt their books short to capitalize on distribution tops and impending corrections.

---

## 3. Behavioral Archetypes (Clustering)
A K-Means analysis separated the trader population into four clear operational archetypes (`cluster_profile.csv`):
* **Cluster 0 (High-Exposure Whales):** High average trade size ($30,586), low active days (31.5), but massive mean daily PnL outcomes ($44,112.77). Highly opportunistic.
* **Cluster 1 (Sniper Specialists):** Low average trade size ($3,984) and moderate activity, but an exceptional **97% win rate**, pulling a highly efficient $15,225.09 mean daily PnL.
* **Cluster 2 (Retail Grind):** High active presence (79.25 days), moderate position size ($4,469), but lower efficiency (82% win rate) resulting in a modest $2,569.35 daily return.
* **Cluster 3 (High-Frequency Volume Drivers):** Hyper-persistent accounts present for **288.5 active days** with an average trade size of $10,266.98. They are the core baseline liquidity providers.

---

## 4. Actionable Strategy Rules of Thumb (Part C)

### Rule 1: The Contrarian Liquidity Scalping Rule
* **Condition:** When the market enters an extended **Fear** regime, institutional accounts should deploy high-frequency market-making algorithms to capture the massive surge in activity (**106.4 trades/day**). 
* **Execution:** Broaden execution spreads to capture volatile premium chunks, while strictly maintaining a net-long exposure structural bias to align with successful contrarian whale patterns.

### Rule 2: Exposure Ceiling Policy on Euphoria
* **Condition:** When the market shifts to a prolonged **Greed** regime, downscale overall capital exposure limits.
* **Execution:** Caps should be placed on high-frequency execution pathways because trade frequency contracts naturally to **~50 trades/day** and absolute mean PnL velocity decelerates by over 40%. Shift strategies toward momentum-based trailing shorts to match the structural short bias (**0.479**) observed among high-performing accounts.
