# Model card: ICARUS survival ensemble

| | |
|---|---|
| **Task** | Probability that an asset falls 20% (bear) or 40% (burst) below its running peak within k = 1..24 months |
| **Model** | Blend of LightGBM (450 trees, 15 leaves, min 300 samples per leaf, monotone in horizon) and L2 logistic regression; blend weight learned walk-forward |
| **Training data** | Month-end samples from 67 assets, 1927-2026: equity indices in 9 countries, commodities, crypto, ETFs, 40+ individual stocks |
| **Features** | 21 price, volatility, volume and explosiveness features (see `icarus/models/panel.py`); no macro inputs, so the model transfers across countries and asset classes |
| **Evaluation** | Walk-forward by calendar year from 1995 with purging of overlapping labels; AUC, Brier skill vs base rate, log loss, reliability |
| **Intended use** | Research and education: quantifying crash risk for broad baskets such as the AI-20, Nasdaq-100 or S&P 500 |
| **Not for** | Trading decisions, single-name stock picking, or timing to the day |

## Performance (walk-forward, out of sample)

The current figures live in `reports/results.json` (`survival.burst.metrics`) and in the README's
auto-generated table. At launch, 12-month burst AUC was about 0.78 with Brier skill about +22% over the base
rate, and the reliability curve stayed within about two percentage points of the diagonal over most of the range.

## Known limitations and biases

* **Survivorship:** Yahoo has no delisted stocks, so dead dot-com names are missing; single-stock crash rates
  are biased down.
* **Dependence:** panel samples overlap in time and assets crash together (2000, 2008), so effective sample
  size is near the number of distinct episodes, not the number of rows. The year-block bootstrap reflects
  this in the intervals.
* **Regime change:** markets with circuit breakers, central-bank backstops and passive flows may crash
  differently than in the 1930s or 1980s.
* **Target construction:** the AI-20 is picked with hindsight; it is scored, never trained on.
* **Thresholds:** 20% and 40% are conventions. A 35% fall is not a "miss" in any economic sense.
* **Misses:** the model did not flag the Shanghai 2007 top (9% at the peak against an 18% base rate).

## Ethical considerations

Probabilities of crashes can be misread as predictions or advice. Every public output states the
probability, its uncertainty band, and that it is not investment advice.
