# ICARUS methodology

This document explains every modelling choice, in the order the pipeline runs.

## 1. Data

| Source | What | Coverage | Module |
|---|---|---|---|
| Yahoo Finance | Daily OHLCV, split and dividend adjusted; valuation snapshot; quarterly and annual statements | 102 tickers, back to 1927 | `icarus/ingest/prices.py` |
| FRED | 16 series: Baa and Aaa spreads, high-yield OAS, yield curve, real yields, NFCI, VIX, M2, Fed balance sheet, recessions | 1854+ | `icarus/ingest/fred.py` |
| SEC EDGAR XBRL | Capex, operating cash flow, revenue, net income, depreciation | 2009+ (needs `SEC_USER_AGENT`) | `icarus/ingest/edgar.py` |
| Wikimedia | Daily pageviews for 12 articles (ChatGPT, Nvidia, Dot-com bubble, AI bubble ...) | July 2015+ | `icarus/ingest/attention.py` |
| GDELT DOC 2.0 | Daily news volume for "AI bubble", "dot-com bubble", "AI spending" | 2017+ | `icarus/ingest/attention.py` |
| multpl | Shiller CAPE and S&P 500 P/E (HTML table scrape) | 1871+ | `icarus/ingest/attention.py` |
| Press / filings | Anthropic and OpenAI funding rounds, run-rates, S-1 figures, mega-IPOs | 1999+ | `config/private_marks.yaml` (curated, sourced) |

**Quality gate** (`icarus/validate.py`): non-empty, unique (ticker, date), positive prices, no future dates,
no impossible index moves, and eight famous closes matched to within 0.5%. Intraday bars are dropped when a
scrape runs before the US close.

## 2. The AI-20 index

Equal-weight, rebalanced daily, total return, built in SQL (`sql/02_ai20_index.sql`). Members enter when
they list (CoreWeave 2025, Arm 2023). Single-day member moves are winsorised at +/-75% to stop a data glitch
from moving the index. The basket is chosen with 2026 hindsight, so it is used only for scoring, never for
training, and its pre-2016 history is effectively a large-cap tech and semiconductor basket.

## 3. Crash labels (SQL)

A drawdown *episode* starts at every new all-time high. Within an episode, the *bear* event is the first day
the price is 20% below the episode peak and the *burst* event the first day it is 40% below.
For each day, a forward window function finds the next event strictly after it.

A sample is eligible for event E only if its current episode has **not yet** crossed E's threshold
(`episode_worst_dd > -threshold`). Without this rule, an asset that fell 30% and recovered to -19% looks
"not yet in a bear market", which is wrong and badly biased the first version of the model.

## 4. Features (SQL, calendar-time windows)

Returns over 1, 3, 6, 12 and 24 months; run-up from the two-year low; acceleration (last year's log return
minus the year before's); the same two in volatility units; drawdown from the all-time high and the 12-month
high; worst drawdown in the current episode; distance from the 200-day average; 3-month, 12-month and 5-year
volatility and their ratio; 12-month skewness and share of up days; dollar-volume trend; and the PSY
explosiveness of the last 10 years (BSADF minus its 95% critical value).

These follow Greenwood, Shleifer & You (2019): a large run-up alone barely predicts a crash, but its
acceleration, volatility and turnover do.

## 5. Survival ensemble ("when")

The panel holds month-end samples for 67 assets (indices in 9 countries, commodities, crypto, sector and
thematic ETFs, and 40+ stocks that lived through manias). Each sample is stacked for horizons
k = 1, 2, 3, 6, 9, 12, 18 and 24 months with label "event within k months", keeping only labels that are
known (the event happened, or k months of future data exist): discrete-time survival with administrative
censoring.

Two models:
* LightGBM with a **monotone-increasing constraint on k**, so each predicted curve is a valid CDF.
* L2-regularised logistic regression on standardised features plus log k.

Blend: `p = w * p_gbm + (1 - w) * p_logit`, with w chosen from {0, 0.25, 0.5, 0.75, 1} by out-of-sample log
loss. In validation the weight for year Y uses only years before Y.

**Walk-forward validation.** For each test year Y from 1995, train on samples whose label window ended before
1 January Y (purging), predict year Y. Metrics: AUC, Brier score, Brier skill vs the training base rate,
log loss, a reliability curve, and per-episode checks on famous bubbles.

**Uncertainty.** A year-block bootstrap (resample calendar years, refit, predict) gives the 80% band. Crashes
cluster in time, so resampling whole years respects that dependence.

## 6. PSY explosive-root test ("is it a bubble?")

BSADF(t) is the largest right-tailed ADF statistic over all windows ending at t. Each window's regression
uses prefix sums of the cross-products, so every window costs O(1) and a full sequence costs O(T^2) batched
3x3 solves. Critical values come from 999 Monte Carlo random walks with asymptotically negligible drift.
Dating follows the two-step rule: the global GSADF test must reject first, and an episode must last at least
log(T) periods. On pure random walks this keeps false episodes near the nominal rate, whereas dating every
date on its own flags a short "bubble" in about 4 of 10 random walks.

## 7. LPPLS bubble clock ("when, from the price path alone")

ln p(t) = A + B f + C1 f cos(w ln|tc - t|) + C2 f sin(w ln|tc - t|), f = |tc - t|^m.

The four linear parameters are solved exactly for each (tc, m, w), and the search runs a 12x8x12 grid followed
by bounded Nelder-Mead from the three best grid points. Windows run from 125 to 750 trading days in steps
of 25. A fit *qualifies* if m is in [0.01, 0.99], w in [2, 25], tc within [-5%, +10%] of the window length
from its end, at least 2.5 oscillations, and damping of at least 0.5 (Sornette et al. 2015). The confidence
indicator is the share of windows with a qualified positive-bubble fit.

The *ex-ante* test only uses fits made strictly before the observed peak. On seven historical bubbles the
critical time came 3-11 weeks early (median 57 days), a known property of LPPLS that the report states next to
every forecast.

## 8. HMM regimes

A 3-state, full-covariance Gaussian HMM on weekly Nasdaq-100 log returns and log realised volatility since
1985, best of 12 restarts. States are named by mean volatility. The fitted model decodes the AI-20.
Validation: stress covers 96% of 2000-02 bust weeks, 93% of GFC weeks and 0% of the calm 2013-14 bull.

## 9. Historical analogs

The AI-20's last 104 weekly log prices are z-normalised and matched against 12 bubbles at 8 time-stretch
factors (0.6x to 2x) over each bubble's window from 4 years before to 2 years after its peak. The best match
per bubble gives "today's equivalent date", months to that bubble's peak, and the drawdown that followed.
Weights are 1 / RMSE^2. Because every analog is a bubble, the output is conditional: *if* this is a bubble,
where are we in it?

## 10. Fundamentals

* **Capex strain:** sum of capex and operating cash flow for Microsoft, Alphabet, Amazon, Meta and Oracle per
  calendar quarter (fiscal quarters mapped to the calendar quarter holding most of their days); quarters with
  missing filings are shown as gaps, never interpolated.
* **AI revenue gap (Cahn 2024):** NVIDIA quarterly revenue x 4 x data-center share (0.89) x 2 (GPUs are about
  half of data-center cost) x 2 (end customers need ~50% gross margin) = end-user AI revenue required.
* **Anthropic reverse DCF:** value = R_N x FCF margin x exit multiple x (1 - dilution)^N / (1 + r)^N, solved for
  R_N with N = 10 and 20,000 draws of R0 ~ U(50, 70) $B, r ~ U(9%, 13%), margin ~ U(20%, 35%),
  multiple ~ U(18, 30) and dilution ~ U(1%, 3%).

## 11. Bubble Risk Index

Seven pillars, each scored 0-100 (a percentile against its own history where one exists):
price dynamics, valuation, fundamentals, market structure, credit and macro, AI hype, and mega-IPO
issuance. Fear signals (crash searches, bubble news) are reported as context but not scored, because they
cut both ways. The composite uses the weights in `config/settings.yaml` and is also reported across 5,000
Dirichlet-random weightings. A point-in-time version for the Nasdaq-100 using only expanding-window data read
98 at the 2000 peak and 84 in 2021.

## 12. Verdict rules

* **Is it a bubble?** Yes only if the risk index is at least 70 **and** there is statistical evidence of
  explosiveness in the AI era (a PSY episode since 2023 on an AI-linked series, or qualified LPPLS fits
  before the peak).
* **Stage:** burst (already -40%), deflating (stress regime and at least 15% below the peak), wobbling, or
  inflating.
* **When:** the survival CDF; the most likely month is the largest monthly increase in burst probability, and
  the conditional median is the month by which half of the 24-month burst probability has arrived.
