-- 04 · Bubble features, computed daily with calendar-time windows (RANGE ...
-- INTERVAL) so that 7-day crypto and 5-day stock markets are treated alike,
-- then sampled at each month end.
--
-- The features follow what Greenwood, Shleifer & You (2019) found predicts
-- whether a run-up ends in a crash: the size of the run-up, its *acceleration*
-- (convexity of the price path), volatility, and turnover.

CREATE OR REPLACE TABLE features_daily AS
WITH base AS (
    SELECT s.date, s.ticker, s.price, s.r, s.dollar_volume,
           -- price roughly 1, 3, 6, 12 and 24 months ago (first price in the window)
           first_value(s.price) OVER (PARTITION BY s.ticker ORDER BY s.date RANGE BETWEEN INTERVAL 31 DAYS PRECEDING AND CURRENT ROW)  AS p_1m,
           first_value(s.price) OVER (PARTITION BY s.ticker ORDER BY s.date RANGE BETWEEN INTERVAL 92 DAYS PRECEDING AND CURRENT ROW)  AS p_3m,
           first_value(s.price) OVER (PARTITION BY s.ticker ORDER BY s.date RANGE BETWEEN INTERVAL 183 DAYS PRECEDING AND CURRENT ROW) AS p_6m,
           first_value(s.price) OVER (PARTITION BY s.ticker ORDER BY s.date RANGE BETWEEN INTERVAL 365 DAYS PRECEDING AND CURRENT ROW) AS p_12m,
           first_value(s.price) OVER (PARTITION BY s.ticker ORDER BY s.date RANGE BETWEEN INTERVAL 730 DAYS PRECEDING AND CURRENT ROW) AS p_24m,
           min(s.price)         OVER (PARTITION BY s.ticker ORDER BY s.date RANGE BETWEEN INTERVAL 730 DAYS PRECEDING AND CURRENT ROW) AS low_24m,
           max(s.price)         OVER (PARTITION BY s.ticker ORDER BY s.date RANGE BETWEEN INTERVAL 365 DAYS PRECEDING AND CURRENT ROW) AS high_12m,
           avg(s.price)         OVER (PARTITION BY s.ticker ORDER BY s.date RANGE BETWEEN INTERVAL 290 DAYS PRECEDING AND CURRENT ROW) AS ma_200d,
           stddev_samp(s.r)     OVER (PARTITION BY s.ticker ORDER BY s.date RANGE BETWEEN INTERVAL 91 DAYS PRECEDING AND CURRENT ROW)  AS sd_3m,
           stddev_samp(s.r)     OVER (PARTITION BY s.ticker ORDER BY s.date RANGE BETWEEN INTERVAL 365 DAYS PRECEDING AND CURRENT ROW) AS sd_12m,
           stddev_samp(s.r)     OVER (PARTITION BY s.ticker ORDER BY s.date RANGE BETWEEN INTERVAL 1826 DAYS PRECEDING AND CURRENT ROW) AS sd_5y,
           count(s.r)           OVER (PARTITION BY s.ticker ORDER BY s.date RANGE BETWEEN INTERVAL 365 DAYS PRECEDING AND CURRENT ROW) AS obs_12m,
           skewness(s.r)        OVER (PARTITION BY s.ticker ORDER BY s.date RANGE BETWEEN INTERVAL 365 DAYS PRECEDING AND CURRENT ROW) AS skew_12m,
           avg(CASE WHEN s.r > 0 THEN 1.0 ELSE 0.0 END)
                                OVER (PARTITION BY s.ticker ORDER BY s.date RANGE BETWEEN INTERVAL 365 DAYS PRECEDING AND CURRENT ROW) AS up_share_12m,
           avg(s.dollar_volume) OVER (PARTITION BY s.ticker ORDER BY s.date RANGE BETWEEN INTERVAL 91 DAYS PRECEDING AND CURRENT ROW)  AS dvol_3m,
           avg(s.dollar_volume) OVER (PARTITION BY s.ticker ORDER BY s.date RANGE BETWEEN INTERVAL 730 DAYS PRECEDING AND CURRENT ROW) AS dvol_24m,
           min(s.date)          OVER (PARTITION BY s.ticker)                                                                          AS first_date
    FROM series s
)
SELECT b.date,
       b.ticker,
       b.price,
       ln(b.price / b.p_1m)                                   AS r_1m,
       ln(b.price / b.p_3m)                                   AS r_3m,
       ln(b.price / b.p_6m)                                   AS r_6m,
       ln(b.price / b.p_12m)                                  AS r_12m,
       ln(b.price / b.p_24m)                                  AS r_24m,
       b.price / b.low_24m - 1                                AS runup_24m,
       -- acceleration: last year's log return minus the year before's
       ln(b.price / b.p_12m) - ln(b.p_12m / b.p_24m)          AS accel_12m,
       b.price / b.high_12m - 1                               AS dd_12m_high,
       ln(b.price / b.ma_200d)                                AS dist_ma200,
       b.sd_3m  * sqrt(b.obs_12m)                             AS vol_3m,
       b.sd_12m * sqrt(b.obs_12m)                             AS vol_12m,
       b.sd_5y  * sqrt(b.obs_12m)                             AS vol_5y,
       b.sd_3m / NULLIF(b.sd_12m, 0)                          AS vol_ratio,
       b.skew_12m,
       b.up_share_12m,
       ln(NULLIF(b.dvol_3m, 0) / NULLIF(b.dvol_24m, 0))       AS turnover_trend,
       date_diff('day', b.first_date, b.date) / 365.25        AS history_years,
       f.ath, f.dd, f.episode_peak_date, f.episode_worst_dd,
       f.next_bear_date, f.next_burst_date, f.last_date
FROM base b
JOIN forward_events f USING (ticker, date);

-- Month-end sample: the last trading day of each month for each series.
CREATE OR REPLACE TABLE features_monthly AS
SELECT * EXCLUDE (rn)
FROM (
    SELECT *,
           row_number() OVER (PARTITION BY ticker, date_trunc('month', date) ORDER BY date DESC) AS rn
    FROM features_daily
)
WHERE rn = 1
ORDER BY ticker, date;
