-- 02 · ICARUS AI-20 index: equal-weight, rebalanced daily, total return.
-- Before every member existed (CoreWeave listed in 2025, Arm in 2023) the index
-- holds whichever members were trading, so its pre-2016 history is effectively a
-- "large-cap tech and semis" basket. That history includes the same companies'
-- 2000-02 collapse.

CREATE OR REPLACE TABLE ai20_daily AS
WITH member_returns AS (
    SELECT r.date,
           r.ticker,
           -- Winsorise single-day moves at +/-75% to stop a data glitch from
           -- moving the whole index.
           greatest(least(exp(r.r) - 1, 0.75), -0.75) AS ret,
           r.dollar_volume
    FROM returns r
    JOIN ai20_members m USING (ticker)
    WHERE r.r IS NOT NULL
      AND r.gap_days <= 10
),
daily AS (
    SELECT date,
           avg(ret)            AS ret,
           count(*)            AS n_members,
           sum(dollar_volume)  AS dollar_volume
    FROM member_returns
    GROUP BY date
    HAVING count(*) >= 3
)
SELECT date,
       ret,
       n_members,
       dollar_volume,
       100 * exp(sum(ln(1 + ret)) OVER (ORDER BY date ROWS UNBOUNDED PRECEDING)) AS level
FROM daily
ORDER BY date;

-- One unified daily series table: every scraped ticker plus the AI-20.
CREATE OR REPLACE TABLE series AS
SELECT date, ticker, price, dollar_volume, r
FROM returns
UNION ALL
SELECT date,
       'AI20'                                                    AS ticker,
       level                                                     AS price,
       dollar_volume,
       ln(level / lag(level) OVER (ORDER BY date))               AS r
FROM ai20_daily;
