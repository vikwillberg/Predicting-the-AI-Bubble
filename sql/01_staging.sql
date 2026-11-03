-- 01 · Staging: typed, de-duplicated raw tables plus daily log returns.
-- {raw} is replaced with the data/raw path by icarus/warehouse/build.py.

CREATE OR REPLACE TABLE prices AS
SELECT DISTINCT ON (ticker, CAST(date AS DATE))
       CAST(date AS DATE)       AS date,
       ticker,
       close,
       adj_close,
       NULLIF(volume, 0)        AS volume,
       split_factor
FROM read_parquet('{raw}/prices.parquet')
WHERE close > 0 AND adj_close > 0
ORDER BY ticker, CAST(date AS DATE);

-- Daily log returns on total-return prices. gap_days flags holes in the data
-- (weekends are 3, a suspended stock might be 30).
CREATE OR REPLACE TABLE returns AS
SELECT date,
       ticker,
       adj_close                                                AS price,
       ln(adj_close / lag(adj_close) OVER w)                    AS r,
       close * volume                                           AS dollar_volume,
       date_diff('day', lag(date) OVER w, date)                 AS gap_days
FROM prices
WINDOW w AS (PARTITION BY ticker ORDER BY date);
