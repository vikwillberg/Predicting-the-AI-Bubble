-- 05 · Macro, valuation and attention series at monthly frequency.

CREATE OR REPLACE TABLE fred AS
SELECT CAST(date AS DATE) AS date, series_id, value
FROM read_parquet('{raw}/fred.parquet');

-- Last observation in each month, pivoted wide.
CREATE OR REPLACE TABLE macro_monthly AS
PIVOT (
    SELECT last_day(date) AS month, series_id, arg_max(value, date) AS value
    FROM fred
    GROUP BY ALL
)
ON series_id USING any_value(value)
ORDER BY month;

CREATE OR REPLACE TABLE cape_monthly AS
SELECT last_day(CAST(date AS DATE)) AS month,
       arg_max(value, date) FILTER (WHERE metric = 'cape') AS cape,
       arg_max(value, date) FILTER (WHERE metric = 'pe')   AS pe
FROM read_parquet('{raw}/multpl.parquet')
GROUP BY ALL
ORDER BY month;

CREATE OR REPLACE TABLE wiki_monthly AS
SELECT last_day(CAST(date AS DATE)) AS month, article, sum(views) AS views, count(*) AS days
FROM read_parquet('{raw}/wiki_pageviews.parquet')
GROUP BY ALL
ORDER BY article, month;

CREATE OR REPLACE TABLE gdelt_monthly AS
SELECT last_day(CAST(date AS DATE)) AS month,
       query_key,
       sum(articles)                                         AS articles,
       sum(articles) / NULLIF(sum(total_articles), 0) * 1e6  AS per_million_articles
FROM read_parquet('{raw}/gdelt.parquet')
GROUP BY ALL
ORDER BY query_key, month;
