-- 06 · The AI capex cycle from company filings.
--
-- 10-Q cash-flow statements are year-to-date (Q2 shows six months, Q3 nine), so
-- discrete quarters are rebuilt by differencing within each fiscal year:
--   Q1 = 3M,  Q2 = 6M - 3M,  Q3 = 9M - 6M,  Q4 = FY - 9M.
-- Source priority: SEC EDGAR XBRL (long history) > Yahoo statements (fallback).

-- Pick one concept per (company, metric): the one reported most recently.
CREATE OR REPLACE TABLE edgar_concept_choice AS
SELECT ticker, concept_group, arg_max(concept, last_end) AS concept
FROM (
    SELECT ticker, concept_group, concept, max("end") AS last_end, count(*) AS n
    FROM edgar_raw
    GROUP BY ALL
)
GROUP BY ALL;

CREATE OR REPLACE TABLE edgar_quarterly AS
WITH facts AS (
    -- Latest filing wins when a period is restated.
    SELECT DISTINCT ON (e.ticker, e.concept_group, e.start, e."end")
           e.ticker, e.concept_group, CAST(e.start AS DATE) AS start, CAST(e."end" AS DATE) AS end_date, e.val
    FROM edgar_raw e
    JOIN edgar_concept_choice c USING (ticker, concept_group, concept)
    WHERE e.start IS NOT NULL
      AND date_diff('day', CAST(e.start AS DATE), CAST(e."end" AS DATE)) BETWEEN 80 AND 380
    ORDER BY e.ticker, e.concept_group, e.start, e."end", e.filed DESC
),
chained AS (
    SELECT ticker, concept_group, start, end_date,
           val - coalesce(lag(val) OVER w, 0)                                 AS q_val,
           date_diff('day', coalesce(lag(end_date) OVER w, start), end_date)  AS q_days
    FROM facts
    WINDOW w AS (PARTITION BY ticker, concept_group, start ORDER BY end_date)
)
SELECT DISTINCT ON (ticker, concept_group, end_date)
       ticker, concept_group, end_date AS quarter_end, q_val AS value, 'edgar' AS source
FROM chained
WHERE q_days BETWEEN 80 AND 100
ORDER BY ticker, concept_group, end_date;

CREATE OR REPLACE TABLE yahoo_quarterly AS
SELECT ticker,
       CASE item
           WHEN 'CapitalExpenditure'                THEN 'capex'
           WHEN 'OperatingCashFlow'                 THEN 'ocf'
           WHEN 'TotalRevenue'                      THEN 'revenue'
           WHEN 'NetIncome'                         THEN 'net_income'
           WHEN 'DepreciationAndAmortization'       THEN 'depreciation'
           WHEN 'DepreciationAmortizationDepletion' THEN 'depreciation'
       END                                                      AS concept_group,
       CAST(period_end AS DATE)                                 AS quarter_end,
       CASE WHEN item = 'CapitalExpenditure' THEN -value ELSE value END AS value,
       'yahoo'                                                  AS source
FROM yahoo_fund_raw
WHERE freq = 'quarterly'
  AND item IN ('CapitalExpenditure', 'OperatingCashFlow', 'TotalRevenue', 'NetIncome',
               'DepreciationAndAmortization', 'DepreciationAmortizationDepletion');

-- Union with EDGAR first; a Yahoo quarter is used only when EDGAR lacks a
-- quarter ending within 10 days of it.
CREATE OR REPLACE TABLE fundamentals_quarterly AS
WITH unioned AS (
    SELECT * FROM edgar_quarterly
    UNION ALL
    SELECT y.* FROM yahoo_quarterly y
    WHERE y.concept_group IS NOT NULL
      AND NOT EXISTS (
        SELECT 1 FROM edgar_quarterly e
        WHERE e.ticker = y.ticker AND e.concept_group = y.concept_group
          AND abs(date_diff('day', e.quarter_end, y.quarter_end)) <= 10)
)
SELECT DISTINCT ON (ticker, concept_group, quarter_end) *,
       -- Calendar quarter containing most of the fiscal quarter.
       date_trunc('quarter', quarter_end - INTERVAL 45 DAY) AS cal_quarter
FROM unioned
ORDER BY ticker, concept_group, quarter_end, source;

-- Hyperscaler capex: the dollars that pay for the AI build-out.
CREATE OR REPLACE TABLE hyperscaler_capex AS
WITH q AS (
    SELECT cal_quarter,
           sum(value) FILTER (WHERE concept_group = 'capex')       AS capex,
           sum(value) FILTER (WHERE concept_group = 'ocf')         AS ocf,
           sum(value) FILTER (WHERE concept_group = 'revenue')     AS revenue,
           count(DISTINCT ticker) FILTER (WHERE concept_group = 'capex') AS n_companies
    FROM fundamentals_quarterly
    WHERE ticker IN ('MSFT', 'GOOGL', 'AMZN', 'META', 'ORCL')
    GROUP BY cal_quarter
)
SELECT *,
       sum(capex) OVER w4 AS capex_ttm,
       sum(ocf)   OVER w4 AS ocf_ttm,
       sum(capex) OVER w4 / NULLIF(sum(ocf) OVER w4, 0)   AS capex_to_ocf_ttm,
       capex / NULLIF(lag(capex, 4) OVER (ORDER BY cal_quarter), 0) - 1 AS capex_yoy
FROM q
WINDOW w4 AS (ORDER BY cal_quarter ROWS BETWEEN 3 PRECEDING AND CURRENT ROW)
ORDER BY cal_quarter;
