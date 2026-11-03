-- 07 · Market structure: breadth inside the AI trade and concentration of the
-- wider market. Late-stage bubbles narrow: a few generals keep making highs
-- while the troops are already in bear markets (1999-2000 was the textbook case).

CREATE OR REPLACE TABLE ai20_breadth AS
SELECT date,
       count(*)                                                        AS n_members,
       avg(CASE WHEN dd <= -0.20 THEN 1.0 ELSE 0.0 END)                AS share_in_bear,
       avg(CASE WHEN dd <= -0.40 THEN 1.0 ELSE 0.0 END)                AS share_burst,
       avg(CASE WHEN dist_ma200 > 0 THEN 1.0 ELSE 0.0 END)             AS share_above_ma200,
       median(dd)                                                      AS median_dd_from_ath,
       median(dd_12m_high)                                             AS median_dd_12m_high
FROM features_daily
WHERE ticker IN (SELECT ticker FROM ai20_members)
GROUP BY date
HAVING count(*) >= 5
ORDER BY date;

-- Equal-weight vs cap-weight S&P 500 (RSP / SPY, total return). A falling ratio
-- means a handful of mega-caps are carrying the index.
CREATE OR REPLACE TABLE concentration AS
SELECT r.date,
       r.price / s.price                                               AS rsp_spy,
       r.price / s.price / first_value(r.price / s.price) OVER (ORDER BY r.date) AS rsp_spy_rebased
FROM returns r
JOIN returns s ON s.date = r.date AND s.ticker = 'SPY'
WHERE r.ticker = 'RSP'
ORDER BY r.date;
