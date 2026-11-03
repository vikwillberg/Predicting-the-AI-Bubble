-- 03 · Drawdowns and "burst" events.
--
-- A drawdown episode starts at every new all-time high. Within an episode:
--   bear event  = first day the price is >= {bear}  below the episode's peak
--   burst event = first day the price is >= {burst} below the episode's peak
-- For every day t we then look *forward* for the next event: that becomes the
-- label (event within k months?) and the survival time.

CREATE OR REPLACE TABLE drawdowns AS
WITH running AS (
    SELECT date, ticker, price,
           max(price) OVER (PARTITION BY ticker ORDER BY date ROWS UNBOUNDED PRECEDING) AS ath
    FROM series
),
episodes AS (
    SELECT *,
           price / ath - 1 AS dd,
           sum(CASE WHEN price >= ath THEN 1 ELSE 0 END)
               OVER (PARTITION BY ticker ORDER BY date ROWS UNBOUNDED PRECEDING) AS episode_id
    FROM running
)
SELECT *,
       min(date) OVER (PARTITION BY ticker, episode_id) AS episode_peak_date,
       -- Worst drawdown so far in this episode. Without it, an asset that fell
       -- 30%, bounced back to -19% and has not made a new high would look like
       -- it had "not yet" entered a bear market. It already has.
       min(dd) OVER (PARTITION BY ticker, episode_id ORDER BY date ROWS UNBOUNDED PRECEDING) AS episode_worst_dd
FROM episodes;

CREATE OR REPLACE TABLE events AS
SELECT ticker,
       episode_id,
       any_value(episode_peak_date)                                        AS peak_date,
       min(date) FILTER (WHERE dd <= -{bear})                              AS bear_date,
       min(date) FILTER (WHERE dd <= -{burst})                             AS burst_date,
       min(dd)                                                             AS max_drawdown
FROM drawdowns
GROUP BY ticker, episode_id;

-- Next bear / burst event strictly after each date, plus the last observed date
-- (needed to know whether a label is censored).
CREATE OR REPLACE TABLE forward_events AS
WITH flagged AS (
    SELECT d.date, d.ticker, d.price, d.ath, d.dd, d.episode_peak_date, d.episode_worst_dd,
           CASE WHEN d.date = e.bear_date  THEN d.date END AS bear_event,
           CASE WHEN d.date = e.burst_date THEN d.date END AS burst_event
    FROM drawdowns d
    JOIN events e USING (ticker, episode_id)
)
SELECT date, ticker, price, ath, dd, episode_peak_date, episode_worst_dd,
       min(bear_event)  OVER fwd                                   AS next_bear_date,
       min(burst_event) OVER fwd                                   AS next_burst_date,
       max(date) OVER (PARTITION BY ticker)                        AS last_date
FROM flagged
WINDOW fwd AS (PARTITION BY ticker ORDER BY date ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING);
