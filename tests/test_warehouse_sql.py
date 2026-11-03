"""The SQL layer, tested on hand-built series whose answers are known."""

import duckdb
import pandas as pd
import pytest

from icarus import config
from icarus.warehouse.build import render


def _run(con, name):
    con.execute(render((config.SQL_DIR / name).read_text()))


@pytest.fixture
def con():
    c = duckdb.connect()
    yield c
    c.close()


def test_drawdown_events_and_rebound_trap(con):
    # 100 -> 130 (peak) -> 91 (-30%, bear) -> 110 (bounce, still below peak) -> 75 (-42%, burst) -> 140 (new high)
    prices = [100, 130, 91, 110, 75, 140]
    dates = pd.bdate_range("2020-01-01", periods=len(prices))
    con.register("s", pd.DataFrame({"date": dates, "ticker": "X", "price": prices, "dollar_volume": 1.0, "r": 0.0}))
    con.execute("CREATE TABLE series AS SELECT CAST(date AS DATE) date, ticker, price, dollar_volume, r FROM s")
    _run(con, "03_drawdowns_events.sql")
    d = con.execute("SELECT * FROM drawdowns ORDER BY date").df()
    assert d["dd"].round(2).tolist() == [0.0, 0.0, -0.3, -0.15, -0.42, 0.0]
    # The bounce to -15% must remember that this episode already fell 30%.
    assert d["episode_worst_dd"].round(2).tolist() == [0.0, 0.0, -0.3, -0.3, -0.42, 0.0]
    ev = con.execute("SELECT * FROM events WHERE max_drawdown < -0.2").df()
    assert len(ev) == 1
    assert pd.Timestamp(ev["bear_date"].iloc[0]) == dates[2]
    assert pd.Timestamp(ev["burst_date"].iloc[0]) == dates[4]
    fw = con.execute("SELECT date, next_bear_date, next_burst_date FROM forward_events ORDER BY date").df()
    assert pd.Timestamp(fw["next_burst_date"].iloc[0]) == dates[4]   # from day 0 the burst lies ahead
    assert pd.isna(fw["next_bear_date"].iloc[3])                  # on the bounce, no *new* bear is ahead


def test_edgar_ytd_cash_flows_become_quarters(con):
    rows = []
    for end, val in [("2026-03-31", 30.0), ("2026-06-30", 70.0), ("2026-09-30", 120.0), ("2026-12-31", 180.0)]:
        rows.append({"ticker": "MSFT", "cik": 1, "concept_group": "capex",
                     "concept": "PaymentsToAcquirePropertyPlantAndEquipment", "unit": "USD",
                     "start": pd.Timestamp("2026-01-01"), "end": pd.Timestamp(end), "val": val, "fy": 2026,
                     "fp": "Q", "form": "10-Q", "filed": pd.Timestamp(end) + pd.Timedelta(days=30), "frame": None,
                     "accn": "a"})
    con.register("e", pd.DataFrame(rows))
    con.execute("CREATE TABLE edgar_raw AS SELECT * FROM e")
    con.execute("CREATE TABLE yahoo_fund_raw (ticker VARCHAR, period_end TIMESTAMP, freq VARCHAR, item VARCHAR, "
                "value DOUBLE, source VARCHAR)")
    _run(con, "06_fundamentals.sql")
    q = con.execute("SELECT quarter_end, value FROM edgar_quarterly ORDER BY quarter_end").df()
    assert q["value"].tolist() == [30.0, 40.0, 50.0, 60.0]       # Q1, 6M-3M, 9M-6M, FY-9M
