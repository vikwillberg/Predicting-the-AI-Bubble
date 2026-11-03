"""Parser tests on small hand-written payloads shaped like each source's real response."""

import pandas as pd

from icarus.ingest import attention, edgar, fred, prices


def test_fred_csv_handles_missing_dots_and_both_header_styles():
    new_style = "observation_date,BAA10Y\n2026-10-01,1.62\n2026-10-02,.\n2026-10-05,1.70\n"
    old_style = "DATE,BAA10Y\n2026-10-01,1.62\n"
    df = fred.parse_fred_csv(new_style, "BAA10Y")
    assert list(df["value"]) == [1.62, 1.70]
    assert df["date"].iloc[-1] == pd.Timestamp("2026-10-05")
    assert len(fred.parse_fred_csv(old_style, "BAA10Y")) == 1


def test_yahoo_normaliser_tz_splits_and_duplicates():
    idx = pd.DatetimeIndex(["2024-06-07 00:00", "2024-06-10 00:00", "2024-06-10 00:00"],
                           tz="America/New_York")
    hist = pd.DataFrame({"Open": [1, 2, 2], "High": [1, 2, 2], "Low": [1, 2, 2],
                         "Close": [1200.0, 121.0, 121.5], "Adj Close": [1199.0, 120.9, 121.4],
                         "Volume": [10, 20, 21], "Stock Splits": [0.0, 10.0, 10.0]}, index=idx)
    df = prices._normalise_yahoo(hist, "NVDA")
    assert len(df) == 2                                 # duplicate session dropped
    assert df["date"].dt.tz is None
    assert df["date"].iloc[1] == pd.Timestamp("2024-06-10")
    assert list(df["split_factor"]) == [1.0, 10.0]      # 0 means "no split"
    assert df["close"].iloc[1] == 121.5                 # last duplicate wins


def test_companyfacts_flattening_keeps_usd_only():
    payload = {"cik": 1045810, "facts": {"us-gaap": {
        "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": [
            {"start": "2026-01-26", "end": "2026-04-26", "val": 1_000, "fy": 2027, "fp": "Q1",
             "form": "10-Q", "filed": "2026-05-28", "accn": "x"}]}},
        "Revenues": {"units": {"shares": [{"end": "2026-04-26", "val": 5}]}}}}}
    df = edgar.parse_companyfacts(payload, "NVDA", {"capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
                                                    "revenue": ["Revenues"]})
    assert len(df) == 1
    assert df["concept_group"].iloc[0] == "capex"
    assert df["start"].iloc[0] == pd.Timestamp("2026-01-26")


def test_pageviews_and_gdelt_parsers():
    pv = attention.parse_pageviews({"items": [{"timestamp": "2026110100", "views": 1234},
                                              {"timestamp": "2026110200", "views": 99}]}, "AI_bubble")
    assert pv["views"].sum() == 1333 and pv["date"].iloc[0] == pd.Timestamp("2026-11-01")
    gd = attention.parse_gdelt_timeline({"timeline": [{"series": "Article Count", "data": [
        {"date": "20261101T000000Z", "value": 42, "norm": 100000}]}]}, "ai_bubble")
    assert gd["articles"].iloc[0] == 42 and gd["total_articles"].iloc[0] == 100000


def test_multpl_table_parser():
    html = """<table id="datatable"><tr><th>Date</th><th>Value</th></tr>
              <tr><td>Oct 1, 2026</td><td>&#x2002;41.07</td></tr>
              <tr><td>Sep 1, 2026</td><td>40.50 estimate</td></tr></table>"""
    df = attention.parse_multpl_table(html, "cape")
    assert list(df["value"]) == [40.50, 41.07]          # sorted ascending by date
