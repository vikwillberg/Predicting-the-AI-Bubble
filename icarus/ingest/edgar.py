"""SEC EDGAR XBRL company facts: the AI capex cycle, straight from 10-Qs and 10-Ks.

Cash-flow items (capex, operating cash flow) are reported year-to-date in 10-Qs,
so quarterly values are derived downstream in SQL (sql/05_edgar_quarterly.sql).

SEC asks automated clients to identify themselves. Set SEC_USER_AGENT
(e.g. "Your Name your@email.com") as a repository variable to comply fully.
"""

from __future__ import annotations

import logging
import os
import time

import pandas as pd

from .http import DEFAULT_UA, session

log = logging.getLogger(__name__)

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
COLUMNS = ["ticker", "cik", "concept_group", "concept", "unit", "start", "end", "val",
           "fy", "fp", "form", "filed", "frame", "accn"]


def _sec_session():
    return session(user_agent=os.environ.get("SEC_USER_AGENT") or DEFAULT_UA)


def ticker_to_cik(sess=None) -> dict[str, int]:
    sess = sess or _sec_session()
    resp = sess.get(TICKER_MAP_URL, timeout=60)
    resp.raise_for_status()
    return {row["ticker"].upper(): int(row["cik_str"]) for row in resp.json().values()}


def parse_companyfacts(payload: dict, ticker: str, concepts: dict[str, list[str]]) -> pd.DataFrame:
    """Flatten the companyfacts JSON for the requested us-gaap concepts."""
    gaap = payload.get("facts", {}).get("us-gaap", {})
    rows = []
    for group, candidates in concepts.items():
        for concept in candidates:
            node = gaap.get(concept)
            if not node:
                continue
            for unit, facts in node.get("units", {}).items():
                if unit != "USD":
                    continue
                for f in facts:
                    rows.append(
                        {
                            "ticker": ticker,
                            "cik": int(payload.get("cik", 0)),
                            "concept_group": group,
                            "concept": concept,
                            "unit": unit,
                            "start": f.get("start"),
                            "end": f.get("end"),
                            "val": f.get("val"),
                            "fy": f.get("fy"),
                            "fp": f.get("fp"),
                            "form": f.get("form"),
                            "filed": f.get("filed"),
                            "frame": f.get("frame"),
                            "accn": f.get("accn"),
                        }
                    )
    df = pd.DataFrame(rows, columns=COLUMNS)
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def fetch_many(tickers: list[str], concepts: dict[str, list[str]]) -> tuple[pd.DataFrame, dict[str, str]]:
    sess = _sec_session()
    failures: dict[str, str] = {}
    try:
        cik_map = ticker_to_cik(sess)
    except Exception as exc:
        return pd.DataFrame(columns=COLUMNS), {"_ticker_map": str(exc)}
    frames = []
    for t in tickers:
        cik = cik_map.get(t.upper())
        if cik is None:
            failures[t] = "ticker not in SEC map (foreign filer or delisted)"
            continue
        try:
            resp = sess.get(FACTS_URL.format(cik=cik), timeout=90)
            resp.raise_for_status()
            df = parse_companyfacts(resp.json(), t, concepts)
            frames.append(df)
            log.info("EDGAR %s (CIK %d): %d facts", t, cik, len(df))
        except Exception as exc:
            failures[t] = str(exc)
            log.error("EDGAR %s failed: %s", t, exc)
        time.sleep(0.25)  # SEC fair-access limit is 10 req/s
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    return out, failures
