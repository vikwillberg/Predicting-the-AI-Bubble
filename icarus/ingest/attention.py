"""Attention and narrative data.

* Wikipedia pageviews (Wikimedia REST API, daily since July 2015): how many
  people look up "ChatGPT", "Nvidia" or "Dot-com bubble" is a clean,
  reproducible proxy for retail attention and bubble anxiety.
* GDELT DOC 2.0 (global online news, 2017+): daily share of articles that
  mention "AI bubble".
* multpl.com: the Shiller CAPE ratio, scraped from its HTML table.
"""

from __future__ import annotations

import io
import logging
import time
from urllib.parse import quote

import pandas as pd

from .http import session

log = logging.getLogger(__name__)

WIKI_URL = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "en.wikipedia.org/all-access/user/{article}/daily/{start}/{end}"
)
GDELT_URL = (
    "https://api.gdeltproject.org/api/v2/doc/doc?query={q}&mode=timelinevolraw"
    "&format=json&startdatetime={start}000000&enddatetime={end}235959&timelinesmooth=0"
)
MULTPL_URL = "https://www.multpl.com/{slug}/table/by-month"


# --------------------------------------------------------------------------- wiki
def parse_pageviews(payload: dict, article: str) -> pd.DataFrame:
    items = payload.get("items", [])
    df = pd.DataFrame(
        {
            "date": pd.to_datetime([i["timestamp"][:8] for i in items], format="%Y%m%d"),
            "article": article,
            "views": [int(i["views"]) for i in items],
        }
    )
    return df


def fetch_pageviews(articles: list[str], start: str = "20150701") -> tuple[pd.DataFrame, dict[str, str]]:
    sess = session()
    end = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d")
    frames, failures = [], {}
    for a in articles:
        try:
            url = WIKI_URL.format(article=quote(a, safe=""), start=start, end=end)
            resp = sess.get(url, timeout=60)
            if resp.status_code == 404:
                failures[a] = "no pageview data (article missing)"
                continue
            resp.raise_for_status()
            df = parse_pageviews(resp.json(), a)
            frames.append(df)
            log.info("wiki %s: %d days", a, len(df))
        except Exception as exc:
            failures[a] = str(exc)
            log.error("wiki %s failed: %s", a, exc)
        time.sleep(0.2)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["date", "article", "views"])
    return out, failures


# -------------------------------------------------------------------------- gdelt
def parse_gdelt_timeline(payload: dict, key: str) -> pd.DataFrame:
    rows = []
    for series in payload.get("timeline", []):
        for point in series.get("data", []):
            rows.append(
                {
                    "date": pd.to_datetime(point["date"][:8], format="%Y%m%d"),
                    "query_key": key,
                    "articles": float(point.get("value", 0.0)),
                    "total_articles": float(point.get("norm", 0.0)) or None,
                }
            )
    return pd.DataFrame(rows, columns=["date", "query_key", "articles", "total_articles"])


def fetch_gdelt(queries: dict[str, str], start: str = "20170101") -> tuple[pd.DataFrame, dict[str, str]]:
    sess = session(total_retries=3)
    end = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d")
    frames, failures = [], {}
    for key, q in queries.items():
        try:
            url = GDELT_URL.format(q=quote(q), start=start, end=end)
            resp = sess.get(url, timeout=120)
            resp.raise_for_status()
            df = parse_gdelt_timeline(resp.json(), key)
            if df.empty:
                raise RuntimeError("empty timeline")
            frames.append(df)
            log.info("gdelt %s: %d points", key, len(df))
        except Exception as exc:
            failures[key] = str(exc)
            log.error("gdelt %s failed: %s", key, exc)
        time.sleep(6)  # GDELT asks for at most one request every 5 seconds
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["date", "query_key", "articles", "total_articles"])
    return out, failures


# ------------------------------------------------------------------------- multpl
def parse_multpl_table(html: str, metric: str) -> pd.DataFrame:
    tables = pd.read_html(io.StringIO(html))
    if not tables:
        raise RuntimeError("no table found")
    t = tables[0]
    t.columns = ["date", "value"]
    # Values look like "41.07" or "41.07 estimate"; keep the number only.
    t["value"] = pd.to_numeric(t["value"].astype(str).str.extract(r"([-+]?\d*\.?\d+)")[0], errors="coerce")
    t["date"] = pd.to_datetime(t["date"], errors="coerce")
    t["metric"] = metric
    return t.dropna(subset=["date", "value"]).sort_values("date")[["date", "metric", "value"]]


def fetch_multpl(slugs: dict[str, str] | None = None) -> tuple[pd.DataFrame, dict[str, str]]:
    slugs = slugs or {"cape": "shiller-pe", "pe": "s-p-500-pe-ratio"}
    sess = session()
    frames, failures = [], {}
    for metric, slug in slugs.items():
        try:
            resp = sess.get(MULTPL_URL.format(slug=slug), timeout=60)
            resp.raise_for_status()
            frames.append(parse_multpl_table(resp.text, metric))
            log.info("multpl %s: %d months", metric, len(frames[-1]))
        except Exception as exc:
            failures[metric] = str(exc)
            log.error("multpl %s failed: %s", metric, exc)
        time.sleep(1.0)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["date", "metric", "value"])
    return out, failures
