"""Shared HTTP session: retries with exponential backoff and a polite User-Agent."""

from __future__ import annotations

import os

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_UA = (
    "ICARUS-AI-Bubble-Research/1.0 "
    "(+https://github.com/vikwillberg/Predicting-the-AI-Bubble)"
)


def session(user_agent: str | None = None, total_retries: int = 5) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=total_retries,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update(
        {
            "User-Agent": user_agent or os.environ.get("ICARUS_USER_AGENT", DEFAULT_UA),
            "Accept-Encoding": "gzip, deflate",
        }
    )
    return s
