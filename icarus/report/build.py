"""Render every output from reports/results.json."""

from __future__ import annotations

import logging

from . import dashboard, figures, readme

log = logging.getLogger(__name__)


def run() -> None:
    done = figures.run()
    log.info("figures: %s", ", ".join(done))
    dashboard.run()
    log.info("dashboard data written")
    readme.run()
    log.info("README results block refreshed")
