"""Build the DuckDB warehouse by running sql/*.sql in order."""

from __future__ import annotations

import logging
import time

import duckdb
import pandas as pd
import pyarrow.parquet as pq

from .. import config

log = logging.getLogger(__name__)

EDGAR_COLS = {"ticker": "VARCHAR", "cik": "BIGINT", "concept_group": "VARCHAR", "concept": "VARCHAR",
              "unit": "VARCHAR", "start": "TIMESTAMP", "end": "TIMESTAMP", "val": "DOUBLE", "fy": "BIGINT",
              "fp": "VARCHAR", "form": "VARCHAR", "filed": "TIMESTAMP", "frame": "VARCHAR", "accn": "VARCHAR"}
YAHOO_FUND_COLS = {"ticker": "VARCHAR", "period_end": "TIMESTAMP", "freq": "VARCHAR", "item": "VARCHAR",
                   "value": "DOUBLE", "source": "VARCHAR"}


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    config.WAREHOUSE.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(config.WAREHOUSE), read_only=read_only)


def _load_optional(con, table: str, parquet: str, cols: dict[str, str]) -> None:
    """Create `table` from a parquet file, or as an empty typed table if the source
    was unavailable (so downstream SQL never breaks)."""
    path = config.DATA_RAW / parquet
    if path.exists() and pq.ParquetFile(path).metadata.num_rows > 0:
        con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM read_parquet('{path}')")
    else:
        ddl = ", ".join(f'"{c}" {t}' for c, t in cols.items())
        con.execute(f"CREATE OR REPLACE TABLE {table} ({ddl})")
        log.warning("%s unavailable; created empty %s", parquet, table)


def render(sql: str) -> str:
    ev = config.settings()["events"]
    return (sql.replace("{raw}", str(config.DATA_RAW))
               .replace("{bear}", str(ev["bear_drawdown"]))
               .replace("{burst}", str(ev["burst_drawdown"])))


def run() -> None:
    config.ensure_dirs()
    if config.WAREHOUSE.exists():
        config.WAREHOUSE.unlink()
    con = connect()
    members = pd.DataFrame(config.universe()["ai20"])
    con.register("ai20_members_df", members)
    con.execute("CREATE OR REPLACE TABLE ai20_members AS SELECT * FROM ai20_members_df")
    _load_optional(con, "edgar_raw", "edgar_facts.parquet", EDGAR_COLS)
    _load_optional(con, "yahoo_fund_raw", "yahoo_fundamentals.parquet", YAHOO_FUND_COLS)
    for path in sorted(config.SQL_DIR.glob("*.sql")):
        t0 = time.time()
        con.execute(render(path.read_text()))
        log.info("warehouse: %s (%.1fs)", path.name, time.time() - t0)
    for (name,) in con.execute("SELECT table_name FROM information_schema.tables ORDER BY 1").fetchall():
        n = con.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0]
        log.info("  %-24s %10d rows", name, n)
    con.close()


def query(sql: str) -> pd.DataFrame:
    with connect(read_only=True) as con:
        return con.execute(sql).df()
