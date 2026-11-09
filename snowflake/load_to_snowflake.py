"""Load the warehouse marts into Snowflake (optional).

    pip install "snowflake-connector-python[pandas]"
    export SNOWFLAKE_ACCOUNT=... SNOWFLAKE_USER=... SNOWFLAKE_PASSWORD=...   # or key-pair auth
    export SNOWFLAKE_WAREHOUSE=COMPUTE_WH SNOWFLAKE_DATABASE=CLOUD_COST SNOWFLAKE_SCHEMA=MARTS
    python snowflake/load_to_snowflake.py

The full pipeline is built locally in DuckDB (fast, free); this script publishes the
reporting marts -- the tables the BI app, models and agent read -- to Snowflake with
`write_pandas`, which stages Parquet and runs COPY INTO under the hood.
NOTE: written against the documented connector API but not run against a live account.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cca.warehouse import connect  # noqa: E402

TABLES = [
    "dim_vm_shape", "dim_subscription", "dim_time", "dq_reconciliation",
    "mart_hourly_total", "mart_hourly_category", "mart_hourly_top_subscription",
    "mart_daily_subscription", "mart_daily_category", "mart_daily_shape", "mart_daily_driver",
    "mart_vm_utilization",
]


def main() -> None:
    import snowflake.connector
    from snowflake.connector.pandas_tools import write_pandas

    sf = snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"], user=os.environ["SNOWFLAKE_USER"],
        password=os.environ.get("SNOWFLAKE_PASSWORD"), warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
    )
    db = os.environ.get("SNOWFLAKE_DATABASE", "CLOUD_COST")
    schema = os.environ.get("SNOWFLAKE_SCHEMA", "MARTS")
    cur = sf.cursor()
    cur.execute(f"CREATE DATABASE IF NOT EXISTS {db}")
    cur.execute(f"CREATE SCHEMA IF NOT EXISTS {db}.{schema}")
    cur.execute(f"USE SCHEMA {db}.{schema}")
    duck = connect()
    for t in TABLES:
        df = duck.execute(f"SELECT * FROM {t}").df()
        df.columns = [c.upper() for c in df.columns]
        ok, _, rows, _ = write_pandas(sf, df, t.upper(), auto_create_table=True, overwrite=True)
        print(f"{t:32s} {'OK' if ok else 'FAILED'} {rows:>12,} rows")
    # Same reconciliation idea, now across systems: totals must tie out after the load.
    duck_total = duck.execute("SELECT SUM(cost_usd) FROM mart_daily_driver").fetchone()[0]
    sf_total = cur.execute("SELECT SUM(COST_USD) FROM MART_DAILY_DRIVER").fetchone()[0]
    print(f"Reconciliation DuckDB vs Snowflake: {duck_total:,.2f} vs {sf_total:,.2f}")


if __name__ == "__main__":
    main()
