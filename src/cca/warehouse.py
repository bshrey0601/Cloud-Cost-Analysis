"""Build the DuckDB warehouse from the official Microsoft Azure trace.

    python -m cca.warehouse            # download (if needed) + build + reconcile

The SQL lives in /sql and is written to be close to Snowflake SQL; see
snowflake/README.md for the few dialect differences.
"""

from __future__ import annotations

import argparse
import os
import time
import urllib.request
from pathlib import Path

import duckdb
import pandas as pd

from cca import pricing

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
DB_PATH = ROOT / "data" / "warehouse.duckdb"          # full warehouse (~900 MB, built locally, not in git)
MARTS_DIR = ROOT / "data" / "marts"                    # compact Parquet marts (committed to git)
LITE_DB_PATH = ROOT / "data" / "warehouse_lite.duckdb"  # built from MARTS_DIR in seconds on first run

# Everything the app, models, agent and tests read. The 12.7M-row hourly fact table and the
# VM-level staging table stay in the full build only.
LITE_TABLES = [
    "dim_vm_shape", "dim_subscription", "dim_time", "dq_reconciliation",
    "mart_hourly_total", "mart_hourly_category", "mart_hourly_top_subscription",
    "mart_daily_subscription", "mart_daily_category", "mart_daily_shape", "mart_daily_driver",
    "mart_vm_utilization",
    "dim_deployment", "mart_daily_deployment",
]
SQL_DIR = ROOT / "sql"

RELEASE = "https://github.com/Azure/AzurePublicDataset/releases/download/dataset-v2/"
RAW_FILES = [
    "trace_data_vmtable_vmtable.csv.gz",          # ~438 MB
    "trace_data_subscriptions_subscriptions.csv.gz",
    "trace_data_deployments_deployments.csv.gz",
    "schema.csv",
]

VM_COLUMNS = {
    "vm_id": "VARCHAR", "subscription_id": "VARCHAR", "deployment_id": "VARCHAR",
    "created_s": "BIGINT", "deleted_s": "BIGINT", "max_cpu": "DOUBLE", "avg_cpu": "DOUBLE",
    "p95_max_cpu": "DOUBLE", "vm_category": "VARCHAR", "core_bucket": "VARCHAR",
    "memory_bucket": "VARCHAR",
}


def download_raw() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in RAW_FILES:
        dest = RAW_DIR / name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        print(f"Downloading {name} ...")
        urllib.request.urlretrieve(RELEASE + name, dest)


def convert_vmtable_to_parquet(con: duckdb.DuckDBPyConnection) -> None:
    out = RAW_DIR / "vmtable.parquet"
    if out.exists():
        return
    cols = ", ".join(f"'{k}': '{v}'" for k, v in VM_COLUMNS.items())
    con.execute(
        f"COPY (SELECT * FROM read_csv('{RAW_DIR}/trace_data_vmtable_vmtable.csv.gz', "
        f"header=false, columns={{{cols}}})) TO '{out}' (FORMAT parquet)"
    )


def run_sql_file(con: duckdb.DuckDBPyConnection, path: Path) -> None:
    sql = path.read_text().replace("{raw_dir}", str(RAW_DIR))
    t = time.time()
    con.execute(sql)
    print(f"  ran {path.name} in {time.time() - t:.1f}s")


def reconcile(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Reconciliation checks: every total in the warehouse must tie back to the raw table."""
    checks = {
        "VM count: raw vs staging": (
            "SELECT COUNT(*) FROM read_parquet('{raw}/vmtable.parquet')",
            "SELECT COUNT(*) FROM stg_vm",
        ),
        "Spend: VM-level (lifetime x price) vs hourly fact": (
            "SELECT SUM(v.lifetime_hours * s.hourly_price_usd) FROM stg_vm v JOIN dim_vm_shape s USING (shape_id)",
            "SELECT SUM(cost_usd) FROM fct_hourly_cost",
        ),
        "VM-hours: VM-level vs hourly fact": (
            "SELECT SUM(lifetime_hours) FROM stg_vm",
            "SELECT SUM(vm_hours) FROM fct_hourly_cost",
        ),
        "Spend: hourly fact vs daily subscription mart": (
            "SELECT SUM(cost_usd) FROM fct_hourly_cost",
            "SELECT SUM(cost_usd) FROM mart_daily_subscription",
        ),
        "Spend: hourly fact vs subscription dimension": (
            "SELECT SUM(cost_usd) FROM fct_hourly_cost",
            "SELECT SUM(total_spend_usd) FROM dim_subscription",
        ),
        "Spend: hourly fact vs daily deployment mart": (
            "SELECT SUM(cost_usd) FROM fct_hourly_cost",
            "SELECT SUM(cost_usd) FROM mart_daily_deployment",
        ),
        "Spend: hourly fact vs deployment dimension": (
            "SELECT SUM(cost_usd) FROM fct_hourly_cost",
            "SELECT SUM(total_spend_usd) FROM dim_deployment",
        ),
        "Unpriced shapes (must be 0)": (
            "SELECT 0",
            "SELECT COUNT(*) FROM stg_vm WHERE shape_id NOT IN (SELECT shape_id FROM dim_vm_shape)",
        ),
    }
    rows = []
    for name, (expected_sql, actual_sql) in checks.items():
        exp = con.execute(expected_sql.replace("{raw}", str(RAW_DIR))).fetchone()[0]
        act = con.execute(actual_sql).fetchone()[0]
        diff = abs(float(act) - float(exp))
        rel = diff / abs(float(exp)) if exp else diff
        rows.append({"check": name, "expected": exp, "actual": act, "abs_diff": diff,
                     "status": "PASS" if rel < 1e-9 else "FAIL"})
    df = pd.DataFrame(rows)
    con.execute("CREATE OR REPLACE TABLE dq_reconciliation AS SELECT * FROM df")
    return df


def build(download: bool = True) -> Path:
    if download:
        download_raw()
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()
    con = duckdb.connect(str(DB_PATH))
    con.execute("SET memory_limit='4GB'")
    con.execute(f"SET temp_directory='{ROOT / 'data' / 'tmp'}'")
    convert_vmtable_to_parquet(con)

    run_sql_file(con, SQL_DIR / "01_staging.sql")
    shapes = pd.DataFrame(pricing.shape_rows())
    con.execute("CREATE OR REPLACE TABLE dim_vm_shape AS SELECT * FROM shapes")
    for name in ["02_dimensions.sql", "03_fact_hourly_cost.sql", "04_marts.sql", "05_deployments.sql"]:
        run_sql_file(con, SQL_DIR / name)

    report = reconcile(con)
    print(report.to_string(index=False))
    if (report["status"] != "PASS").any():
        raise SystemExit("Reconciliation failed -- see dq_reconciliation table")
    export_marts(con)
    con.close()
    return DB_PATH


def export_marts(con: duckdb.DuckDBPyConnection) -> None:
    """Write the reporting marts as compressed Parquet so the app runs from a plain git clone."""
    MARTS_DIR.mkdir(parents=True, exist_ok=True)
    for t in LITE_TABLES:
        cols = {"mart_vm_utilization": "* EXCLUDE (vm_id)",            # 88-char hashes not needed
                "dim_deployment": "* EXCLUDE (deployment_id)"}.get(t, "*")
        con.execute(f"COPY (SELECT {cols} FROM {t}) TO '{MARTS_DIR / (t + '.parquet')}' "
                    f"(FORMAT parquet, COMPRESSION zstd)")
    print(f"  exported {len(LITE_TABLES)} marts to {MARTS_DIR}")
    if LITE_DB_PATH.exists():
        LITE_DB_PATH.unlink()   # rebuilt from the fresh marts on next use


def build_lite() -> Path:
    """Build the lite warehouse from the committed Parquet marts (a few seconds).

    Written to a temporary file and moved into place atomically, so two app sessions
    starting at the same moment can never see a half-built database.
    """
    missing = [t for t in LITE_TABLES if not (MARTS_DIR / f"{t}.parquet").exists()]
    if missing:
        raise FileNotFoundError(f"Missing marts {missing}. Run: python -m cca.warehouse")
    tmp = LITE_DB_PATH.with_suffix(f".{os.getpid()}.tmp")
    if tmp.exists():
        tmp.unlink()
    con = duckdb.connect(str(tmp))
    for t in LITE_TABLES:
        con.execute(f"CREATE TABLE {t} AS SELECT * FROM read_parquet('{MARTS_DIR / (t + '.parquet')}')")
    con.close()
    os.replace(tmp, LITE_DB_PATH)
    return LITE_DB_PATH


def active_db_path() -> Path:
    """The full warehouse if it has been built locally, otherwise the lite one from the marts."""
    if DB_PATH.exists():
        return DB_PATH
    if not LITE_DB_PATH.exists():
        build_lite()
    return LITE_DB_PATH


def connect(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(active_db_path()), read_only=read_only)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--export-marts-only", action="store_true",
                   help="re-export the Parquet marts from an existing full warehouse")
    args = p.parse_args()
    if args.export_marts_only:
        c = duckdb.connect(str(DB_PATH), read_only=True)
        export_marts(c)
        c.close()
    else:
        build(download=not args.no_download)
