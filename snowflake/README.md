# Running on Snowflake

The pipeline is developed locally in DuckDB so anyone can rebuild it for free in ~5 minutes.
`load_to_snowflake.py` publishes the reporting marts to Snowflake and re-runs the spend
reconciliation across the two systems.

To run the full SQL build natively in Snowflake instead, the scripts in `/sql` need only these
dialect changes:

| DuckDB (this repo) | Snowflake |
|---|---|
| `read_parquet('{raw_dir}/vmtable.parquet')` | external stage + `COPY INTO stg_vm FROM @stage FILE_FORMAT=(TYPE=PARQUET) MATCH_BY_COLUMN_NAME=CASE_INSENSITIVE` |
| `FROM range(0, 720) t(h)` | `FROM TABLE(GENERATOR(ROWCOUNT => 720))` with `SEQ4() AS h` |
| `QUANTILE_CONT(x, 0.05)` | `PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY x)` |
| `GROUP BY ALL` | supported (Snowflake also has `GROUP BY ALL`) |
| `CREATE OR REPLACE TABLE ... AS` | identical |

The range join in `03_fact_hourly_cost.sql` (VM lifetimes x hour calendar) is plain ANSI SQL.
For production, cluster `fct_hourly_cost` by `hour_idx` and run the hourly build incrementally.

**Streamlit in Snowflake:** `app/streamlit_app.py` reads only the marts above, so it can be
pointed at Snowflake by replacing `cca.warehouse.connect()` with a Snowpark session query.
These Snowflake steps are documented but were not run against a live account.
