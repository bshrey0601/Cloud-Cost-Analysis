-- 01_staging.sql
-- Stage the official Microsoft Azure Public Dataset V2 VM table (2019, 30 days, 2.7M VMs).
-- Source: https://github.com/Azure/AzurePublicDataset (AzurePublicDatasetV2.md)
-- Timestamps are seconds from trace start; trace length = 30 days = 2,592,000 s.
-- Each timestamp marks the START of a 5-minute CPU reading, so a VM with last reading at
-- `deleted_s` ran until deleted_s + 300. VMs still running at the final reading of the trace
-- are right-censored: they were alive when observation stopped, so we run them to the end of
-- day 30 (2,592,000 s). Without both rules the last hour of the month is partially observed and
-- shows a fake end-of-month dip -- found because the 7-day forecast extrapolated that dip.

CREATE OR REPLACE TABLE stg_vm AS
SELECT
    vm_id,
    subscription_id,
    deployment_id,
    created_s,
    deleted_s,
    CASE WHEN deleted_s >= MAX(deleted_s) OVER () THEN 2592000
         ELSE deleted_s + 300 END                        AS end_s,
    (CASE WHEN deleted_s >= MAX(deleted_s) OVER () THEN 2592000
          ELSE deleted_s + 300 END - created_s) / 3600.0 AS lifetime_hours,
    max_cpu,
    avg_cpu,
    p95_max_cpu,
    vm_category,                                         -- Interactive | Delay-insensitive | Unknown
    core_bucket,
    memory_bucket,
    core_bucket || 'c_' || memory_bucket || 'g'          AS shape_id
FROM read_parquet('{raw_dir}/vmtable.parquet')
WHERE deleted_s >= created_s;                            -- guard against malformed rows

CREATE OR REPLACE TABLE stg_subscription AS
SELECT * FROM read_csv('{raw_dir}/trace_data_subscriptions_subscriptions.csv.gz', header = false,
    columns = {'subscription_id': 'VARCHAR', 'first_vm_created_s': 'BIGINT', 'vm_count': 'BIGINT'});

CREATE OR REPLACE TABLE stg_deployment AS
SELECT * FROM read_csv('{raw_dir}/trace_data_deployments_deployments.csv.gz', header = false,
    columns = {'deployment_id': 'VARCHAR', 'deployment_size': 'BIGINT'});
