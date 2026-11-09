-- 05_deployments.sql
-- Deployment-level marts for ROOT-CAUSE ATTRIBUTION: when a team's spend moves, which
-- deployment (a group of VMs launched together, e.g. a service or test environment) caused it.

-- Daily spend per deployment: each VM's lifetime is split across the trace days it overlaps.
CREATE OR REPLACE TABLE mart_daily_deployment AS
WITH vm AS (
    SELECT v.deployment_id, v.subscription_id, v.created_s, v.end_s, s.hourly_price_usd
    FROM stg_vm v JOIN dim_vm_shape s USING (shape_id)
),
days AS (SELECT d AS trace_day FROM range(1, 31) t(d)),
overlap AS (
    SELECT d.trace_day, vm.deployment_id, vm.subscription_id,
           (LEAST(vm.end_s, d.trace_day * 86400) - GREATEST(vm.created_s, (d.trace_day - 1) * 86400)) / 3600.0
               AS vm_hours,
           vm.hourly_price_usd,
           CAST(vm.created_s >= (d.trace_day - 1) * 86400 AND vm.created_s < d.trace_day * 86400 AS INTEGER) AS created_today
    FROM vm JOIN days d
      ON vm.created_s < d.trace_day * 86400
     AND vm.end_s     > (d.trace_day - 1) * 86400
)
SELECT trace_day, deployment_id, subscription_id,
       SUM(vm_hours * hourly_price_usd) AS cost_usd,
       SUM(vm_hours)                    AS vm_hours,
       COUNT(*)                         AS active_vms,
       SUM(created_today)               AS vms_created
FROM overlap
WHERE vm_hours > 0
GROUP BY ALL;

-- Deployment dimension with a readable alias ranked by 30-day spend.
CREATE OR REPLACE TABLE dim_deployment AS
WITH agg AS (
    SELECT v.deployment_id, v.subscription_id,
           COUNT(*)                                   AS vm_count,
           MIN(v.created_s) // 86400 + 1              AS first_day,
           (MAX(v.end_s) - 1) // 86400 + 1            AS last_day,
           SUM(v.lifetime_hours * s.hourly_price_usd) AS total_spend_usd,
           MODE(v.vm_category)                        AS primary_category,
           MODE(s.reference_sku)                      AS primary_sku,
           BOOL_OR(v.created_s = 0)                   AS running_at_trace_start
    FROM stg_vm v JOIN dim_vm_shape s USING (shape_id)
    GROUP BY 1, 2
)
SELECT 'DEP-' || LPAD(CAST(ROW_NUMBER() OVER (ORDER BY a.total_spend_usd DESC, a.deployment_id) AS VARCHAR), 5, '0')
           AS deployment_alias,
       a.deployment_id, d.subscription_alias, a.vm_count, a.first_day, a.last_day,
       a.total_spend_usd, a.primary_category, a.primary_sku, a.running_at_trace_start,
       sd.deployment_size AS microsoft_deployment_size
FROM agg a
JOIN dim_subscription d USING (subscription_id)
LEFT JOIN stg_deployment sd USING (deployment_id);

-- Swap raw hashes for aliases in the daily mart (keeps the committed Parquet small).
CREATE OR REPLACE TABLE mart_daily_deployment AS
SELECT m.trace_day, dd.deployment_alias, dd.subscription_alias,
       m.cost_usd, m.vm_hours, m.active_vms, m.vms_created
FROM mart_daily_deployment m JOIN dim_deployment dd USING (deployment_id);
