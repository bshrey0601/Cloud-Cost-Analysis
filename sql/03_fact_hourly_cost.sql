-- 03_fact_hourly_cost.sql
-- Grain: one row per (hour, subscription, vm_category, shape).
-- Each VM's lifetime is split across the hours it overlaps, so partial hours are
-- billed fractionally (per-second billing, like Azure).
--   vm_hours       = hours the VM was running in that hour bucket
--   cost_usd       = vm_hours * list price of its shape
--   used_cost_usd  = cost_usd * avg CPU utilisation (share of spend doing work)

CREATE OR REPLACE TABLE fct_hourly_cost AS
WITH vm AS (
    SELECT v.subscription_id, v.vm_category, v.shape_id, v.avg_cpu,
           v.created_s / 3600.0 AS start_h,
           v.end_s / 3600.0 AS end_h,
           s.hourly_price_usd, s.vcpu
    FROM stg_vm v
    JOIN dim_vm_shape s USING (shape_id)
),
overlap AS (
    -- Range join against the 720-hour calendar: one row per hour a VM was alive.
    SELECT t.hour_idx, vm.subscription_id, vm.vm_category, vm.shape_id, vm.avg_cpu,
           vm.hourly_price_usd, vm.vcpu,
           LEAST(vm.end_h, t.hour_idx + 1) - GREATEST(vm.start_h, t.hour_idx) AS vm_hours
    FROM vm
    JOIN dim_time t
      ON t.hour_idx + 1 > vm.start_h
     AND t.hour_idx     < vm.end_h
)
SELECT
    hour_idx,
    subscription_id,
    vm_category,
    shape_id,
    COUNT(*)                                        AS vm_count,
    SUM(vm_hours)                                   AS vm_hours,
    SUM(vm_hours * vcpu)                            AS vcpu_hours,
    SUM(vm_hours * hourly_price_usd)                AS cost_usd,
    SUM(vm_hours * hourly_price_usd * avg_cpu / 100.0) AS used_cost_usd
FROM overlap
WHERE vm_hours > 0
GROUP BY ALL;
