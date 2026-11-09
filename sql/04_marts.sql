-- 04_marts.sql  -- reporting marts used by the BI app, the ML models and the agent tools.

CREATE OR REPLACE TABLE mart_hourly_total AS
SELECT f.hour_idx, t.trace_day, t.hour_of_day,
       SUM(cost_usd) AS cost_usd, SUM(used_cost_usd) AS used_cost_usd,
       SUM(vm_hours) AS vm_hours, SUM(vcpu_hours) AS vcpu_hours
FROM fct_hourly_cost f JOIN dim_time t USING (hour_idx)
GROUP BY ALL ORDER BY 1;

CREATE OR REPLACE TABLE mart_daily_subscription AS
SELECT t.trace_day, d.subscription_alias, f.subscription_id,
       SUM(cost_usd) AS cost_usd, SUM(used_cost_usd) AS used_cost_usd,
       SUM(vm_hours) AS vm_hours
FROM fct_hourly_cost f
JOIN dim_time t USING (hour_idx)
JOIN dim_subscription d USING (subscription_id)
GROUP BY ALL;

CREATE OR REPLACE TABLE mart_daily_category AS
SELECT t.trace_day, f.vm_category,
       SUM(cost_usd) AS cost_usd, SUM(used_cost_usd) AS used_cost_usd
FROM fct_hourly_cost f JOIN dim_time t USING (hour_idx)
GROUP BY ALL;

CREATE OR REPLACE TABLE mart_daily_shape AS
SELECT t.trace_day, f.shape_id, s.reference_sku,
       SUM(cost_usd) AS cost_usd, SUM(vm_hours) AS vm_hours
FROM fct_hourly_cost f JOIN dim_time t USING (hour_idx)
JOIN dim_vm_shape s USING (shape_id)
GROUP BY ALL;

-- Hourly series for the top 50 subscriptions (used for per-team anomaly detection).
CREATE OR REPLACE TABLE mart_hourly_top_subscription AS
SELECT f.hour_idx, d.subscription_alias, SUM(f.cost_usd) AS cost_usd
FROM fct_hourly_cost f
JOIN dim_subscription d USING (subscription_id)
WHERE d.subscription_alias <= 'SUB-0050'
GROUP BY ALL;

-- VM-level utilisation facts used by the optimisation (rightsizing / idle) tool.
CREATE OR REPLACE TABLE mart_vm_utilization AS
SELECT v.vm_id, d.subscription_alias, v.vm_category, v.shape_id, s.vcpu, s.memory_gb,
       s.hourly_price_usd, v.lifetime_hours, v.avg_cpu, v.p95_max_cpu, v.max_cpu,
       v.lifetime_hours * s.hourly_price_usd AS cost_usd
FROM stg_vm v
JOIN dim_vm_shape s USING (shape_id)
JOIN dim_subscription d USING (subscription_id)
WHERE v.lifetime_hours >= 24;          -- only long-running VMs are rightsizing candidates

-- Hourly spend by workload category (forecasting / anomaly scope = category).
CREATE OR REPLACE TABLE mart_hourly_category AS
SELECT hour_idx, vm_category, SUM(cost_usd) AS cost_usd
FROM fct_hourly_cost GROUP BY ALL;

-- Daily spend at driver grain: lets the variance tool decompose a change by any dimension.
CREATE OR REPLACE TABLE mart_daily_driver AS
SELECT t.trace_day, d.subscription_alias, f.vm_category, f.shape_id,
       SUM(f.cost_usd) AS cost_usd, SUM(f.used_cost_usd) AS used_cost_usd, SUM(f.vm_hours) AS vm_hours
FROM fct_hourly_cost f
JOIN dim_time t USING (hour_idx)
JOIN dim_subscription d USING (subscription_id)
GROUP BY ALL;
