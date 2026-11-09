-- 02_dimensions.sql
-- dim_vm_shape is loaded from Python (src/cca/pricing.py) before this script runs.

-- Subscriptions get a readable alias ranked by 30-day spend (the raw IDs are encrypted hashes).
CREATE OR REPLACE TABLE dim_subscription AS
WITH spend AS (
    SELECT v.subscription_id,
           SUM(v.lifetime_hours * s.hourly_price_usd) AS spend_usd,
           COUNT(*)                                    AS vm_count,
           COUNT(DISTINCT v.deployment_id)             AS deployment_count,
           MODE(v.vm_category)                         AS primary_category
    FROM stg_vm v
    JOIN dim_vm_shape s USING (shape_id)
    GROUP BY 1
)
SELECT
    'SUB-' || LPAD(CAST(ROW_NUMBER() OVER (ORDER BY spend_usd DESC, subscription_id) AS VARCHAR), 4, '0') AS subscription_alias,
    subscription_id,
    vm_count,
    deployment_count,
    primary_category,
    spend_usd AS total_spend_usd
FROM spend;

CREATE OR REPLACE TABLE dim_time AS
SELECT
    h                        AS hour_idx,
    h // 24 + 1              AS trace_day,           -- 1..30
    h % 24                   AS hour_of_day,
    (h // 24) // 7 + 1       AS trace_week,          -- 1..5 (week 5 = days 29-30)
    (h // 24) % 7            AS day_of_trace_week    -- calendar weekday is not published in the trace
FROM range(0, 720) t(h);
