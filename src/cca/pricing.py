"""Price book: maps each VM shape in the Microsoft Azure trace to an hourly list price.

Source of prices
----------------
Azure Linux pay-as-you-go list prices (USD/hour, East US and other lowest-price US
regions), as published for Nov 2026 on azurespeed.com and cloudprice.net
(both aggregate Microsoft's Azure Retail Prices API):

    Standard_D2s_v5   2 vCPU   8 GB   $0.0960   general purpose
    Standard_D8s_v5   8 vCPU  32 GB   $0.3840   general purpose
    Standard_D32s_v5 32 vCPU 128 GB   $1.5360   general purpose
    Standard_E2s_v5   2 vCPU  16 GB   $0.1260   memory optimized
    Standard_E4s_v5   4 vCPU  32 GB   $0.2520   memory optimized
    Standard_E4-2s_v5 2 vCPU  32 GB   $0.2520   memory optimized, constrained vCPU
    Standard_E8s_v5   8 vCPU  64 GB   $0.5040   memory optimized
    Standard_F2s_v2   2 vCPU   4 GB   $0.0846   compute optimized
    Standard_F4s_v2   4 vCPU   8 GB   $0.1690   compute optimized

Method
------
1. Where a trace shape (vCPU bucket, memory bucket) matches a real SKU exactly,
   we use that SKU's list price.
2. Otherwise we use a linear price model  price = a * vCPU + b * GB
   fitted exactly on D2s_v5 and E2s_v5 (same CPU generation, different memory):
       2a +  8b = 0.096
       2a + 16b = 0.126   ->   b = $0.00375 per GB-hour, a = $0.033 per vCPU-hour
   Out-of-sample check: the model prices F2s_v2 (2 vCPU, 4 GB) at $0.0810
   versus its list price of $0.0846 (4.3% low), D8s_v5 at $0.384 (exact) and
   E8s_v5 at $0.504 (exact).
3. The trace's open-ended buckets (">24" vCPU, ">64" GB) are priced as
   Standard_D32s_v5 (32 vCPU, 128 GB) -- an explicit, conservative assumption.

Important caveat
----------------
The trace is 2019 usage; we price it at today's (Nov 2026) list rates. The goal is a
realistic *cost* signal for a FinOps model, not Microsoft's actual internal cost.
"""

from __future__ import annotations

PRICE_AS_OF = "2026-11"
PRICE_SOURCE = (
    "Azure Linux pay-as-you-go list prices, lowest US region "
    "(azurespeed.com / cloudprice.net, from Azure Retail Prices API), Nov 2026"
)

VCPU_RATE = 0.033  # $/vCPU-hour, fitted on D2s_v5 + E2s_v5
GB_RATE = 0.00375  # $/GB-hour, fitted on D2s_v5 + E2s_v5

# Official SKUs used directly (vcpu, gb) -> (sku, price)
EXACT_SKUS: dict[tuple[int, int], tuple[str, float]] = {
    (2, 4): ("Standard_F2s_v2", 0.0846),
    (2, 8): ("Standard_D2s_v5", 0.0960),
    (2, 16): ("Standard_E2s_v5", 0.1260),
    (2, 32): ("Standard_E4-2s_v5", 0.2520),
    (4, 8): ("Standard_F4s_v2", 0.1690),
    (4, 32): ("Standard_E4s_v5", 0.2520),
    (8, 32): ("Standard_D8s_v5", 0.3840),
    (8, 64): ("Standard_E8s_v5", 0.5040),
    (32, 128): ("Standard_D32s_v5", 1.5360),
}

# Trace buckets -> numeric value used for pricing
CORE_BUCKET_VALUE = {"2": 2, "4": 4, "8": 8, "24": 24, ">24": 32}
MEMORY_BUCKET_VALUE = {"2": 2, "4": 4, "8": 8, "32": 32, "64": 64, ">64": 128}

# Ordered ladder of shapes used by rightsizing (one step down halves vCPU)
DOWNSIZE_STEP = {32: 24, 24: 8, 8: 4, 4: 2}


def linear_price(vcpu: float, gb: float) -> float:
    return round(VCPU_RATE * vcpu + GB_RATE * gb, 4)


def price_for(core_bucket: str, memory_bucket: str) -> tuple[str, float, str]:
    """Return (reference_sku, hourly_price, method) for a trace shape."""
    vcpu = CORE_BUCKET_VALUE[core_bucket]
    gb = MEMORY_BUCKET_VALUE[memory_bucket]
    if (vcpu, gb) in EXACT_SKUS:
        sku, price = EXACT_SKUS[(vcpu, gb)]
        return sku, price, "exact list price"
    return f"custom_{vcpu}vcpu_{gb}gb", linear_price(vcpu, gb), "linear model (D/E v5 fit)"


def shape_rows() -> list[dict]:
    """All shape combinations as rows for the dim_vm_shape table."""
    rows = []
    for cb, vcpu in CORE_BUCKET_VALUE.items():
        for mb, gb in MEMORY_BUCKET_VALUE.items():
            sku, price, method = price_for(cb, mb)
            rows.append(
                {
                    "shape_id": f"{cb}c_{mb}g",
                    "core_bucket": cb,
                    "memory_bucket": mb,
                    "vcpu": vcpu,
                    "memory_gb": gb,
                    "reference_sku": sku,
                    "hourly_price_usd": price,
                    "price_method": method,
                    "vcpu_rate_usd": round(price / vcpu, 5),
                }
            )
    return rows
