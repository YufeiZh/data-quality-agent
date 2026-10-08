"""Synthetic orders dataset with reproducible, labelled anomaly injection.

Writes `<out>/orders.csv` and `<out>/orders.truth.json`. The truth file lists, per anomaly kind,
the affected `order_id`s (and whether it is a real defect or a legitimate oddity that must NOT
be "repaired"), so detection precision/recall can be scored later.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFECT_KINDS = {
    "missing_customer": "explicit: NULL customer_id",
    "duplicate_rows": "explicit: exact duplicate rows",
    "mixed_type_quantity": "explicit: non-numeric tokens in numeric column",
    "price_outlier": "statistical: unit_price inflated x1000",
    "weight_unit_error": "statistical: weight recorded in grams instead of kg",
    "ship_before_order": "implicit: ship_date earlier than order_date",
    "total_mismatch": "implicit: total_amount != quantity * unit_price",
}
LEGIT_KINDS = {"bulk_orders": "legitimate: very large quantity, totals consistent"}


def generate(out_dir: str | Path, n: int = 5000, seed: int = 42) -> Path:
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    order_date = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    qty = rng.integers(1, 20, n)
    price = np.round(rng.lognormal(3.0, 0.4, n), 2)
    df = pd.DataFrame(
        {
            "order_id": [f"O{i:06d}" for i in range(n)],
            "customer_id": [f"C{c:04d}" for c in rng.integers(0, 800, n)],
            "order_date": order_date,
            "ship_date": order_date + pd.to_timedelta(rng.integers(1, 8, n), unit="D"),
            "quantity": qty.astype(object),
            "unit_price": price,
            "total_amount": np.round(qty * price, 2),
            "weight_kg": np.round(rng.normal(2.0, 0.5, n).clip(0.1), 3),
            "currency": "USD",
            "status": rng.choice(["shipped", "delivered", "returned"], n, p=[0.3, 0.65, 0.05]),
        }
    )

    pool = list(rng.permutation(n))  # disjoint victims per anomaly kind
    truth: dict[str, dict] = {}

    def take(k: int) -> list[int]:
        return [pool.pop() for _ in range(k)]

    def record(kind: str, idx: list[int], desc: str, defect: bool = True) -> None:
        truth[kind] = {
            "defect": defect,
            "description": desc,
            "order_ids": sorted(df.loc[idx, "order_id"]),
        }

    idx = take(250)
    df.loc[idx, "customer_id"] = None
    record("missing_customer", idx, DEFECT_KINDS["missing_customer"])

    idx = take(12)
    df.loc[idx, "quantity"] = rng.choice(["N/A", "unknown", "ten"], len(idx))
    record("mixed_type_quantity", idx, DEFECT_KINDS["mixed_type_quantity"])

    idx = take(10)
    df.loc[idx, "unit_price"] = df.loc[idx, "unit_price"] * 1000
    df.loc[idx, "total_amount"] = df.loc[idx, "total_amount"] * 1000
    record("price_outlier", idx, DEFECT_KINDS["price_outlier"])

    idx = take(40)
    df.loc[idx, "weight_kg"] = df.loc[idx, "weight_kg"] * 1000
    record("weight_unit_error", idx, DEFECT_KINDS["weight_unit_error"])

    idx = take(30)
    df.loc[idx, "ship_date"] = df.loc[idx, "order_date"] - pd.to_timedelta(
        rng.integers(1, 6, len(idx)), unit="D"
    )
    record("ship_before_order", idx, DEFECT_KINDS["ship_before_order"])

    idx = take(20)
    df.loc[idx, "total_amount"] = np.round(df.loc[idx, "total_amount"] * 1.5 + 5, 2)
    record("total_mismatch", idx, DEFECT_KINDS["total_mismatch"])

    # Legitimate: big bulk orders, internally consistent (not a defect).
    idx = take(15)
    bulk = rng.integers(500, 900, len(idx))
    df.loc[idx, "quantity"] = bulk.astype(object)
    df.loc[idx, "total_amount"] = np.round(bulk * df.loc[idx, "unit_price"].astype(float), 2)
    record("bulk_orders", idx, LEGIT_KINDS["bulk_orders"], defect=False)

    # Exact duplicates appended last (copies of random existing rows).
    dup_src = rng.choice(n, 25, replace=False)
    dups = df.iloc[dup_src].copy()
    truth["duplicate_rows"] = {
        "defect": True,
        "description": DEFECT_KINDS["duplicate_rows"],
        "order_ids": sorted(dups["order_id"]),
    }
    df = pd.concat([df, dups], ignore_index=True).sample(frac=1, random_state=seed)

    df.to_csv(out / "orders.csv", index=False, date_format="%Y-%m-%d")
    (out / "orders.truth.json").write_text(json.dumps(truth, indent=2))
    return out / "orders.csv"
