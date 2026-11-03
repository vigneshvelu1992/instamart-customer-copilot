
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# PART B - TASK 1: BUILD RFM TABLE
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

ORDERS_FILE = ROOT / "data" / "clean" / "orders.csv"
OUTPUT_FILE = ROOT / "outputs" / "rfm_table.csv"

SNAPSHOT_DATE = pd.Timestamp("2025-10-01 00:00:00")


def main():

    print("=" * 60)
    print("PART B - TASK 1: BUILD RFM TABLE")
    print("=" * 60)

    # 1. Load cleaned orders
    orders = pd.read_csv(ORDERS_FILE)

    # 2. Check required columns
    required_columns = {
        "order_id",
        "customer_id",
        "order_ts",
        "status",
        "order_amount"
    }

    missing_columns = required_columns - set(orders.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # 3. Convert order timestamp
    orders["order_ts"] = pd.to_datetime(
        orders["order_ts"],
        errors="coerce"
    )

    # 4. Select completed orders only
    completed = orders[
        orders["status"].str.strip().str.lower() == "completed"
    ].copy()

    completed["order_amount"] = pd.to_numeric(
        completed["order_amount"],
        errors="coerce"
    )

    if completed["order_amount"].isna().any():
        raise ValueError(
            "Completed orders contain missing order amounts."
        )

    if completed["order_ts"].isna().any():
        raise ValueError(
            "Completed orders contain invalid timestamps."
        )

    # 5. Calculate RFM metrics
    rfm = (
        completed.groupby("customer_id")
        .agg(
            last_order_ts=("order_ts", "max"),
            frequency=("order_id", "nunique"),
            monetary=("order_amount", "sum")
        )
        .reset_index()
    )

    # 6. Calculate recency in whole days
    rfm["recency"] = (
        SNAPSHOT_DATE - rfm["last_order_ts"]
    ).dt.days

    # 7. Arrange columns
    rfm = rfm[
        [
            "customer_id",
            "recency",
            "frequency",
            "monetary"
        ]
    ]

    rfm = rfm.sort_values(
        "customer_id"
    ).reset_index(drop=True)

    # 8. Validate reference checkpoints
    print("\nRFM REFERENCE CHECKS")

    checks = {
        "Customer count": len(rfm) == 1175,
        "Recency median": rfm["recency"].median() == 19,
        "Recency minimum": rfm["recency"].min() == 0,
        "Recency maximum": rfm["recency"].max() == 180,
        "Frequency maximum": rfm["frequency"].max() == 33,
        "Frequency mean": round(rfm["frequency"].mean(), 2) == 9.46,
        "Monetary mean": round(rfm["monetary"].mean(), 2) == 2552.06,
        "Monetary maximum": round(rfm["monetary"].max(), 2) == 14766.55
    }

    for name, passed in checks.items():
        print(f"{name}: {'PASSED' if passed else 'FAILED'}")

    if not all(checks.values()):
        raise AssertionError(
            "One or more RFM reference checks failed."
        )

    # 9. Save RFM table
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    rfm.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # 10. Display summary
    print("\nRFM SUMMARY")
    print(rfm[["recency", "frequency", "monetary"]].describe())

    print("\nFIRST 10 CUSTOMERS")
    print(rfm.head(10).to_string(index=False))

    print("\n" + "=" * 60)
    print("TASK 1 COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(f"\nSaved output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
