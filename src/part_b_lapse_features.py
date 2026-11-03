
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

ORDERS_FILE = ROOT / "data" / "clean" / "orders.csv"
CUSTOMERS_FILE = ROOT / "data" / "clean" / "customers.csv"
PRODUCTS_FILE = ROOT / "data" / "clean" / "products.csv"

OUTPUT_FILE = ROOT / "outputs" / "lapse_cohort.csv"

CUTOFF = pd.Timestamp("2025-09-01")
LABEL_END = pd.Timestamp("2025-10-01")


def main():
    print("=" * 60)
    print("PART B - TASK 4: LAPSE PREDICTION COHORT")
    print("=" * 60)

    # 1. Load datasets
    orders = pd.read_csv(ORDERS_FILE)
    customers = pd.read_csv(CUSTOMERS_FILE)
    products = pd.read_csv(PRODUCTS_FILE)

    orders["order_ts"] = pd.to_datetime(orders["order_ts"])
    customers["signup_date"] = pd.to_datetime(customers["signup_date"])

    # 2. Create the cohort using completed orders before cutoff
    completed_before = orders[
        (orders["status"] == "completed") &
        (orders["order_ts"] < CUTOFF)
    ].copy()

    cohort_ids = completed_before["customer_id"].unique()

    # 3. Create lapse labels using September completed orders
    september_completed = orders[
        (orders["status"] == "completed") &
        (orders["order_ts"] >= CUTOFF) &
        (orders["order_ts"] < LABEL_END)
    ]

    september_active_ids = set(
        september_completed["customer_id"].unique()
    )

    cohort = pd.DataFrame({"customer_id": cohort_ids})

    cohort["lapsed"] = (
        ~cohort["customer_id"].isin(september_active_ids)
    ).astype(int)

    print("\nCOHORT SUMMARY")
    print("Total cohort customers:", len(cohort))
    print("Lapsed customers:", int(cohort["lapsed"].sum()))
    print("Not-lapsed customers:", int((cohort["lapsed"] == 0).sum()))
    print(
        "Lapse percentage:",
        round(cohort["lapsed"].mean() * 100, 2), "%"
    )

    # 4. Use only pre-cutoff orders for model features
    pre_cutoff_orders = orders[
        orders["order_ts"] < CUTOFF
    ].copy()

    pre_completed = pre_cutoff_orders[
        pre_cutoff_orders["status"] == "completed"
    ].copy()

    # Add product category to completed orders
    pre_completed = pre_completed.merge(
        products[["product_id", "category"]],
        on="product_id",
        how="left"
    )

    # 5. Calculate completed-order features
    completed_features = (
        pre_completed.groupby("customer_id")
        .agg(
            last_order_ts=("order_ts", "max"),
            frequency=("order_id", "nunique"),
            monetary=("order_amount", "sum"),
            avg_discount_pct=("discount_pct", "mean"),
            avg_delivery_minutes=("delivery_minutes", "mean"),
            favourite_category=(
                "category",
                lambda x: x.mode().iloc[0]
            )
        )
        .reset_index()
    )

    completed_features["recency"] = (
        CUTOFF - completed_features["last_order_ts"]
    ).dt.days

    completed_features = completed_features.drop(
        columns=["last_order_ts"]
    )

    # 6. Calculate cancellation rate from all final-status orders
    final_orders = pre_cutoff_orders[
        pre_cutoff_orders["status"].isin(["completed", "cancelled"])
    ].copy()

    final_orders["is_cancelled"] = (
        final_orders["status"] == "cancelled"
    ).astype(int)

    cancellation_features = (
        final_orders.groupby("customer_id")
        .agg(
            cancellation_rate=("is_cancelled", "mean")
        )
        .reset_index()
    )

    # 7. Combine customer details
    customer_features = customers[
        ["customer_id", "city", "age", "gender", "signup_date"]
    ].copy()

    customer_features["tenure_days"] = (
        CUTOFF - customer_features["signup_date"]
    ).dt.days

    customer_features = customer_features.drop(
        columns=["signup_date"]
    )

    # 8. Merge all features with the cohort and labels
    result = cohort.merge(
        completed_features,
        on="customer_id",
        how="left"
    )

    result = result.merge(
        cancellation_features,
        on="customer_id",
        how="left"
    )

    result = result.merge(
        customer_features,
        on="customer_id",
        how="left"
    )

    # 9. Validate feature completeness
    feature_columns = [
        "recency",
        "frequency",
        "monetary",
        "avg_discount_pct",
        "cancellation_rate",
        "avg_delivery_minutes",
        "favourite_category",
        "city",
        "age",
        "gender",
        "tenure_days"
    ]

    missing_counts = result[feature_columns].isna().sum()

    print("\nMISSING FEATURE VALUES")
    print(missing_counts.to_string())

    assert not result[feature_columns].isna().any().any(), (
        "Missing values found in features. Investigate before proceeding."
    )

    assert len(result) == 1156
    assert int(result["lapsed"].sum()) == 443

    # 10. Save cohort
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)

    print("\nFEATURE PREVIEW")
    print(result.head(5).to_string(index=False))

    print("\nFINAL VALIDATION")
    print("Cohort size:", len(result))
    print("Lapsed:", int(result["lapsed"].sum()))
    print("Not lapsed:", int((result["lapsed"] == 0).sum()))
    print("Training feature columns:", len(feature_columns))
    print("All feature values present: PASSED")
    print("Reference checks: PASSED")

    print("\n" + "=" * 60)
    print("TASK 4 COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Saved output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
