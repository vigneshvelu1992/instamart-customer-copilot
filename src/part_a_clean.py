
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"

CLEAN.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# TASK 3: CLEAN CUSTOMERS
# --------------------------------------------------

def clean_customers(customers):
    df = customers.copy()

    print("\nCleaning customers...")

    # Standardise city names
    df["city"] = (
        df["city"]
        .astype("string")
        .str.strip()
        .str.title()
    )

    # Standardise gender
    gender_raw = (
        df["gender"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    gender_mapping = {
        "male": "male",
        "m": "male",
        "man": "male",
        "female": "female",
        "f": "female",
        "woman": "female",
    }

    df["gender"] = gender_raw.map(gender_mapping).fillna("unknown")

    # Clean ages
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    invalid_age = ~df["age"].between(10, 100)
    df.loc[invalid_age, "age"] = float("nan")

    # Fill missing ages using city median
    city_medians = df.groupby("city")["age"].transform("median")
    df["age"] = df["age"].fillna(city_medians)

    if df["age"].isna().any():
        raise ValueError("Missing customer ages remain.")

    df["age"] = df["age"].round().astype(int)

    # Convert signup date
    df["signup_date"] = pd.to_datetime(
        df["signup_date"], errors="coerce"
    )

    if df["signup_date"].isna().any():
        raise ValueError("Invalid signup dates found.")

    # Validate city counts
    expected_cities = {
        "Pune": 227,
        "Delhi": 210,
        "Bengaluru": 197,
        "Chennai": 191,
        "Hyderabad": 191,
        "Mumbai": 184,
    }

    actual_cities = df["city"].value_counts().to_dict()

    assert actual_cities == expected_cities, (
        f"City counts mismatch: {actual_cities}"
    )

    # Validate gender counts
    expected_genders = {
        "female": 610,
        "male": 560,
        "unknown": 30,
    }

    actual_genders = df["gender"].value_counts().to_dict()

    assert actual_genders == expected_genders, (
        f"Gender counts mismatch: {actual_genders}"
    )

    assert df["age"].notna().all()
    assert df["age"].between(18, 60).all()

    print("Customer cleaning successful.")
    print(df["city"].value_counts())
    print(df["gender"].value_counts())
    print("Age range:", df["age"].min(), "-", df["age"].max())

    return df


# --------------------------------------------------
# CLEAN PRODUCTS
# --------------------------------------------------

def clean_products(products):
    df = products.copy()

    print("\nCleaning products...")

    if df["product_id"].isna().any():
        raise ValueError("Missing product IDs.")

    if df["product_id"].duplicated().any():
        raise ValueError("Duplicate product IDs.")

    df["price_inr"] = pd.to_numeric(
        df["price_inr"], errors="raise"
    )

    assert (df["price_inr"] > 0).all()

    print("Product cleaning successful.")

    return df


# --------------------------------------------------
# TASK 4: CLEAN ORDERS
# --------------------------------------------------

def clean_orders(orders, products):
    df = orders.copy()

    print("\nCleaning orders...")

    # Convert order timestamps
    df["order_ts"] = pd.to_datetime(
        df["order_ts"], errors="coerce"
    )

    if df["order_ts"].isna().any():
        raise ValueError("Invalid order timestamps.")

    # Remove exact duplicate rows
    duplicates_removed = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    # Keep latest record for each order ID
    df = df.sort_values(
        ["order_id", "order_ts"],
        kind="mergesort"
    )

    df = df.drop_duplicates(
        subset=["order_id"],
        keep="last"
    ).copy()

    # Standardise status
    df["status"] = (
        df["status"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    # Convert order amounts
    df["order_amount"] = pd.to_numeric(
        df["order_amount"], errors="coerce"
    )

    # Get product price and category
    product_lookup = products.set_index("product_id")

    price_lookup = product_lookup["price_inr"]
    category_lookup = product_lookup["category"]

    df["product_price"] = df["product_id"].map(price_lookup)
    df["product_category"] = df["product_id"].map(category_lookup)

    if df["product_price"].isna().any():
        raise ValueError("Orders contain unknown product IDs.")

    # Convert quantity and discount
    df["quantity"] = pd.to_numeric(
        df["quantity"], errors="raise"
    )

    df["discount_pct"] = pd.to_numeric(
        df["discount_pct"], errors="raise"
    )

    # Calculate missing order amounts
    calculated_amount = (
        df["product_price"]
        * df["quantity"]
        * (1 - df["discount_pct"] / 100)
    ).round(2)

    missing_amount = df["order_amount"].isna()

    df.loc[missing_amount, "order_amount"] = (
        calculated_amount.loc[missing_amount]
    )

    if df["order_amount"].isna().any():
        raise ValueError("Missing order amounts remain.")

    # Clean delivery minutes
    df["delivery_minutes"] = pd.to_numeric(
        df["delivery_minutes"], errors="coerce"
    )

    negative_delivery = df["delivery_minutes"] < 0

    df.loc[negative_delivery, "delivery_minutes"] = float("nan")

    # Calculate completed-order category medians
    completed_mask = df["status"].eq("completed")

    completed_orders = df.loc[completed_mask].copy()

    category_medians = (
        completed_orders
        .groupby("product_category")["delivery_minutes"]
        .median()
    )

    # Impute missing delivery times for completed orders
    missing_completed = (
        df["status"].eq("completed")
        & df["delivery_minutes"].isna()
    )

    missing_rows = df.loc[missing_completed].copy()

    imputed_values = missing_rows["product_category"].map(
        category_medians
    )

    df.loc[missing_completed, "delivery_minutes"] = (
        imputed_values.to_numpy()
    )

    # Check completed orders
    completed_mask = df["status"].eq("completed")

    if df.loc[completed_mask, "delivery_minutes"].isna().any():
        raise ValueError(
            "Completed orders still have missing delivery times."
        )

    # Cancelled orders must have missing delivery time
    cancelled_mask = df["status"].eq("cancelled")

    df.loc[cancelled_mask, "delivery_minutes"] = float("nan")

    # Calculate final reference values
    completed_mask = df["status"].eq("completed")
    cancelled_mask = df["status"].eq("cancelled")

    completed_revenue = df.loc[
        completed_mask, "order_amount"
    ].sum()

    # Validate results
    assert len(df) == 12309, (
        f"Unexpected order count: {len(df)}"
    )

    assert df["order_id"].nunique() == len(df)

    expected_status = {
        "completed": 11110,
        "cancelled": 1199,
    }

    actual_status = df["status"].value_counts().to_dict()

    assert actual_status == expected_status, (
        f"Status counts mismatch: {actual_status}"
    )

    assert df["order_amount"].notna().all()

    assert df.loc[
        completed_mask, "delivery_minutes"
    ].notna().all()

    assert df.loc[
        cancelled_mask, "delivery_minutes"
    ].isna().all()

    assert round(completed_revenue, 2) == 2998668.80, (
        f"Unexpected revenue: {completed_revenue:.2f}"
    )

    # Remove helper columns
    df = df.drop(
        columns=["product_price", "product_category"]
    )

    # Sort final orders
    df = df.sort_values(
        ["order_ts", "order_id"]
    ).reset_index(drop=True)

    print("Order cleaning successful.")
    print("Exact duplicates removed:", duplicates_removed)
    print("Final order count:", len(df))
    print("\nStatus counts:")
    print(df["status"].value_counts())
    print("Completed revenue:", round(completed_revenue, 2))

    return df


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

def main():

    print("Starting Instamart data cleaning...")

    # Read raw CSV files
    customers_raw = pd.read_csv(RAW / "customers.csv")
    products_raw = pd.read_csv(RAW / "products.csv")
    orders_raw = pd.read_csv(RAW / "orders.csv")

    # Clean each dataset
    customers = clean_customers(customers_raw)
    products = clean_products(products_raw)
    orders = clean_orders(orders_raw, products)

    # Save cleaned customers
    customers.to_csv(
        CLEAN / "customers.csv",
        index=False
    )

    # Save cleaned products
    products.to_csv(
        CLEAN / "products.csv",
        index=False
    )

    # Format order timestamps for SQLite
    orders["order_ts"] = orders["order_ts"].dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    # Save cleaned orders
    orders.to_csv(
        CLEAN / "orders.csv",
        index=False
    )

    print("\n----------------------------------")
    print("ALL CLEANING OPERATIONS COMPLETED")
    print("----------------------------------")

    print("\nClean files saved in:")
    print(CLEAN)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        raise
