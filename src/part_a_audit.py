
from pathlib import Path
import pandas as pd

# Project root is the parent of the src folder
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"


def report_table(name, df):
    """Print shape, data types, and missing values for every column."""
    print(f"\n{'=' * 65}")
    print(f"TABLE: {name}")
    print(f"Shape (rows, columns): {df.shape}")
    print("\nData types:")
    print(df.dtypes.to_string())

    # Treat whitespace-only strings as blank for the audit.
    missing = df.isna().copy()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        missing[col] = (
            df[col].isna()
            | df[col].astype("string").str.strip().eq("")
        )

    count = missing.sum()
    percent = (count / len(df) * 100).round(2) if len(df) else count

    summary = pd.DataFrame({
        "missing_count": count,
        "missing_percent": percent,
    })
    print("\nMissing values by column:")
    print(summary.to_string())


def main():
    customers = pd.read_csv(RAW / "customers.csv")
    products = pd.read_csv(RAW / "products.csv")
    orders = pd.read_csv(RAW / "orders.csv")

    # Preserve blank ticket text as an empty string.
    tickets = pd.read_csv(
        RAW / "tickets.csv",
        keep_default_na=False
    )

    for name, df in [
        ("customers.csv", customers),
        ("products.csv", products),
        ("orders.csv", orders),
        ("tickets.csv", tickets),
    ]:
        report_table(name, df)

    # Required order audit counts
    order_amount_num = pd.to_numeric(
        orders["order_amount"], errors="coerce"
    )
    delivery_num = pd.to_numeric(
        orders["delivery_minutes"], errors="coerce"
    )

    blank_delivery = (
        orders["delivery_minutes"].isna()
        | orders["delivery_minutes"].astype(str).str.strip().eq("")
    )

    print(f"\n{'=' * 65}")
    print("REQUIRED AUDIT CHECKPOINTS")

    order_checks = {
        "Raw order rows": len(orders),
        "Exact duplicate order rows": int(orders.duplicated().sum()),
        "Pending order rows": int(
            orders["status"].astype(str).str.strip().str.lower()
            .eq("pending").sum()
        ),
        "Distinct order IDs": int(orders["order_id"].nunique()),
        "Non-numeric order amounts": int(
            order_amount_num.isna().sum()
        ),
        "Negative delivery times": int((delivery_num < 0).sum()),
        "Blank delivery times": int(blank_delivery.sum()),
    }

    # Required customer audit counts
    age_num = pd.to_numeric(customers["age"], errors="coerce")
    blank_age = (
        customers["age"].isna()
        | customers["age"].astype(str).str.strip().eq("")
    )
    gender = customers["gender"].astype("string").str.strip()
    city = customers["city"].astype("string").str.strip()

    customer_checks = {
        "Blank ages": int(blank_age.sum()),
        "Ages outside 10-100": int(
            ((age_num < 10) | (age_num > 100)).sum()
        ),
        "Distinct raw city spellings": int(
            customers["city"].nunique()
        ),
        "Distinct non-blank raw gender spellings": int(
            gender[gender.notna() & gender.ne("")].nunique()
        ),
        "Blank genders": int(
            (gender.isna() | gender.eq("")).sum()
        ),
    }

    for label, value in {**order_checks, **customer_checks}.items():
        print(f"{label}: {value}")


if __name__ == "__main__":
    main()
