
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean"
OUTPUTS = ROOT / "outputs"

OUTPUTS.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOAD CLEANED DATA
# --------------------------------------------------

def load_data():

    customers = pd.read_csv(CLEAN / "customers.csv")
    products = pd.read_csv(CLEAN / "products.csv")
    orders = pd.read_csv(CLEAN / "orders.csv")

    orders["order_ts"] = pd.to_datetime(
        orders["order_ts"], errors="coerce"
    )

    return customers, products, orders


# --------------------------------------------------
# TASK 7A: PANDAS CROSS-CHECK
# --------------------------------------------------

def pandas_cross_check(customers, products, orders):

    print("\n" + "=" * 60)
    print("TASK 7A - PANDAS CROSS-CHECK")
    print("=" * 60)

    # Only completed orders count as revenue.
    completed = orders.loc[
        orders["status"].eq("completed")
    ].copy()

    # Join orders, customers and products.
    analysis = (
        completed
        .merge(
            customers[["customer_id", "city"]],
            on="customer_id",
            how="left",
            validate="many_to_one"
        )
        .merge(
            products[["product_id", "category"]],
            on="product_id",
            how="left",
            validate="many_to_one"
        )
    )

    if analysis[["city", "category"]].isna().any().any():
        raise ValueError("Missing city or category after joining data.")

    # Build Q2 again using a pandas pivot table.
    pivot = pd.pivot_table(
        analysis,
        index="city",
        columns="category",
        values="order_amount",
        aggfunc="sum",
        fill_value=0,
        margins=True,
        margins_name="All"
    )

    print("\nQ2 PANDAS PIVOT TABLE")
    print(pivot.round(2).to_string())

    pandas_total = round(float(pivot.loc["All", "All"]), 2)

    print("\nPandas grand total: INR", f"{pandas_total:,.2f}")

    # Cross-check with the expected SQL Q2 total.
    expected_sql_total = 2998668.80

    assert pandas_total == expected_sql_total, (
        f"Pandas total {pandas_total} does not match "
        f"SQL total {expected_sql_total}"
    )

    print("Pandas and SQL revenue totals match.")

    # Group by city using named aggregation.
    city_summary = (
        analysis
        .groupby("city")
        .agg(
            total_revenue=("order_amount", "sum"),
            average_order_amount=("order_amount", "mean"),
            completed_order_count=("order_amount", "count")
        )
        .reset_index()
        .sort_values("city")
    )

    city_summary["total_revenue"] = (
        city_summary["total_revenue"].round(2)
    )

    city_summary["average_order_amount"] = (
        city_summary["average_order_amount"].round(2)
    )

    print("\nCOMPLETED REVENUE BY CITY")
    print(city_summary.to_string(index=False))

    # Check two reference cities.
    bengaluru = city_summary.loc[
        city_summary["city"].eq("Bengaluru")
    ].iloc[0]

    assert round(bengaluru["total_revenue"], 2) == 523643.40
    assert round(bengaluru["average_order_amount"], 2) == 276.47
    assert int(bengaluru["completed_order_count"]) == 1894

    mumbai = city_summary.loc[
        city_summary["city"].eq("Mumbai")
    ].iloc[0]

    assert round(mumbai["total_revenue"], 2) == 479058.50
    assert round(mumbai["average_order_amount"], 2) == 297.00
    assert int(mumbai["completed_order_count"]) == 1613

    print("\nCity reference checks passed.")

    return analysis, pivot, city_summary


# --------------------------------------------------
# TASK 7B: EXPLORATORY DATA ANALYSIS
# --------------------------------------------------

def exploratory_analysis(analysis):

    print("\n" + "=" * 60)
    print("TASK 7B - EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    amount = analysis["order_amount"].dropna()

    # Descriptive statistics.
    mean_value = amount.mean()
    median_value = amount.median()
    skew_value = amount.skew()

    q1 = amount.quantile(0.25)
    q2 = amount.quantile(0.50)
    q3 = amount.quantile(0.75)

    iqr = q3 - q1

    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    lower_outliers = int((amount < lower_fence).sum())
    upper_outliers = int((amount > upper_fence).sum())

    print("\nORDER AMOUNT STATISTICS")

    print(f"Mean: INR {mean_value:,.2f}")
    print(f"Median: INR {median_value:,.2f}")
    print(f"Skewness: {skew_value:.2f}")

    print(f"Q1: INR {q1:,.2f}")
    print(f"Q2 (Median): INR {q2:,.2f}")
    print(f"Q3: INR {q3:,.2f}")

    print(f"IQR: INR {iqr:,.2f}")

    print(f"Lower fence: INR {lower_fence:,.2f}")
    print(f"Upper fence: INR {upper_fence:,.2f}")

    print("Outliers below lower fence:", lower_outliers)
    print("Outliers above upper fence:", upper_outliers)

    # Correlation analysis.
    correlation_columns = [
        "order_amount",
        "discount_pct",
        "quantity",
        "delivery_minutes"
    ]

    correlation_data = analysis[correlation_columns].copy()

    for column in correlation_columns:
        correlation_data[column] = pd.to_numeric(
            correlation_data[column],
            errors="coerce"
        )

    correlation_matrix = correlation_data.corr()

    print("\nCORRELATION MATRIX")
    print(correlation_matrix.round(3).to_string())

    # Find the strongest absolute off-diagonal correlation.
    absolute_correlations = correlation_matrix.abs()

    for column in correlation_columns:
        absolute_correlations.loc[column, column] = 0

    strongest_pair = (
        absolute_correlations
        .stack()
        .idxmax()
    )

    strongest_value = correlation_matrix.loc[
        strongest_pair[0],
        strongest_pair[1]
    ]

    print("\nSTRONGEST CORRELATION PAIR")
    print(strongest_pair[0], "and", strongest_pair[1])
    print(f"Correlation: {strongest_value:.3f}")

    # Reference checks.
    assert round(mean_value, 2) == 269.91
    assert round(median_value, 2) == 135.00
    assert round(skew_value, 2) == 4.61

    assert round(q1, 2) == 64.80
    assert round(q3, 2) == 297.00

    assert round(upper_fence, 2) == 645.30
    assert upper_outliers == 897
    assert lower_outliers == 0

    assert strongest_pair == (
        "order_amount",
        "discount_pct"
    ) or strongest_pair == (
        "discount_pct",
        "order_amount"
    )

    assert round(strongest_value, 3) == -0.095

    print("\nEDA reference checks passed.")

    return {
        "mean": mean_value,
        "median": median_value,
        "skewness": skew_value,
        "q1": q1,
        "q3": q3,
        "lower_fence": lower_fence,
        "upper_fence": upper_fence,
        "lower_outliers": lower_outliers,
        "upper_outliers": upper_outliers,
        "correlation_pair": strongest_pair,
        "correlation_value": strongest_value
    }


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

def main():

    print("INSTAMART CUSTOMER INTELLIGENCE")
    print("PART A - TASK 7")

    customers, products, orders = load_data()

    analysis, pivot, city_summary = pandas_cross_check(
        customers,
        products,
        orders
    )

    statistics = exploratory_analysis(analysis)

    # Save city summary for reporting.
    city_summary.to_csv(
        OUTPUTS / "part_a_city_summary.csv",
        index=False
    )

    print("\n" + "=" * 60)
    print("TASK 7 COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("\nSaved output:")
    print(OUTPUTS / "part_a_city_summary.csv")


if __name__ == "__main__":
    main()
