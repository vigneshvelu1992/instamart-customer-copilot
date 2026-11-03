
from pathlib import Path
import re
import sqlite3
import pandas as pd

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "instamart.db"
SQL_PATH = ROOT / "sql" / "queries.sql"


# --------------------------------------------------
# LOAD SEVEN SQL QUERIES
# --------------------------------------------------

def load_queries():

    if not SQL_PATH.exists():
        raise FileNotFoundError(f"SQL file not found: {SQL_PATH}")

    sql_text = SQL_PATH.read_text(encoding="utf-8")

    # Split the file at markers such as -- Q1, -- Q2, etc.
    parts = re.split(
        r"(?m)^-- Q([1-7])\s*$",
        sql_text
    )

    queries = {}

    for index in range(1, len(parts), 2):
        query_number = int(parts[index])
        query_text = parts[index + 1].strip()

        if query_text:
            queries[query_number] = query_text

    if len(queries) != 7:
        raise ValueError(
            f"Expected 7 SQL queries, found {len(queries)}."
        )

    return queries


# --------------------------------------------------
# EXECUTE QUERIES
# --------------------------------------------------

def main():

    print("INSTAMART PART A - TASK 6")
    print("SQL ANALYSIS")
    print("=" * 50)

    if not DB_PATH.exists():
        raise FileNotFoundError(
            "Database not found. Run part_a_database.py first."
        )

    queries = load_queries()

    connection = sqlite3.connect(DB_PATH)

    try:

        results = {}

        for number in range(1, 8):

            print(f"\n{'=' * 50}")
            print(f"QUERY Q{number}")
            print("=" * 50)

            result = pd.read_sql_query(
                queries[number],
                connection
            )

            results[number] = result

            print("Rows returned:", len(result))
            print(result.to_string(index=False))

        # --------------------------------------------------
        # REFERENCE CHECKS
        # --------------------------------------------------

        print("\n")
        print("=" * 50)
        print("TASK 6 REFERENCE CHECKS")
        print("=" * 50)

        # Q1: 30 customers
        assert len(results[1]) == 30
        print("Q1 passed: 30 customers")

        # Q2: 30 city/category combinations
        assert len(results[2]) == 30

        q2_total = round(results[2]["revenue"].sum(), 2)
        assert q2_total == 2998668.80

        largest_q2 = results[2].sort_values(
            "revenue", ascending=False
        ).iloc[0]

        assert largest_q2["city"] == "Bengaluru"
        assert largest_q2["category"] == "Electronics"
        assert round(largest_q2["revenue"], 2) == 233909.45

        print("Q2 passed: 30 rows and correct revenue total")
        print("Q2 total revenue: INR", f"{q2_total:,.2f}")
        print(
            "Largest city/category:",
            largest_q2["city"],
            "/",
            largest_q2["category"],
            "- INR",
            f"{largest_q2['revenue']:,.2f}"
        )

        # Q3: 11 products above average price
        assert len(results[3]) == 11
        print("Q3 passed: 11 products")

        # Q4: 25 customers without completed orders
        assert len(results[4]) == 25
        print("Q4 passed: 25 customers")

        # Q5: 550 customers above average spend
        assert len(results[5]) == 550

        q5_average = round(
            results[5]["average_spend"].iloc[0], 2
        )

        assert q5_average == 2552.06

        print("Q5 passed: 550 customers")
        print("Average customer spend: INR", f"{q5_average:,.2f}")

        # Q6: Exactly 8 rows where RANK differs from ROW_NUMBER
        q6_differences = (
            results[6]["spend_rank"]
            != results[6]["row_number"]
        ).sum()

        assert q6_differences == 8

        print(
            "Q6 passed: 8 rows where RANK differs from ROW_NUMBER"
        )

        # Display top customer in each city
        top_customers = (
            results[6]
            .sort_values(["city", "row_number"])
            .groupby("city")
            .head(1)
        )

        print("\nTop customer in each city:")
        print(
            top_customers[
                ["city", "customer_id", "total_spend"]
            ].to_string(index=False)
        )

        # Q7: 183 daily records and correct final cumulative revenue
        assert len(results[7]) == 183

        final_revenue = round(
            results[7]["cumulative_revenue"].iloc[-1],
            2
        )

        assert final_revenue == 2998668.80

        print("\nQ7 passed: 183 daily records")
        print(
            "Final cumulative revenue: INR",
            f"{final_revenue:,.2f}"
        )

        print("\nALL SEVEN SQL QUERIES EXECUTED SUCCESSFULLY")

    finally:
        connection.close()


if __name__ == "__main__":
    main()
