
from pathlib import Path
import sqlite3
import pandas as pd

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean"
DB_PATH = ROOT / "data" / "instamart.db"

DB_PATH.parent.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOAD CLEANED CSV FILES
# --------------------------------------------------

def load_clean_data():

    required_files = [
        CLEAN / "customers.csv",
        CLEAN / "products.csv",
        CLEAN / "orders.csv",
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required cleaned file not found: {file_path}\n"
                "Run part_a_clean.py first."
            )

    customers = pd.read_csv(CLEAN / "customers.csv")
    products = pd.read_csv(CLEAN / "products.csv")
    orders = pd.read_csv(CLEAN / "orders.csv")

    return customers, products, orders


# --------------------------------------------------
# CREATE SQLITE DATABASE
# --------------------------------------------------

def create_database(customers, products, orders):

    print("\nCreating SQLite database...")

    # Remove the previous database to avoid stale data.
    if DB_PATH.exists():
        DB_PATH.unlink()

    connection = sqlite3.connect(DB_PATH)

    try:
        customers.to_sql(
            "customers",
            connection,
            if_exists="replace",
            index=False
        )

        products.to_sql(
            "products",
            connection,
            if_exists="replace",
            index=False
        )

        orders.to_sql(
            "orders",
            connection,
            if_exists="replace",
            index=False
        )

        connection.commit()

    finally:
        connection.close()

    print("Database created:", DB_PATH)


# --------------------------------------------------
# VALIDATE DATABASE
# --------------------------------------------------

def validate_database():

    print("\nRunning database validation...")

    connection = sqlite3.connect(DB_PATH)

    try:
        # Check 1: Customer count
        customer_count = pd.read_sql_query(
            "SELECT COUNT(*) AS total FROM customers",
            connection
        ).iloc[0]["total"]

        assert customer_count == 1200, (
            f"Customer count incorrect: {customer_count}"
        )

        print("Checkpoint 1 passed: 1,200 customers")

        # Check 2: Product count
        product_count = pd.read_sql_query(
            "SELECT COUNT(*) AS total FROM products",
            connection
        ).iloc[0]["total"]

        assert product_count == 40, (
            f"Product count incorrect: {product_count}"
        )

        print("Checkpoint 2 passed: 40 products")

        # Check 3: Order count
        order_count = pd.read_sql_query(
            "SELECT COUNT(*) AS total FROM orders",
            connection
        ).iloc[0]["total"]

        assert order_count == 12309, (
            f"Order count incorrect: {order_count}"
        )

        print("Checkpoint 3 passed: 12,309 orders")

        # Check 4: Unique order IDs
        unique_orders = pd.read_sql_query(
            "SELECT COUNT(DISTINCT order_id) AS total FROM orders",
            connection
        ).iloc[0]["total"]

        assert unique_orders == 12309

        print("Checkpoint 4 passed: Unique order IDs")

        # Check 5: Order status counts
        status_data = pd.read_sql_query(
            """
            SELECT status, COUNT(*) AS total
            FROM orders
            GROUP BY status
            """,
            connection
        )

        actual_status = dict(
            zip(status_data["status"], status_data["total"])
        )

        expected_status = {
            "completed": 11110,
            "cancelled": 1199,
        }

        assert actual_status == expected_status, (
            f"Status mismatch: {actual_status}"
        )

        print("Checkpoint 5 passed: Order status counts")

        # Check 6: No missing order amounts
        missing_amounts = pd.read_sql_query(
            """
            SELECT COUNT(*) AS total
            FROM orders
            WHERE order_amount IS NULL
            """,
            connection
        ).iloc[0]["total"]

        assert missing_amounts == 0

        print("Checkpoint 6 passed: No missing order amounts")

        # Check 7: Completed orders have delivery times
        missing_completed_delivery = pd.read_sql_query(
            """
            SELECT COUNT(*) AS total
            FROM orders
            WHERE status = 'completed'
            AND delivery_minutes IS NULL
            """,
            connection
        ).iloc[0]["total"]

        assert missing_completed_delivery == 0

        print("Checkpoint 7 passed: Completed delivery times")

        # Check 8: Cancelled orders have missing delivery times
        missing_cancelled_delivery = pd.read_sql_query(
            """
            SELECT COUNT(*) AS total
            FROM orders
            WHERE status = 'cancelled'
            AND delivery_minutes IS NULL
            """,
            connection
        ).iloc[0]["total"]

        assert missing_cancelled_delivery == 1199

        print("Checkpoint 8 passed: Cancelled delivery times")

        # Check 9: Completed order revenue
        revenue = pd.read_sql_query(
            """
            SELECT ROUND(SUM(order_amount), 2) AS total_revenue
            FROM orders
            WHERE status = 'completed'
            """,
            connection
        ).iloc[0]["total_revenue"]

        assert round(revenue, 2) == 2998668.80, (
            f"Revenue mismatch: {revenue}"
        )

        print("Checkpoint 9 passed: Completed revenue")

        # Check 10: No pending orders
        pending_orders = pd.read_sql_query(
            """
            SELECT COUNT(*) AS total
            FROM orders
            WHERE status = 'pending'
            """,
            connection
        ).iloc[0]["total"]

        assert pending_orders == 0

        print("Checkpoint 10 passed: No pending orders")

        # Check 11: Verify database tables
        tables = pd.read_sql_query(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            """,
            connection
        )["name"].tolist()

        assert set(["customers", "products", "orders"]).issubset(
            set(tables)
        )

        print("Checkpoint 11 passed: All three tables exist")

        print("\nALL CHECKPOINTS PASSED")

    finally:
        connection.close()


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

def main():

    print("INSTAMART SQLITE DATABASE CREATION")

    customers, products, orders = load_clean_data()

    create_database(customers, products, orders)

    validate_database()


if __name__ == "__main__":
    main()
