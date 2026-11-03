
-- =====================================================
-- Q1: WHERE THEN HAVING
-- Customers whose completed orders above INR 500
-- add up to more than INR 8,000.
-- =====================================================

-- Q1
SELECT
    o.customer_id,
    c.city,
    COUNT(*) AS qualifying_order_count,
    ROUND(SUM(o.order_amount), 2) AS qualifying_spend
FROM orders AS o
JOIN customers AS c
    ON o.customer_id = c.customer_id
WHERE
    o.status = 'completed'
    AND o.order_amount > 500
GROUP BY
    o.customer_id,
    c.city
HAVING
    SUM(o.order_amount) > 8000
ORDER BY
    qualifying_spend DESC;


-- =====================================================
-- Q2: THREE-TABLE JOIN
-- Completed revenue by city and product category.
-- =====================================================

-- Q2
SELECT
    c.city,
    p.category,
    ROUND(SUM(o.order_amount), 2) AS revenue
FROM orders AS o
JOIN customers AS c
    ON o.customer_id = c.customer_id
JOIN products AS p
    ON o.product_id = p.product_id
WHERE
    o.status = 'completed'
GROUP BY
    c.city,
    p.category
ORDER BY
    c.city,
    p.category;


-- =====================================================
-- Q3: SCALAR SUBQUERY
-- Products priced above average catalogue price.
-- =====================================================

-- Q3
SELECT
    product_id,
    product_name,
    category,
    price_inr
FROM products
WHERE price_inr > (
    SELECT AVG(price_inr)
    FROM products
)
ORDER BY
    price_inr DESC;


-- =====================================================
-- Q4: LEFT JOIN
-- Customers who have no completed orders.
-- =====================================================

-- Q4
SELECT
    c.customer_id,
    c.city,
    c.gender,
    c.age
FROM customers AS c
LEFT JOIN orders AS o
    ON c.customer_id = o.customer_id
    AND o.status = 'completed'
WHERE
    o.order_id IS NULL
ORDER BY
    c.customer_id;


-- =====================================================
-- Q5: COMMON TABLE EXPRESSIONS (CTEs)
-- Customers spending more than average customer spend.
-- =====================================================

-- Q5
WITH customer_spend AS (
    SELECT
        customer_id,
        SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
average_spend AS (
    SELECT
        AVG(total_spend) AS avg_spend
    FROM customer_spend
)
SELECT
    cs.customer_id,
    c.city,
    ROUND(cs.total_spend, 2) AS total_spend,
    ROUND(a.avg_spend, 2) AS average_spend
FROM customer_spend AS cs
JOIN customers AS c
    ON cs.customer_id = c.customer_id
CROSS JOIN average_spend AS a
WHERE
    cs.total_spend > a.avg_spend
ORDER BY
    cs.total_spend DESC;


-- =====================================================
-- Q6: WINDOW FUNCTIONS
-- Rank customers by completed spend within each city.
-- =====================================================

-- Q6
WITH customer_spend AS (
    SELECT
        c.city,
        o.customer_id,
        SUM(o.order_amount) AS total_spend
    FROM orders AS o
    JOIN customers AS c
        ON o.customer_id = c.customer_id
    WHERE o.status = 'completed'
    GROUP BY
        c.city,
        o.customer_id
)
SELECT
    city,
    customer_id,
    ROUND(total_spend, 2) AS total_spend,

    RANK() OVER (
        PARTITION BY city
        ORDER BY total_spend DESC
    ) AS spend_rank,

    DENSE_RANK() OVER (
        PARTITION BY city
        ORDER BY total_spend DESC
    ) AS dense_spend_rank,

    ROW_NUMBER() OVER (
        PARTITION BY city
        ORDER BY total_spend DESC
    ) AS row_number

FROM customer_spend
ORDER BY
    city,
    spend_rank,
    customer_id;


-- =====================================================
-- Q7: RUNNING TOTAL
-- Daily completed revenue and cumulative revenue.
-- =====================================================

-- Q7
WITH daily_revenue AS (
    SELECT
        SUBSTR(order_ts, 1, 10) AS day,
        SUM(order_amount) AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY SUBSTR(order_ts, 1, 10)
)
SELECT
    day,
    ROUND(revenue, 2) AS daily_revenue,

    ROUND(
        SUM(revenue) OVER (
            ORDER BY day
            ROWS BETWEEN UNBOUNDED PRECEDING
            AND CURRENT ROW
        ),
        2
    ) AS cumulative_revenue

FROM daily_revenue
ORDER BY day;
