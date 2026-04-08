from typing import Dict, Any, List

TASKS: Dict[str, Dict[str, Any]] = {

    # ──────────────────────────────────────────────────────────────────
    # TASK 1 — EASY: Basic Query Anti-pattern Detection
    # ──────────────────────────────────────────────────────────────────
    "task_1_basic_antipatterns": {
        "task_id": "task_1_basic_antipatterns",
        "task_name": "Basic SQL Anti-pattern Detection",
        "task_description": (
            "Analyze the SQL query below for common anti-patterns that hurt performance. "
            "Identify issues such as: SELECT *, missing WHERE clauses causing full table scans, "
            "implicit type conversions, and non-SARGable predicates that prevent index usage. "
            "For each issue, report: issue_type, line number, description, severity (critical|high|medium|low), and a suggested fix."
        ),
        "difficulty": "easy",
        "dialect": "postgresql",
        "max_steps": 3,
        "schema_info": """\
Table: orders (id SERIAL PK, customer_id INT FK, status VARCHAR(20), total DECIMAL(10,2), created_at TIMESTAMPTZ)
Index: idx_orders_customer_id ON orders(customer_id)
Index: idx_orders_created_at ON orders(created_at)
Table size: ~5 million rows
""",
        "sql_query": """\
-- Fetch recent orders for reporting dashboard
SELECT *
FROM orders
WHERE CAST(customer_id AS TEXT) = '12345'
  AND YEAR(created_at) = 2024;
""",
        "ground_truth_issues": [
            {
                "type": "select_star",
                "line": 2,
                "keywords": ["select *", "select star", "all columns", "specific columns", "unnecessary columns", "bandwidth"]
            },
            {
                "type": "non_sargable_predicate",
                "line": 4,
                "keywords": ["cast", "convert", "non-sargable", "sargable", "index", "function on column", "type conversion", "implicit"]
            },
            {
                "type": "non_sargable_predicate",
                "line": 5,
                "keywords": ["year(", "function on column", "non-sargable", "index", "date range", "between", "extract"]
            },
        ],
        "approved_expected": False,
    },

    # ──────────────────────────────────────────────────────────────────
    # TASK 2 — MEDIUM: N+1 Query and Join Optimization
    # ──────────────────────────────────────────────────────────────────
    "task_2_join_optimization": {
        "task_id": "task_2_join_optimization",
        "task_name": "N+1 Pattern & Join Optimization",
        "task_description": (
            "Review the SQL query below for join performance issues and N+1 query patterns. "
            "Identify: missing indexes on join columns, inefficient subquery patterns that could be CTEs or JOINs, "
            "correlated subqueries executing per-row, missing covering indexes, and cartesian join risks. "
            "For each issue, report issue_type, line, description, severity, and a specific fix."
        ),
        "difficulty": "medium",
        "dialect": "postgresql",
        "max_steps": 4,
        "schema_info": """\
Table: users (id SERIAL PK, email VARCHAR UNIQUE, tier VARCHAR(10), region VARCHAR(50), created_at TIMESTAMPTZ)
Table: orders (id SERIAL PK, user_id INT FK->users.id, product_id INT FK->products.id, amount DECIMAL, placed_at TIMESTAMPTZ, status VARCHAR(20))
Table: products (id SERIAL PK, name VARCHAR, category VARCHAR(50), price DECIMAL)
Table: order_items (id SERIAL PK, order_id INT FK->orders.id, product_id INT FK->products.id, qty INT, unit_price DECIMAL)
Indexes: users(id) PK, orders(user_id), products(id) PK
No index on: orders(product_id), orders(status), order_items(order_id)
Approximate sizes: users=500k rows, orders=10M rows, order_items=40M rows, products=50k rows
""",
        "sql_query": """\
SELECT
    u.email,
    u.tier,
    (SELECT COUNT(*) FROM orders o WHERE o.user_id = u.id) AS order_count,
    (SELECT SUM(o.amount) FROM orders o WHERE o.user_id = u.id AND o.status = 'completed') AS total_spent,
    (SELECT MAX(o.placed_at) FROM orders o WHERE o.user_id = u.id) AS last_order_date
FROM users u
WHERE u.region = 'US'
  AND u.created_at > '2023-01-01'
ORDER BY total_spent DESC
LIMIT 100;
""",
        "ground_truth_issues": [
            {
                "type": "correlated_subquery",
                "line": 4,
                "keywords": ["correlated", "subquery", "per row", "n+1", "repeated", "each user", "lateral", "join"]
            },
            {
                "type": "correlated_subquery",
                "line": 5,
                "keywords": ["correlated", "subquery", "per row", "n+1", "repeated", "each user", "lateral", "join"]
            },
            {
                "type": "correlated_subquery",
                "line": 6,
                "keywords": ["correlated", "subquery", "per row", "n+1", "repeated", "each user", "lateral", "join"]
            },
            {
                "type": "missing_index",
                "line": 8,
                "keywords": ["missing index", "no index", "region", "full scan", "index on region", "composite"]
            },
            {
                "type": "sort_without_index",
                "line": 10,
                "keywords": ["order by", "sort", "filesort", "index", "total_spent", "computed", "no index for sort"]
            },
        ],
        "approved_expected": False,
    },

    # ──────────────────────────────────────────────────────────────────
    # TASK 3 — HARD: Complex Aggregation & Window Function Audit
    # ──────────────────────────────────────────────────────────────────
    "task_3_advanced_optimization": {
        "task_id": "task_3_advanced_optimization",
        "task_name": "Advanced Query & Window Function Audit",
        "task_description": (
            "Perform a deep performance audit of the complex analytical SQL query below. "
            "Identify: missing partition/covering indexes for window functions, "
            "inefficient GROUP BY with HAVING that could be pre-filtered, "
            "implicit data type coercions preventing index usage, "
            "redundant subqueries or CTEs that materialize too early, "
            "missing query hints or planner directives, "
            "and lock contention risks from large aggregations on live tables. "
            "For each issue report: issue_type, line, severity (critical|high|medium|low), description, and a concrete fix."
        ),
        "difficulty": "hard",
        "dialect": "postgresql",
        "max_steps": 5,
        "schema_info": """\
Table: events (id BIGSERIAL PK, user_id INT, session_id UUID, event_type VARCHAR(50), properties JSONB, occurred_at TIMESTAMPTZ)
Table: sessions (id UUID PK, user_id INT, started_at TIMESTAMPTZ, ended_at TIMESTAMPTZ, device VARCHAR(30))
Table: users (id INT PK, plan VARCHAR(20), country VARCHAR(3), created_at DATE)
Indexes: events(user_id, occurred_at), events(session_id), sessions(user_id)
No index on: events(event_type), events(occurred_at) standalone, users(plan, country)
Table sizes: events=500M rows, sessions=50M rows, users=2M rows
Autovacuum lag: events table has ~10% dead tuples
""",
        "sql_query": """\
WITH user_sessions AS (
    SELECT
        e.user_id,
        e.session_id,
        COUNT(*) AS event_count,
        SUM(CASE WHEN e.event_type = 'purchase' THEN 1 ELSE 0 END) AS purchases,
        MIN(e.occurred_at) AS session_start,
        MAX(e.occurred_at) AS session_end
    FROM events e
    JOIN sessions s ON s.id = e.session_id
    WHERE e.occurred_at BETWEEN '2024-01-01' AND '2024-12-31'
      AND properties->>'plan' = 'premium'
    GROUP BY e.user_id, e.session_id
),
ranked_sessions AS (
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY purchases DESC, session_end DESC) AS rn,
        AVG(event_count) OVER (PARTITION BY user_id) AS avg_events_per_session
    FROM user_sessions
)
SELECT
    u.country,
    u.plan,
    AVG(rs.purchases) AS avg_purchases,
    COUNT(DISTINCT rs.user_id) AS active_users,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY rs.event_count) AS median_events
FROM ranked_sessions rs
JOIN users u ON u.id = rs.user_id
WHERE rs.rn = 1
  AND u.plan::text IN ('premium', 'enterprise')
GROUP BY u.country, u.plan
HAVING COUNT(DISTINCT rs.user_id) > 10
ORDER BY avg_purchases DESC;
""",
        "ground_truth_issues": [
            {
                "type": "json_extraction_kills_index",
                "line": 10,
                "keywords": ["jsonb", "properties->", "arrow", "json", "index", "expression index", "gin", "no index", "json field"]
            },
            {
                "type": "redundant_cte_materialization",
                "line": 1,
                "keywords": ["cte", "materialize", "materialized", "inline", "common table expression", "scan twice", "performance"]
            },
            {
                "type": "window_function_missing_index",
                "line": 16,
                "keywords": ["row_number", "window", "partition", "index", "sort", "covering index", "partition by user_id"]
            },
            {
                "type": "implicit_cast_prevents_index",
                "line": 28,
                "keywords": ["cast", "::text", "implicit", "coerce", "index", "type cast", "data type", "prevent"]
            },
            {
                "type": "vacuum_bloat_risk",
                "line": 8,
                "keywords": ["vacuum", "dead tuple", "bloat", "autovacuum", "table bloat", "live rows", "500M", "performance"]
            },
            {
                "type": "having_without_pre_filter",
                "line": 30,
                "keywords": ["having", "group by", "pre-filter", "where", "filter before", "aggregate", "subquery push"]
            },
        ],
        "approved_expected": False,
    },
}


def get_task_list() -> List[Dict[str, Any]]:
    return [
        {
            "task_id": t["task_id"],
            "task_name": t["task_name"],
            "difficulty": t["difficulty"],
            "description": t["task_description"],
            "action_schema": {
                "suggestions": "List of {issue_type: str, line: int, description: str, severity: str, fix: str}",
                "optimized_query": "str — rewritten SQL query with improvements",
                "summary": "str — overall analysis summary",
                "estimated_improvement": "str — expected performance gain",
                "approved": "bool — whether query is already optimal (True) or not (False)"
            }
        }
        for t in TASKS.values()
    ]
