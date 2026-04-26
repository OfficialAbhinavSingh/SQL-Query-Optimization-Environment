import sys
sys.path.insert(0, '.')
from executor import get_executor
ex = get_executor()

print("=== Schema check ===")
# Task 3 - check what session_id LIKE 'sess_%' matches
r = ex.conn.execute("SELECT COUNT(*) FROM events WHERE session_id LIKE 'sess_%'").fetchone()
print(f"Task3: session_id LIKE 'sess_%' matches: {r[0]} / 1,000,000")

r2 = ex.conn.execute("SELECT COUNT(*) FROM events WHERE event_type = 'purchase'").fetchone()
print(f"Task3: event_type = 'purchase': {r2[0]}")

print()
# Task 4 original result shape
r3 = ex.conn.execute("""
SELECT u.region, u.plan, COUNT(*) AS total_orders, SUM(o.total) AS revenue,
(SELECT AVG(total) FROM orders) AS global_avg,
(SELECT MAX(total) FROM orders WHERE status = 'completed') AS max_deal
FROM users u, orders o
WHERE u.id = o.customer_id
  AND o.status IN ('completed', 'shipped')
GROUP BY u.region, u.plan
""").fetchall()
print(f"Task4 original rows: {len(r3)}")
print(f"Task4 sample row: {r3[0]}")

print()
# Task 5 original - just check row count
r4 = ex.conn.execute("SELECT COUNT(*) FROM events").fetchone()
print(f"Task5 original returns: {r4[0]} rows (all events, no WHERE)")

print()
# Check exact columns in events
r5 = ex.conn.execute("DESCRIBE events").fetchall()
print(f"events columns: {r5}")

r6 = ex.conn.execute("DESCRIBE users").fetchall()
print(f"users columns: {r6}")

r7 = ex.conn.execute("DESCRIBE orders").fetchall()
print(f"orders columns: {r7}")
