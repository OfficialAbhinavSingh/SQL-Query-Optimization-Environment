"""
executor.py — DuckDB In-Memory SQL Execution Engine
=====================================================
The core innovation of this environment: instead of keyword-matching
heuristics, we ACTUALLY execute both the original and optimized queries
against realistic synthetic data and measure real performance differences.

Tables populated:
  users    — 10,000 rows
  orders   — 500,000 rows
  products —  1,000 rows
  events   — 1,000,000 rows
"""

import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import duckdb

_instance: Optional["QueryExecutor"] = None
_lock = threading.Lock()


class QueryExecutor:
    """
    Runs SQL against an in-memory DuckDB database with realistic
    synthetic data.  Provides execution timing, result correctness
    checks, and EXPLAIN plans — all used by the reward function.
    """

    def __init__(self) -> None:
        self.conn = duckdb.connect(database=":memory:")
        self.conn.execute("SET threads=2")
        self._build_tables()

    # ── Schema Setup ─────────────────────────────────────────────────────

    def _build_tables(self) -> None:
        """Create and populate all four synthetic tables."""

        # users — 10k rows
        self.conn.execute("""
            CREATE TABLE users AS
            SELECT
                i                                                      AS id,
                'u' || i || '@mail.com'                                AS email,
                CASE i % 3
                    WHEN 0 THEN 'premium'
                    WHEN 1 THEN 'free'
                    ELSE 'enterprise' END                              AS tier,
                CASE i % 5
                    WHEN 0 THEN 'US'   WHEN 1 THEN 'EU'
                    WHEN 2 THEN 'IN'   WHEN 3 THEN 'UK'
                    ELSE 'AU' END                                      AS region,
                CASE i % 2 WHEN 0 THEN 'premium' ELSE 'basic' END     AS plan,
                DATE '2020-01-01' + CAST(i AS INTEGER)                 AS created_at
            FROM generate_series(1, 10000) t(i)
        """)

        # orders — 500k rows
        self.conn.execute("""
            CREATE TABLE orders AS
            SELECT
                i                                                      AS id,
                1 + (i % 10000)                                        AS customer_id,
                (i % 100) + 1                                          AS product_id,
                CASE i % 4
                    WHEN 0 THEN 'completed'  WHEN 1 THEN 'pending'
                    WHEN 2 THEN 'cancelled'  ELSE 'shipped' END        AS status,
                ROUND((i % 1000) * 1.5 + 49.99, 2)                   AS total,
                DATE '2023-01-01' + CAST(i % 730 AS INTEGER)          AS created_at
            FROM generate_series(1, 500000) t(i)
        """)

        # products — 1k rows
        self.conn.execute("""
            CREATE TABLE products AS
            SELECT
                i                                                      AS id,
                'Product_' || i                                        AS name,
                CASE i % 5
                    WHEN 0 THEN 'Electronics'  WHEN 1 THEN 'Clothing'
                    WHEN 2 THEN 'Food'         WHEN 3 THEN 'Books'
                    ELSE 'Sports' END                                  AS category,
                ROUND((i % 500) + 9.99, 2)                            AS price
            FROM generate_series(1, 1000) t(i)
        """)

        # events — 1M rows
        self.conn.execute("""
            CREATE TABLE events AS
            SELECT
                i                                                      AS id,
                1 + (i % 10000)                                        AS user_id,
                'sess_' || (i % 50000)                                 AS session_id,
                CASE i % 6
                    WHEN 0 THEN 'purchase'  WHEN 1 THEN 'view'
                    WHEN 2 THEN 'click'     WHEN 3 THEN 'signup'
                    WHEN 4 THEN 'logout'    ELSE 'search' END          AS event_type,
                DATE '2024-01-01' + CAST(i % 365 AS INTEGER)          AS occurred_at
            FROM generate_series(1, 1000000) t(i)
        """)

    # ── Execution helpers ─────────────────────────────────────────────────

    def _run(
        self, query: str, runs: int = 3
    ) -> Tuple[float, Optional[List], Optional[str]]:
        """
        Execute *query* up to *runs* times.
        Returns (median_ms, rows, error_or_None).
        """
        timings: List[float] = []
        rows: Optional[List] = None

        for _ in range(runs):
            try:
                t0 = time.perf_counter()
                rows = self.conn.execute(query).fetchall()
                timings.append((time.perf_counter() - t0) * 1000.0)
            except Exception as exc:
                return 99_999.0, None, str(exc)

        timings.sort()
        return round(timings[len(timings) // 2], 3), rows, None

    # ── Public API ────────────────────────────────────────────────────────

    def compare(self, original: str, optimized: str) -> Dict[str, Any]:
        """
        Execute both queries, measure real timing, check correctness.

        Returns a dict with:
          original_ms, optimized_ms, speedup,
          results_match, original_rows, optimized_rows,
          original_error, optimized_error, verdict
        """
        orig_ms, orig_rows, orig_err = self._run(original)
        opt_ms, opt_rows, opt_err = self._run(optimized)

        # ── Correctness: do both queries return the same data? ────────
        results_match = False
        if orig_rows is not None and opt_rows is not None:
            try:
                orig_s = sorted(str(r) for r in orig_rows)
                opt_s = sorted(str(r) for r in opt_rows)
                results_match = orig_s == opt_s
            except Exception:
                results_match = len(orig_rows) == len(opt_rows)

        # ── Speedup ratio ─────────────────────────────────────────────
        speedup = 1.0
        if opt_ms > 0 and orig_ms < 90_000:
            speedup = round(orig_ms / opt_ms, 3)

        # ── Human-readable verdict ────────────────────────────────────
        if opt_err:
            verdict = f"[FAIL] Optimized query error: {opt_err[:120]}"
        elif results_match and speedup >= 2.0:
            verdict = f"[OK] {speedup:.1f}x faster with correct results"
        elif results_match and speedup >= 1.0:
            verdict = f"[WARN] Correct results but only {speedup:.1f}x speedup -- dig deeper"
        elif not results_match and speedup >= 2.0:
            verdict = f"[WARN] {speedup:.1f}x faster but results don't match -- fix the logic"
        else:
            verdict = f"[FAIL] {speedup:.1f}x -- no meaningful improvement"

        return {
            "original_ms":     orig_ms,
            "optimized_ms":    opt_ms,
            "speedup":         speedup,
            "results_match":   results_match,
            "original_rows":   len(orig_rows) if orig_rows is not None else 0,
            "optimized_rows":  len(opt_rows) if opt_rows is not None else 0,
            "original_error":  orig_err,
            "optimized_error": opt_err,
            "verdict":         verdict,
        }

    def explain(self, query: str) -> str:
        """Return EXPLAIN output for a query."""
        try:
            rows = self.conn.execute(f"EXPLAIN {query}").fetchall()
            return "\n".join(str(r[1]) for r in rows)
        except Exception as exc:
            return f"EXPLAIN error: {exc}"

    @property
    def table_stats(self) -> Dict[str, int]:
        tables = ["users", "orders", "products", "events"]
        return {
            t: self.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in tables
        }


# ── Singleton accessor ────────────────────────────────────────────────────

def get_executor() -> QueryExecutor:
    """Return the process-level singleton (lazy init, thread-safe)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = QueryExecutor()
    return _instance
