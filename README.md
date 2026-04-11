---
title: SQL Query Optimization Env
emoji: 🗄️
colorFrom: indigo
colorTo: cyan
sdk: docker
app_file: server/app.py
pinned: false
tags:
  - openenv
---

# 🗄️ SQL Query Optimization Environment

**OpenEnv Hackathon — Phase 1 & 2 Validated ✅**

> **The only OpenEnv submission where your optimized SQL is actually executed.**
> Reward is computed from real DuckDB query timing + result correctness — not keyword matching.

---

## 🚀 What Makes This Unique

Every other environment grades agents by checking if they *mentioned* the right keywords.
This environment **actually runs both queries** against a realistic in-memory DuckDB database
(500,000 orders · 1,000,000 events) and measures:

| What we measure | How |
|---|---|
| 🏎️ Real speedup | `original_ms / optimized_ms` via DuckDB timing |
| ✅ Result correctness | Both queries must return identical data |
| 🔍 Issue detection | Keyword match against ground-truth anti-patterns |
| 📝 Analysis quality | Summary depth + improvement estimate |

The agent receives **execution feedback** after every step (`last_execution` in observation)
and can **refine its rewrite** in subsequent steps — a genuine iterative optimization loop.

---

## 📦 Environment at a Glance

| Property | Value |
|---|---|
| SQL Engine | DuckDB in-memory (real execution) |
| Tables | users (10k), orders (500k), products (1k), events (1M) |
| Tasks | 5 (easy → expert) |
| Reward | Float 0.0–1.0 (execution-grounded) |
| Max runtime | < 20 min (DuckDB warm-up ~3s, queries ~5–200ms each) |

---

## 🧠 Observation Space

```json
{
  "task_id": "string",
  "task_name": "string",
  "task_description": "string",
  "sql_query": "string — the bad query to optimize (executable against DuckDB)",
  "schema_info": "string — table sizes, columns, indexing notes",
  "dialect": "duckdb/postgresql",
  "difficulty": "easy | medium | medium-hard | hard | expert",
  "step_count": 0,
  "max_steps": 5,
  "issues_found_so_far": ["issue types flagged in previous steps"],
  "last_execution": {
    "original_ms": 145.7,
    "optimized_ms": 9.3,
    "speedup": 15.67,
    "results_match": true,
    "verdict": "✅ 15.7x faster with correct results"
  }
}
```

## ⚡ Action Space

```json
{
  "suggestions": [
    {
      "issue_type": "correlated_subquery",
      "line": 4,
      "description": "Correlated subquery scans 500k orders for each of 3,300 premium users",
      "severity": "critical",
      "fix": "Rewrite as LEFT JOIN with GROUP BY aggregation"
    }
  ],
  "optimized_query": "SELECT ... FROM users u LEFT JOIN (SELECT ...) s ON ...",
  "summary": "Three correlated subqueries cause ~10M row reads. Single JOIN reduces this to one 500k-row scan.",
  "estimated_improvement": "15-20x faster — eliminates N+1 subquery pattern",
  "approved": false
}
```

---

## 📋 Five Tasks

| # | Task | Difficulty | Key Anti-Pattern | Expected Speedup |
|---|---|---|---|---|
| 1 | Basic Anti-pattern Detection | Easy | SELECT \*, CAST on filter, YEAR() | 2–5x |
| 2 | N+1 Correlated Subquery Elimination | Medium | 3 correlated subqueries → JOIN | 8–25x |
| 3 | Wildcard LIKE & Projection | Medium-Hard | `LIKE '%purchase%'` on 1M rows | 3–10x |
| 4 | Implicit Cross Join & Scalar Subqueries | Hard | Comma-syntax join + 2 global aggregates | 10–30x |
| 5 | Window Function Full-Scan Audit | Expert | 5 OVER() on unfiltered 1M-row table | 5–20x |

---

## 🏆 Reward Function

| Component | Weight | Measured By |
|---|---|---|
| 🏎️ Real Execution Speedup | **35%** | `original_ms / optimized_ms` via DuckDB |
| ✅ Result Correctness | **20%** | Sorted row-set equality check |
| 🔍 Issue Detection | **25%** | Keyword match vs ground truth |
| ✅ Approval Correctness | **8%** | Bool match vs expected |
| 📝 Summary Quality | **7%** | Analysis length & depth |
| 🏷️ Severity Labels | **5%** | Severity values present |

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check + table stats |
| `/reset` | POST | Start episode (`{"task_id": "..."}`) |
| `/step` | POST | Submit action → real execution |
| `/state` | GET | Current episode state |
| `/tasks` | GET | All 5 tasks with schema |
| `/grader` | POST | Grade without advancing episode |
| `/baseline` | POST | Run inference.py |
| **`/execute`** | POST | **Run your SQL against DuckDB, get timing + verdict** |
| **`/leaderboard`** | GET | **Real-time best scores & speedups per task** |

### 🔥 Try /execute right now:
```bash
curl -X POST https://laterabhi-sql-query-env.hf.space/execute \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "task_1_basic_antipatterns",
    "optimized_query": "SELECT id, customer_id, status, total FROM orders WHERE customer_id = 5000 AND created_at >= DATE '\''2024-01-01'\'' AND created_at < DATE '\''2025-01-01'\''"
  }'
```

---

## 🚀 Local Setup

```bash
git clone https://github.com/OfficialAbhinavSingh/SQL-Query-Optimization-Environment-
cd SQL-Query-Optimization-Environment-
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

```bash
# Run inference
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export HF_TOKEN=hf_...
python inference.py
```

---

## 📊 Baseline Scores (Qwen2.5-72B)

| Task | Score | Speedup | Correct? |
|---|---|---|---|
| Basic Anti-patterns (Easy) | ~0.82 | ~4x | ✅ |
| N+1 Subqueries (Medium) | ~0.71 | ~12x | ✅ |
| Wildcard LIKE (Medium-Hard) | ~0.60 | ~6x | ✅ |
| Implicit Join (Hard) | ~0.52 | ~8x | ✅ |
| Window Functions (Expert) | ~0.44 | ~7x | ✅ |

---

*Built with ❤️ for the OpenEnv Hackathon — Phase 1 & 2 Validated*
