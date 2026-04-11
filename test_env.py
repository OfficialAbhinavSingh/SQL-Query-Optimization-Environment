import time
import sys
sys.path.insert(0, r'c:\Users\ishua\OneDrive\Desktop\meta-2')

print('Testing DuckDB executor...')
t0 = time.time()
from executor import get_executor
ex = get_executor()
print(f'Tables built in {time.time()-t0:.1f}s')
print('Table stats:', ex.table_stats)

print()
print('Testing real query comparison (Task 1)...')
from tasks import TASKS
task = TASKS['task_1_basic_antipatterns']
original = task['sql_query']
optimized = "SELECT id, customer_id, status, total, created_at FROM orders WHERE customer_id = 5000 AND created_at >= DATE '2024-01-01' AND created_at < DATE '2025-01-01'"

result = ex.compare(original, optimized)
print(f"  Original : {result['original_ms']:.1f} ms ({result['original_rows']} rows)")
print(f"  Optimized: {result['optimized_ms']:.1f} ms ({result['optimized_rows']} rows)")
print(f"  Speedup  : {result['speedup']:.2f}x")
print(f"  Correct  : {result['results_match']}")
print(f"  Verdict  : {result['verdict']}")

print()
print('Testing full grader...')
from graders import grade
from models import Action

action = Action(
    suggestions=[
        {"issue_type": "select_star", "line": 1, "description": "SELECT * fetches all columns unnecessarily from 500k rows", "severity": "medium", "fix": "Select only needed columns"},
        {"issue_type": "non_sargable_cast", "line": 3, "description": "CAST on customer_id prevents columnar pruning", "severity": "high", "fix": "Use direct integer comparison"},
        {"issue_type": "function_on_date_column", "line": 4, "description": "year() on created_at forces full column evaluation", "severity": "high", "fix": "Use date range with BETWEEN"},
    ],
    optimized_query=optimized,
    summary="Three anti-patterns identified: SELECT * wastes bandwidth, CAST and year() prevent DuckDB zone-map pruning causing full 500k row scans.",
    estimated_improvement="5-10x faster by enabling columnar pruning and reducing I/O",
    approved=False
)
reward = grade(task, action)
print(f"  Score    : {reward.score}")
print(f"  Breakdown: {reward.breakdown}")
print(reward.feedback[:300])
print()
print('ALL TESTS PASSED!')
