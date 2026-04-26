# Before / after — execution-grounded reward

| Task | Difficulty | Before (no SQL) | After (fallback) | Δ |
|------|------------|-----------------|------------------|---|
| Basic SQL Anti-pattern Detection | easy | 0.4500 | 0.8300 | +0.3800 |
| N+1 Correlated Subquery Elimination | medium | 0.4500 | 0.6900 | +0.2400 |
| Wildcard LIKE & Projection Optimization | medium-hard | 0.4500 | 0.6900 | +0.2400 |
| Implicit Cross Join & Scalar Subquery El | hard | 0.4500 | 0.6900 | +0.2400 |
| Window Function & Full-Scan Audit | expert | 0.4500 | 0.7500 | +0.3000 |

**Mean before:** 0.4500  
**Mean after:** 0.7300  
**Mean Δ:** +0.2800

_Before = non-empty suggestions but `optimized_query` empty — no speedup/correctness signal._