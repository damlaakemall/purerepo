# How routers are evaluated

Every router produces a list of **operating points** by turning its knob: a threshold for binary routers, λ for multi-model routers, a confidence level for cascades. For each point we measure on the **test split**:

- **quality**: mean realized quality of the returned answers
- **cost**: mean cost paid per query. For a cascade this includes every model it called.

Together these points form the router's **quality–cost curve**, which is plotted for every run (`curve.png`). The single models appear as points, and the random router and the oracle as dashed lines.

## Summary numbers ([`core/evaluation.py`](../../llm_routing/core/evaluation.py))

| metric | definition | better |
|---|---|---|
| **AIQ** | Average Improvement in Quality (RouterBench): area under the router's non-decreasing quality-vs-cost curve, with cost normalized over [cheapest, most expensive] model. Empty for a router with one operating point (a single point is not a curve) | ↑ |
| **cost to match best** | lowest cost at which the router reaches the best single model's quality, divided by that model's cost. Values below 1 mean savings; ∞ means it never reaches that quality | ↓ |
| **max quality** | best quality at any operating point. It can exceed every single model | ↑ |
| **small-model share @ ≤1% drop** *(binary)* | Hybrid LLM's headline number: the largest fraction of queries sent to the small model while quality stays within 1% of all-large. Read off the test sweep, so it is descriptive, not a deployable threshold | ↑ |
| **small-model share / drop @ val threshold** *(binary)* | the deployable version (Hybrid LLM App. A): the threshold is chosen on val (most traffic to the small model with ≤1% drop), frozen, and applied once to test | ↑ / ↓ |
| **quality drop @ X%** *(binary)* | quality loss relative to all-large when X% of queries go to the small model | ↓ |

## Reference lines

- **random** *(binary)*: random mix of the small and large model; it gives a straight line between the two. A learned router has to beat it.
- **oracle**: knows the realized test answers and picks the best (cheapest among ties) per query. It's the ceiling; the gap between a router and the oracle is the room left for improvement.
- **single models**: the cost and quality of always using one model.

## Reporting

Report mean ± std over **≥ 3 seeds**. Seeds change the router's initialization and training order; the data stays the same. Never compare numbers with different data fingerprints.
