# LLM Routing — Fundamentals

A living reference for the concepts that come up across routing papers. Extend it as new papers are added.

## What is routing?

Given a query and a pool of LLMs with different costs and abilities, **choose which model answers it**. The goal is the best quality for a given budget, or the lowest cost for a given quality.

## Design axes (use these to compare papers)

| axis | options | Hybrid LLM |
|---|---|---|
| **When is the decision made?** | *pre-generation routing* (only sees query) · *cascade* (generate with cheap model, check, maybe escalate) · *post-hoc selection / ensembling* (generate with all, pick best) | pre-generation |
| **How many models?** | binary (small/large) · N models | binary |
| **What does the router predict?** | win probability · quality score per model · difficulty · cost-adjusted utility | Pr[small ≥ large], soft |
| **Training signal** | reference-based metric (BARTScore, ROUGE) · ground-truth accuracy · LLM-as-judge · human preferences (Arena) | BARTScore |
| **Router model** | similarity/kNN · matrix factorization · fine-tuned encoder (BERT/DeBERTa) · LLM classifier · bandit / RL | DeBERTa-v3-large |
| **Cost/quality control** | threshold at test time · cost penalty in objective · budget constraint | threshold τ |
| **Handles stochastic outputs?** | yes / no | yes (r_prob, r_trans) |

## Key concepts

- **Quality gap** `H(x) = q(S(x)) − q(L(x))`: how much better the small model does on query x. It is a random variable because decoding samples tokens.
- **Soft labels**: training targets in [0,1] instead of {0,1}. They preserve "how often" information. BCE works unchanged.
- **Class imbalance**: if one model almost always wins, binary labels collapse to one class. Fixes: relabel with a margin (Hybrid LLM's *t*), reweight the loss, or resample.
- **Threshold / operating point**: a scoring router plus a threshold gives a whole family of policies. Pick the threshold for your budget on validation data.
- **Cost advantage / quality drop / AUC**: see [`core/metrics.py`](../llm_routing/core/metrics.py). Always report the **random router** and the **oracle** as reference lines.
- **Calibration**: when you threshold on raw probabilities, they should mean what they say. Thresholding on *quantiles*, as `sweep` does, avoids depending on calibration.
- **Distribution shift**: a router trained on one query distribution may misroute on another, e.g. a new domain.

## Adding a new paper to this repo

1. Create `llm_routing/routers/<paper_name>/` and implement the `Router` interface (`fit`, `score`). Keep the convention **higher score = cheap model is fine**.
2. Reuse `core/classifiers.py` if the router is a text classifier; reuse `core/metrics.py` and `experiment.compare_routers` for evaluation.
3. If the paper needs new data (N models, preference labels…), add a loader in `llm_routing/data/` and, if necessary, extend `RoutingDataset`. For N-model routing, add a `MultiModelDataset` rather than breaking the binary one.
4. Write `docs/papers/<paper_name>.md` following the structure of [`hybrid_llm.md`](papers/hybrid_llm.md), and add a row to the comparison table in the README.
5. Add tests in `tests/`.
