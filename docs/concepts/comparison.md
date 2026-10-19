# Comparing routers: what they do and how

One row per router. Fill in a row when you add a paper. Numbers live in the [leaderboard](../../results/leaderboard.md); this page is about **principles**.

## Design axes

| axis | question | options |
|---|---|---|
| **decision time** | when is the model chosen? | before any generation (routing) · after a cheap answer (cascade) · after all answers (selection/ensemble) |
| **candidates** | how many models? | 2 (binary) · N (multi-model) |
| **target** | what does the router learn to predict? | win probability · per-model quality · difficulty · utility (quality − λ·cost) · preference |
| **training signal** | where do labels come from? | reference metric (BARTScore) · ground-truth correctness · LLM judge · human preference |
| **stochasticity** | does it model that LLM answers vary? | yes (uses k samples) · no |
| **router model** | what computes the decision? | TF-IDF/linear · kNN on embeddings · fine-tuned encoder · LLM · matrix factorization · bandit/RL |
| **cost control** | how is the trade-off set? | test-time threshold · λ in objective · hard budget |
| **inference overhead** | what extra compute per query? | encoder pass · cheap-model generation + check · … |

## Router cards

| router | decision time | candidates | learns | labels from | uses k samples | router model | knob | overhead |
|---|---|---|---|---|---|---|---|---|
| quality regression *(baseline)* | before generation | N | each model's quality | benchmark scores | mean over k | TF-IDF + ridge per model | λ | 1 linear pass |
| random *(baseline)* | before generation | 2 | – | – | – | – | mix ratio | none |
| oracle *(reference)* | after all answers (cheats) | N | – | test scores | – | – | λ | – |

## Observations so far

- **There is a large gap to the oracle** everywhere. On RouterBench the TF-IDF quality regressor reaches AIQ 0.72 against the oracle's 0.91, so better query representations are the obvious next lever.
