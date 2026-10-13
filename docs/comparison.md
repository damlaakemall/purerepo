# Comparing routers: what they do and how

One row per router. Fill in a row when you add a paper. Numbers live in the [leaderboard](../results/leaderboard.md); this page is about **principles**.

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
| **Hybrid LLM** r_det | before generation | 2 | Pr[small ≥ large] | BARTScore, 1 sample | no | DeBERTa-v3-large / TF-IDF+LR | threshold τ | 1 encoder pass |
| **Hybrid LLM** r_prob | before generation | 2 | Pr[H(x) ≥ 0], soft | BARTScore, k samples | **yes** | same | τ | same |
| **Hybrid LLM** r_trans | before generation | 2 | Pr[H(x) ≥ −t*], soft, relaxed against class imbalance | BARTScore, k samples | **yes** | same | τ | same |
| quality regression *(baseline)* | before generation | N | each model's quality | benchmark scores | mean over k | TF-IDF + ridge per model | λ | 1 linear pass |
| random *(baseline)* | before generation | 2 | – | – | – | – | mix ratio | none |
| oracle *(reference)* | after all answers (cheats) | N | – | test scores | – | – | λ | – |

## Observations so far

- **Model gap matters more than the router.** On MixInstruct, alpaca-7B → vicuna-13B (close models) lets a router send about 55% of traffic to the small model at under 1% quality loss. flan-t5-xxl → vicuna-13B (far apart) allows only about 5%. See [hybrid_llm.md](papers/hybrid_llm.md#7-our-results-tf-idf-router-cpu).
- **Soft labels help** when k > 1 (r_prob and r_trans beat r_det on synthetic data with k = 10). With k = 1 benchmarks (MixInstruct, RouterBench), r_prob ≡ r_det. This is one reason we build `ours_v1` with k = 10.
- **There is a large gap to the oracle** everywhere. On RouterBench the TF-IDF quality regressor reaches AIQ 0.72 against the oracle's 0.91, so better query representations are the obvious next lever.
