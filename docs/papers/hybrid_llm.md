# Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing

**Ding, Mallick, Wang, Sim, Mukherjee, Rühle, Lakshmanan, Awadallah — ICLR 2024**
Paper: <https://arxiv.org/abs/2404.14618> · Code here: [`llm_routing/routers/hybrid_llm/`](../../llm_routing/routers/hybrid_llm/)

---

## 1. The problem

Large LLMs give good answers but are expensive, so they run in the cloud. Small LLMs are cheap and can even run on edge devices, but their answers are usually worse.

The paper's observation: the small model isn't worse **on every query**. For many easy queries (greetings, simple facts, short rewrites) it answers just as well, and sometimes better. If we could tell in advance which queries are easy, we could send only the hard ones to the large model.

```
            ┌──────────── p_w(x) ≥ τ ────────────▶ SMALL model  (cheap)
 query x ──▶│ router p_w │
            └──────────── p_w(x) < τ ────────────▶ LARGE model  (expensive)
```

**Key constraint:** the router decides *before* any model answers. It only sees the query text, never the responses. That makes this **routing**, not a **cascade**, where you would run the small model first and then decide whether to escalate.

## 2. The quality gap H(x)

| symbol | meaning |
|---|---|
| `S(x)`, `L(x)` | responses of the small / large model to query `x` |
| `q(·)` | quality of a response. The paper uses **BARTScore** against a reference answer |
| `H(x) = q(S(x)) − q(L(x))` | **quality gap**. ≥ 0 means the small model did at least as well |

`H(x)` is a **random variable**. LLM decoding samples tokens, so asking the same model the same question twice can give answers of different quality. This one fact drives the whole paper.

## 3. Three routers, one classifier, three kinds of labels

Every variant uses the **same model**: a DeBERTa-v3-large encoder with a single sigmoid output `p_w(x) ∈ [0,1]`, trained with binary cross-entropy. The variants differ only in the **training labels**:

### r_det — deterministic
```
y(x) = 1[ q(S(x)) ≥ q(L(x)) ]        (one sampled response from each model)
```
Simple, but it ignores randomness. A query whose small-model answer is better 49% of the time gets the same label 0 as a query where it is never better. The labels are noisy.

### r_prob — probabilistic
```
y(x) = Pr[ H(x) ≥ 0 ]                 (estimated from k=10 samples per model)
```
The labels are now **soft** probabilities. "Small is better 49% of the time" becomes 0.49 instead of 0, which is a much richer training signal. BCE accepts soft targets directly.
Implemented in `gap_probability(...)`. We compare all k×k sample pairs by default, or k aligned pairs with `estimator="paired"`.

### r_trans — transformed (relaxed)
```
y_t(x) = Pr[ H(x) ≥ −t ]              "small is at most t worse than large"
t*     = argmax_t  (1/N²) Σ_i Σ_j | y_t(x_i) − y_t(x_j) |
```
**Problem it fixes:** when the small model is much weaker (e.g. FLAN-T5 800M vs Llama-2 13B), `Pr[H ≥ 0] ≈ 0` for almost every query. Nearly all labels are ~0, and the classifier learns a constant. This is extreme **class imbalance**.

**Fix:** relax the requirement by `t`. As `t` grows, labels move from all-0 (t=0) to all-1 (t=∞). Both extremes carry no information. `t*` is chosen to make the labels **as spread out as possible**, measured by the average pairwise absolute difference. This needs no extra hyperparameter tuning and no validation set.
Implemented in `label_spread(...)`, an O(N log N) version of the double sum, and `find_relaxation(...)`, a grid search.

## 4. Test time: one router, any cost/quality trade-off

Route `x` to the small model iff `p_w(x) ≥ τ`. The threshold `τ` is the **knob**:
- `τ = 0` → everything goes to the small model (cheapest, lowest quality)
- `τ > 1` → everything goes to the large model (most expensive, highest quality)

There's no retraining when your budget changes. In practice you pick `τ` on validation data to hit a target fraction of small-model traffic (`Router.threshold_for_cost_advantage`).

## 5. Evaluation (see [`core/metrics.py`](../../llm_routing/core/metrics.py))

- **Cost advantage** = % of queries sent to the small model.
- **Quality drop** = `(q̄_all-large − q̄_routed) / |q̄_all-large| × 100%`. It can be negative when routing *improves* quality, because the small model sometimes wins.
- Sweep `τ` to get a curve. Compare against:
  - **Random router**: a straight line between the two endpoints. A learned router must beat it.
  - **Oracle**: knows the realized responses, so it gives an upper bound.

## 6. Paper setup vs. this repo

| | paper | this repo |
|---|---|---|
| Data | MixInstruct queries | MixInstruct (precomputed scores) · your own generated data · synthetic |
| Model pairs | Llama-2 7B/13B, Llama-2 13B/GPT-3.5, FLAN-T5 800M/Llama-2 13B, … | any MixInstruct pair, or any HF models via `scripts/generate_responses.py` |
| Samples per model | 10 | MixInstruct: **1** (so r_prob = r_det); synthetic / own generation: 10 |
| Router encoder | DeBERTa-v3-large | `--backend deberta` (paper) or `--backend tfidf` (CPU, fast) |
| Quality metric | BARTScore | BARTScore (`core/quality.py`) |

## 7. Our results (TF-IDF router, CPU)

MixInstruct, 10k train / 5k test, BARTScore, one sample per model:

| pair (small → large) | Pr[small ≥ large] | router | cost adv. @ ≤1% drop | drop @ 20% | drop @ 40% |
|---|---|---|---|---|---|
| alpaca-native → vicuna-13b | 0.40 | r_det / r_prob | 54% | −0.15% | 0.47% |
| | | **r_trans** (t*=0.13) | **57%** | 0.03% | **0.25%** |
| | | random | 22% | 0.82% | 1.60% |
| flan-t5-xxl → vicuna-13b | 0.09 | r_det / r_prob | 6% | 4.35% | 10.59% |
| | | **r_trans** (t*=1.09) | 5% | **4.14%** | **9.38%** |
| | | random | 2% | 6.44% | 13.09% |

What this shows:
1. When the two models are close, a learned router sends **more than half** of the traffic to the small model with under 1% quality loss. This is consistent with the paper's "up to 40% fewer large-model calls with no quality drop".
2. When the small model is much weaker, routing helps less, and r_trans (the imbalance fix) helps most at higher cost advantages.
3. With k=1, r_prob is identical to r_det. On synthetic data with k=10 (`python scripts/run_hybrid_llm.py --data synthetic`), r_prob and r_trans consistently beat r_det, as the paper reports.

Re-run with the paper's DeBERTa router (`--backend deberta`) on a GPU for stronger numbers.

## 8. Limitations worth remembering (for future comparisons)

- **Binary only**: one small and one large model. Later work routes among *many* models.
- **Needs reference answers** for BARTScore. Many later routers use preference or judge labels instead.
- **Quality ≠ BARTScore**: BARTScore rewards similarity to one reference, which can penalize valid but different answers.
- **Cost model is crude**: "% to the small model" assumes the small model is free relative to the large one. It ignores token lengths and latency.
- **Router overhead**: running DeBERTa-large is not free, though it is negligible next to a 13B+ LLM.
