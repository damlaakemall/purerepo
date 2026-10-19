# Benchmarks and fairness rules

Every router is trained and tested on a **frozen `Benchmark`** ([`data/benchmark.py`](../../llm_routing/data/benchmark.py)):

| field | shape | meaning |
|---|---|---|
| `queries`, `query_ids` | N | query text and a stable id |
| `models` | M | candidate LLMs |
| `scores` | N × M × k | quality of k sampled responses (higher is better; NaN means missing) |
| `costs` | N × M | cost of answering query n with model m |
| `split_of` | N | `train` / `val` / `test`, fixed on disk |
| `sources` | N | which dataset or task the query came from (for per-domain analysis) |
| `responses` | optional | response texts (needed by cascade routers) |

**Why k samples?** LLM outputs are random, so the same model can answer the same query well once and badly the next time. Routers like Hybrid LLM's r_prob learn from this spread, so our own benchmark stores k = 10 samples. Evaluation always uses sample 0, the answer the user would actually receive.

**Fingerprint.** A hash of ids, models, splits, scores and costs. It's stored in `manifest.json`, checked on every load, and written into every run. The leaderboard only puts numbers side by side when their fingerprints match.

## Available benchmarks

| config | models | k | quality | cost | size | use it for |
|---|---|---|---|---|---|---|
| `synthetic` | 4 toy models | 10 | simulated BARTScore-like | simulated | 4k | tests, learning, CPU smoke runs |
| `mixinstruct` | 12 open LLMs (2023) | 1 | BARTScore | parameter count (proxy) | 16k | Hybrid LLM's original setting |
| `routerbench` | 11 incl. GPT-4, Claude, Mixtral | 1 | 0/1 correctness | real USD | 36k | multi-model routers, real prices |
| `ours_v1` | Qwen2.5 0.5–7B, Llama-3.1-8B (draft) | 10 | BARTScore / exact match / MC accuracy per source | ∝ params × tokens | 10k (draft) | earlier draft, kept for existing runs |
| `shared_v1` | 12 open-weight, 3B-33B (SmolLM3, Llama 3.x, Phi-4, Mistral, OLMo 3, Qwen2.5, R1-Distill-Llama, Gemma 3) | 10 | ground truth (0/1, token F1), ArmoRM and BARTScore, separate views | ∝ params × tokens | ~7.6k | legacy comparison: the team's version of SU Lab LLM-Routing-Project, built on Tosun ([protocol](../team/shared_protocol.md)) |

The next collection is `shared_v3`: 12 price-verifiable local models, the same sources/scorers/splits
as v2, with separate input/output tariffs frozen on 2026-10-19. `cost = (prompt_tokens * input_per_1m +
completion_tokens * output_per_1m) / 1e6`. This is **estimated API USD**, not a GPU expense or an API bill.
Hosted revisions/precision can differ from our local BF16 weights. V1/v2 remain parameter token proxies.
`synthetic_shared_v3` uses the same names/tariffs with invented scores and token counts, for CPU development.
See the [collection write-up](../walkthrough/01_egitim_verisi.md#shared_v3-fiyatı-doğrulanabilen-model-havuzu).

## Building the earlier draft benchmark (`ours_v1`)

`shared_v1` is built with the member workflow instead: see [team_workflow.md](../team/team_workflow.md) and [shared_protocol.md](../team/shared_protocol.md).

1. `llmr build-queries` samples queries from each source, applies a fixed seed, and assigns splits. This runs once.
2. `llmr generate --model M` runs vLLM on a GPU and writes k samples per query. It can resume, and runs as one SLURM array task per model.
3. `llmr score --model M` scores each source with its configured scorer.
4. `llmr assemble` writes the frozen benchmark; `llmr push-data` uploads it to a private HF dataset repo.
5. Everyone else runs `llmr pull-data`, or `llmr train` does it automatically when `hub_repo` is set.

**Mixed quality scales.** `ours_v1` mixes BARTScore (negative, continuous) with 0/1 accuracy, so overall averages are dominated by the source with the widest scale. Look at per-source results too, or normalize per source when you compare. This is still an open decision for the team.
