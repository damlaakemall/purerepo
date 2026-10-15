# Decisions

A log of team decisions that shape the comparison. **Check "Open" before the first cluster run of a new benchmark.** When something is decided, move it to "Decided", with the date and the reason.

## Open — decide later

| # | decision | options / notes | blocks |
|---|---|---|---|
| 2 | **Cost proxy** | `shared_v1` default: `token_proxy` with `cost_per_1k_tokens` = parameters (B; all models are dense) × (prompt + completion tokens). Alternative: measured GPU time — every response now records `batch_seconds_per_sample`, so this can be decided after the pilot without regenerating. | the cost axis of every result |
| 3 | **Mixed quality scales** | `shared_v1` GT mixes 0/1 accuracy (5 sources) with continuous token F1 (Alpaca, Multi-News); `ours_v1` mixes BARTScore with 0/1. Options: report per-source results (already in `evaluate-data`); normalize per source; exclude free-text sources from GT averages. | how results are reported |
| 5 | **Shared accounts** | HF Hub dataset repo name (`hub_repo:` in benchmark configs, private) and W&B entity/project for the team. | sharing data / runs |
| 6 | **GPU types for the large models** | `mistral-small-24b`, `gemma-3-27b` and `qwen2.5-32b` are set to `gpus: 2` (tensor parallel); `phi-4` needs a >= 40 GB card. On an 80 GB card one GPU suffices for the 24B/27B models; check `sinfo -p cuda -o "%N %G"` and set `gpus`/`LLMR_GPU_TYPE` before the first run. | generating those two models |

## Decided

| date | decision | reason |
|---|---|---|
| 2026-10-15 | **No paid/token APIs.** All generation and scoring runs on Tosun GPUs (vLLM, HF fallback); the `api` engine was removed. | Team decision; school-provided GPUs. |
| 2026-10-15 | `shared_v1` pool = 12 open-weight models **disjoint from** SU Lab LLM-Routing-Project's 12 (SmolLM3 3B, Llama 3.2 3B, Phi-4-mini, Mistral 7B, OLMo 3 7B, Qwen2.5 7B/32B, Llama 3.1 8B, R1-Distill-Llama 8B, Phi-4, Mistral Small 24B, Gemma 3 27B); every model, dataset and ArmoRM revision pinned to the HF commit of this date. Llama and Gemma are gated (each member accepts the license). | Same prompts and protocol as the reference, different models; 3B-33B gives a real cost/quality spread. |
| 2026-10-15 | `shared_v1` prompts as in the reference: GSM8K, MMLU-Pro, BBH, OlympiadBench, MBPP (0/1 GT) + Alpaca, Multi-News (token F1); 60/10/30 split; k = 10 samples; max 4096 new tokens. | Matches the reference; k ≥ 5 is needed by best-of-n routers (BEST-Route). |
| 2026-10-15 | Ground truth = the reference repo's audited evaluators, plus three fixes (MCQ article false positive, BBH "Answer: B" for "(B)", MBPP timeout = wrong answer). | Same scores as the reference where it was right. |
| 2026-10-15 | Tosun settings from the reference: partition/QOS `cuda`, modules `python/3.12.9` + `cuda/12.8`, exclude `cn05,cn06`, venv. | Known to work on Tosun. |
| 2026-10-15 | The earlier Hybrid LLM implementation (code, configs, explainer, results) was removed; it is re-implemented on `shared_v1` like the other routers. | Every router built the same way, on the same new data. |
| 2026-10-15 | BARTScore is a third quality channel in `shared_v1` (`views/bartscore`), computed exactly as the reference repo does: sum of log p(response \| prompt) under `facebook/bart-large`, on every response, no reference answer. | For Hybrid LLM; same definition as the reference (verified identical on real `bart-large`). |
| 2026-10-15 | Splits as in the reference: near-duplicate groups (MiniLM cosine ≥ 0.99) kept in one split, then 60/10/30 within every source (`seed ^ crc32(source)`). | No paraphrase twin across train/test; equal ratios per source. On real data 719 of 753 grouped prompts are templated BBH items. |
| 2026-10-15 | Planned routers: Hybrid LLM, RouteLLM, GraphRouter, MixLLM, UniRoute, BEST-Route, PILOT. Implemented by the team; the data release provides what each needs (`docs/router_contract.md`). | Team plan. |
| 2026-10-13 | Routers are re-implemented from each paper and trained by us on the shared data (no released checkpoints); encoders start from the pretrained models the papers use. | Fair comparison: the only difference between routers is the method. |
| 2026-10-13 | Data: public benchmarks (MixInstruct, RouterBench) now, plus our own generated benchmark (`ours_v1`) as the main comparison. | Compare right away; own data gives k=10 samples and full control. |
| 2026-10-13 | Frozen datasets live on Tosun and in a private Hugging Face dataset repo. | Everyone uses byte-identical data (fingerprint-checked). |
| 2026-10-13 | YAML configs + Weights & Biases for tracking; local run files are always written too. | Reproducible runs, easy team comparison. |
| 2026-10-13 | Cluster: Sabancı Tosun (SLURM). | School-provided GPUs. |
| 2026-10-13 | `ours_v1` query mix: open-ended instructions (MixInstruct, BARTScore) + GSM8K (exact match) + MMLU (multiple choice). | Covers both open-ended and verifiable tasks; see open item 3 for the scale issue. |

## Shared-v1 working protocol (2026-10-14)

`shared_v1` adds member collection and immutable releases; existing benchmark configs and
results remain unchanged. GT and ArmoRM are separate channels; missing gold stays null.
Pool, revisions and Tosun settings are decided (above); the private Hub repository and
the cost definition remain open. See [team workflow](docs/team_workflow.md).
