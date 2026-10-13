# Decisions

A log of team decisions that shape the comparison. **Check "Open" before the first cluster run of a new benchmark.** When something is decided, move it to "Decided", with the date and the reason.

## Open — decide later

| # | decision | options / notes | blocks |
|---|---|---|---|
| 1 | **LLM pool for `ours_v1`** | Draft in `configs/benchmarks/ours_v1.yaml`: Qwen2.5 0.5B/1.5B/3B/7B + Llama-3.1-8B (single GPU each). Option: add a ~30–70B model (e.g. Qwen2.5-32B/72B) for a bigger quality gap; it needs multi-GPU or quantization and more GPU time. Plan: run `ours_dev` on Tosun first to measure generation speed, then decide. | generating `ours_v1` |
| 2 | **Cost proxy** | Currently `cost_per_1k_tokens ∝ parameter count` × (prompt + mean completion tokens). Alternatives: measured GPU-seconds on Tosun, or public API prices of comparable models. | the cost axis of every `ours_v1` result |
| 3 | **Mixed quality scales** | `ours_v1` mixes BARTScore (negative, continuous) with 0/1 accuracy. Options: report per-source results only; normalize scores per source (e.g. rank or z-score) before averaging; train routers on normalized scores. | how `ours_v1` results are reported |
| 4 | **Tosun site settings** | Partition, GPU gres, account/QOS, module names and storage paths in `cluster/tosun/config.sh` (run `sinfo -o "%P %G %l %D"` and `module avail` on Tosun). | any cluster job |
| 5 | **Shared accounts** | HF Hub dataset repo name (`hub_repo:` in benchmark configs, private) and W&B entity/project for the team. | sharing data / runs |

## Decided

| date | decision | reason |
|---|---|---|
| 2026-10-13 | Routers are re-implemented from each paper and trained by us on the shared data (no released checkpoints); encoders start from the pretrained models the papers use. | Fair comparison: the only difference between routers is the method. |
| 2026-10-13 | Data: public benchmarks (MixInstruct, RouterBench) now, plus our own generated benchmark (`ours_v1`) as the main comparison. | Compare right away; own data gives k=10 samples and full control. |
| 2026-10-13 | Frozen datasets live on Tosun and in a private Hugging Face dataset repo. | Everyone uses byte-identical data (fingerprint-checked). |
| 2026-10-13 | YAML configs + Weights & Biases for tracking; local run files are always written too. | Reproducible runs, easy team comparison. |
| 2026-10-13 | Cluster: Sabancı Tosun (SLURM). | School-provided GPUs. |
| 2026-10-13 | `ours_v1` query mix: open-ended instructions (MixInstruct, BARTScore) + GSM8K (exact match) + MMLU (multiple choice). | Covers both open-ended and verifiable tasks; see open item 3 for the scale issue. |
