# AGENTS.md: orientation for AI coding agents (and new teammates)

Read this first, then [README.md](README.md). It says what the repo is for, where things live, and the rules a change must not break.

## What this repo is

A **learning project** at Sabancı University. The team re-implements LLM routing papers (routers that decide which LLM answers a query) and compares them fairly on one shared, frozen dataset that is built on the Tosun GPU cluster. The point is to *understand* routers, LLM scoring and cluster work, not to produce the most feature-complete port of each paper.

**So: keep implementations small and readable.** Follow the paper's method exactly, leave out its infrastructure, comment each step with the paper's equation or section, and write down every deliberate deviation. See [CONTRIBUTING.md → Keep it simple](CONTRIBUTING.md#keep-it-simple).

## Map

| path | what |
|---|---|
| `llm_routing/core/` | `Router` base classes and contract (`base.py`), evaluation metrics (AIQ, …), reusable text classifiers |
| `llm_routing/routers/` | one folder per paper + `baselines.py`. Only the quality-regression baseline exists today; paper routers are planned (list in [DECISIONS.md](DECISIONS.md)) |
| `llm_routing/data/` | the shared `Benchmark` format (`benchmark.py`) and loaders for synthetic / MixInstruct / RouterBench |
| `llm_routing/collection/` | `shared_v1` data collection: frozen queries and splits → per-member generation → scoring → contribution bundles → assembled release |
| `llm_routing/ground_truth/` | audited ground-truth evaluators ported from the reference repo (numeric, MCQ, BBH, OlympiadBench, MBPP code execution, token F1) |
| `llm_routing/collection/collect.py` | `shared_v1` generation and the three scoring channels: `ArmoRM`, `BARTScore` (the reference repo's definition), ground truth via `llm_routing/ground_truth`; `answer_text` strips reasoning traces |
| `llm_routing/legacy_pipeline/` | older single-user pipeline (`ours_v1`); not used for team data. Its `scorers.py` `bartscore` is the paper's reference-based variant, not the `shared_v1` channel |
| `llm_routing/runner.py`, `cli.py` | the single train + evaluate procedure, and the `llmr` command |
| `configs/` | `benchmarks/` (`shared_v1.yaml` is the main one), `model_pools/team.yaml` (12 models), `routers/`, `experiments/` |
| `cluster/tosun/` | SLURM scripts for Tosun; start with its README |
| `docs/concepts/` | routing fundamentals, comparison, benchmarks, evaluation metrics (English) |
| `docs/team/` | team workflow, `shared_v1` data protocol, router contract (Turkish) |
| `docs/papers/` | paper explainer template (one explainer per paper) |
| `docs/walkthrough/` | "how we did it and why" write-ups (Turkish), one cohesive doc per project step, with a real run |
| `DECISIONS.md` | decided and open team decisions; check it before changing anything that affects the data or the comparison |

Docs language: `docs/concepts/` is in English; `docs/team/` and `docs/walkthrough/` are in Turkish.

## Commands

```bash
pip install -e '.[dev]'
pytest -q                                                   # all tests, CPU, ~seconds
llmr smoke-pipeline --output /tmp/llmr-smoke                # whole shared-v1 pipeline with test doubles (use a fresh path each time)
llmr train configs/experiments/quality_regression__synthetic.yaml   # CPU router training smoke run
llmr routers                                                # registered routers
```

CI (`.github/workflows/`) runs these three. Real generation and scoring only run on Tosun GPUs:

```bash
bash cluster/tosun/submit_member.sh configs/benchmarks/shared_pilot.yaml <model> <member> 1   # 30-query pilot first
python cluster/tosun/inspect_shard.py configs/benchmarks/shared_pilot.yaml <model> <member>    # check it before packaging
```

Setup, GPUs and known Tosun problems: [cluster/tosun/README.md](cluster/tosun/README.md) and [docs/walkthrough/01_egitim_verisi.md](docs/walkthrough/01_egitim_verisi.md).

## Rules that must not be broken

1. **Never touch the test split during training.** `fit` uses `train` (and `val` for tuning or thresholds). Test labels never choose a checkpoint or threshold.
2. **Never change a benchmark that already has results.** Make a new version (`shared_v2.yaml`); the fingerprint check enforces this. Frozen releases are never overwritten.
3. **No paid APIs.** All generation and scoring run on Tosun (vLLM, HF fallback).
4. **Quality channels stay separate.** Ground truth, ArmoRM and BARTScore are separate views; never fill a missing GT with another score. Paper-specific labels are built inside the router package (`labels.py`), never written into the shared data.
5. **Costs are a token proxy** (parameters × tokens), not USD. Don't report them as dollars.
6. Don't commit `data/`, `runs/`, `outputs/`, weights, parquet or jsonl (except test fixtures).

## When finishing a project step

The team learns from the repo, not only uses it. When a step is finished (a collection, an assembly, a router, a
comparison), add or update its single write-up in `docs/walkthrough/` from `_sablon.md` (one doc per step, its parts as sections): the final procedure (not a trial
log), why each choice was made, how it was run on Tosun, and how the result was checked, with real commands and
numbers. Fix any doc the step made stale.

## When adding a router

Follow [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/team/router_contract.md](docs/team/router_contract.md): package in `llm_routing/routers/<paper>/`, register it, add configs, an explainer from `docs/papers/_template.md`, and tests. An explanatory notebook in `notebooks/<paper>.ipynb` is welcome. It must import the router module, not copy it.
