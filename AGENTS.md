# AGENTS.md: orientation for AI coding agents (and new teammates)

Read this first, then [README.md](README.md). It says what the repo is for, where things live, and the rules a change must not break.

## What this repo is

A **learning project** at Sabancı University. The team re-implements LLM routing papers (routers that decide which LLM answers a query) and compares them fairly on one shared, frozen dataset that is built on the Tosun GPU cluster. The point is to *understand* routers, LLM scoring and cluster work, not to produce the most feature-complete port of each paper.

**So: keep implementations small and readable.** Follow the paper's method exactly, leave out its infrastructure, comment each step with the paper's equation or section, and write down every deliberate deviation. See [CONTRIBUTING.md → Keep it simple](CONTRIBUTING.md#keep-it-simple).

## Map

| path | what |
|---|---|
| `llm_routing/core/` | `Router` base classes and contract (`base.py`), evaluation metrics (AIQ, …), reusable text classifiers (`classifiers.py`), frozen query embeddings (`embeddings.py`), label manifest (`labels.py`) |
| `llm_routing/routers/` | one folder per paper + `baselines.py`. Only the quality-regression baseline exists today; paper routers are planned (list in [DECISIONS.md](DECISIONS.md)) |
| `llm_routing/data/` | the shared `Benchmark` format (`benchmark.py`) and loaders for synthetic / MixInstruct / RouterBench |
| `llm_routing/collection/` | `shared_v1` data collection: frozen queries and splits → per-member generation → scoring → contribution bundles → assembled release |
| `llm_routing/ground_truth/` | audited ground-truth evaluators ported from the reference repo (numeric, MCQ, BBH, OlympiadBench, MBPP code execution, token F1) |
| `llm_routing/collection/collect.py` | `shared_v1` generation and the three scoring channels: `ArmoRM`, `BARTScore` (the reference repo's definition), ground truth via `llm_routing/ground_truth`; `answer_text` strips reasoning traces |
| `llm_routing/legacy_pipeline/` | older single-user pipeline (`ours_v1`); not used for team data. Its `scorers.py` `bartscore` is the paper's reference-based variant, not the `shared_v1` channel |
| `llm_routing/runner.py`, `cli.py` | the single train + evaluate procedure, and the `llmr` command |
| `llm_routing/contract.py` | `llmr check-router`: the fairness checks every router must pass (test labels unused, dynamic-pool boundary, valid decisions) |
| `configs/` | `benchmarks/` (`shared_v6.yaml` for the next collection (generation bounded by ArmoRM context); `synthetic_shared_v3.yaml` = CPU development data; v1/v2 kept), `model_pools/team_v6.yaml` (12 priced models; v3 tariffs retained), `routers/`, `experiments/` |
| `cluster/tosun/` | SLURM scripts for Tosun; start with its README. `team_v3.yaml`: which of the 3 members collects which 4 models for v3 |
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
llmr check-router configs/routers/<router>.yaml             # fairness checks for one router, CPU, seconds
```

CI (`.github/workflows/`) runs the tests, both CPU training smokes, the router contract check and
the collection smoke. Real generation and scoring only run on Tosun GPUs:

```bash
bash cluster/tosun/submit_my_models.sh configs/benchmarks/shared_v5_pilot.yaml <member> 1    # 30-query pilot of a member's 4 models
bash cluster/tosun/submit_member.sh configs/benchmarks/shared_v5_pilot.yaml <model> <member> 1   # or one model
python cluster/tosun/inspect_shard.py configs/benchmarks/shared_v5_pilot.yaml <model> <member>    # check it before packaging
```

Setup, GPUs and known Tosun problems: [cluster/tosun/README.md](cluster/tosun/README.md) and [docs/walkthrough/01_egitim_verisi.md](docs/walkthrough/01_egitim_verisi.md).

## Rules that must not be broken

1. **Never touch the test split during training.** `fit` uses `train` (and `val` for tuning or thresholds). Test labels never choose a checkpoint or threshold.
2. **Never change a benchmark that already has results.** Make a new version (as with `shared_v3.yaml`); the fingerprint check enforces this. Frozen releases are never overwritten.
3. **No paid APIs.** All generation and scoring run on Tosun (vLLM, HF fallback).
4. **Quality channels stay separate.** Ground truth, ArmoRM and BARTScore are separate views; never fill a missing GT with another score. Paper-specific labels are built inside the router package (`labels.py`), never written into the shared data.
5. **Keep cost definitions explicit.** V1/v2 use a parameter token proxy, never USD. V3 uses frozen public input/output tariffs × local token counts: **estimated API USD**, not actual GPU expense or a provider bill. Never refresh prices during training; price changes require a new release. No paid inference APIs.
6. Don't commit `data/`, `runs/`, `outputs/`, weights, parquet or jsonl (except test fixtures).

## When finishing a project step

The team learns from the repo, not only uses it. When a step is finished (a collection, an assembly, a router, a
comparison), add or update its single write-up in `docs/walkthrough/` from `_sablon.md` (one doc per step, its parts as sections): the final procedure (not a trial
log), why each choice was made, how it was run on Tosun, and how the result was checked, with real commands and
numbers. Fix any doc the step made stale.

## When adding a router

Team members implement papers with their own coding agents. If that is you, this section is your task definition.
Details: [CONTRIBUTING.md](CONTRIBUTING.md#add-a-router) and [docs/team/router_contract.md](docs/team/router_contract.md).
For Hybrid LLM, GraphRouter and UniRoute, read [docs/team/uc_router_egitim_plani.md](docs/team/uc_router_egitim_plani.md)
first: it lists what to keep from the paper and official code and which reference-repo behaviours not to copy.

**Before writing code.** Open decisions for the paper are listed in [DECISIONS.md](DECISIONS.md) under "Open"
(e.g. Hybrid LLM's model pair and label policy). Do not settle them yourself: ask the person you work for, and record the
answer under "Decided" before relying on it. Pin the paper revision and the official code commit you follow.
For GraphRouter and UniRoute, the user has chosen the SU Lab reference implementation as the starting point
(see DECISIONS.md): preserve its core architecture and training flow, and limit changes to our data/protocol adapter.

**What to build.**
1. `llm_routing/routers/<paper>/`: `labels.py` (the paper's training targets from train/val scores) and `router.py`
   (`@register_router("<paper>")`), plus a line in `ROUTER_MODULES` in `llm_routing/registry.py`. Pick the base class
   in `core/base.py` that matches the paper's decision rule; subclass `Router` directly if none does. Comment each step
   with the paper's section/equation. Import `torch`, `transformers`, `torch_geometric` inside functions.
2. Reuse, do not rewrite: `core/classifiers.py` (soft-label text classifier, val checkpoint), `core/embeddings.py`
   (frozen encoder, any model/pooling, cache), `core/labels.py` (`label_manifest` in `describe()`).
3. What the data gives you: `bench.scores[N, M, k]` (one quality channel), `bench.costs[N, M]` (sample 0),
   `bench.sources`, `bench.extras` (`sample_cost`, `completion_tokens`, `seconds`, other channels; v3 public `prompt_tokens`),
   `bench.meta["model_info"]` (model ID, revision, parameters_b, frozen pricing; legacy cost proxy). Static knowledge the paper needs about models or
   tasks (e.g. GraphRouter's descriptions) goes in the router config, written from public model cards, never from
   results. A model subset (Hybrid's pair) is the router config's `models`. Train on some models and test on others
   (UniRoute): return them from `training_models()` and `evaluation_models()`; the runner enforces the boundary.
   One operating point per run (a GraphRouter scenario): return a one-element list from `sweep`.
   If the paper truly needs data the release lacks (another channel, more samples, other models), stop and ask:
   that is a new benchmark version (rule 2), not a change inside a router.
4. Configs: `configs/routers/<paper>.yaml` (a fast CPU variant, named as such) and `<paper>_<encoder>.yaml` (the
   paper's setup); experiments `configs/experiments/<paper>__synthetic_shared_v3.yaml` and `<paper>__shared_v3.yaml`
   (copy the `quality_regression__*` ones). Router configs name real team models, so they run on both.
5. Tests in `tests/test_<paper>.py`: labels on a hand-computable example, the paper-specific leakage checks from the
   plan, and the router in `test_check_router` in `tests/test_router_infra.py`.
6. `docs/papers/<paper>.md` from `docs/papers/_template.md`, with a "paper vs. ours" table of every deviation.

**Done means** all of these pass, and their output is shown in the PR:

```bash
pytest -q
llmr check-router configs/routers/<paper>.yaml
llmr train configs/experiments/<paper>__synthetic_shared_v3.yaml
```

The real run waits for the shared release: `bash cluster/tosun/submit_train.sh configs/experiments/<paper>__shared_v3.yaml 3`.
The shared infrastructure and its checked release-to-training example are documented in
[the training walkthrough](docs/walkthrough/02_router_egitimi.md). Paper-specific decisions stay open;
infrastructure readiness is not a claim that any of the three routers is implemented.
An explanatory notebook in `notebooks/<paper>.ipynb` is welcome. It must import the router module, not copy it.
