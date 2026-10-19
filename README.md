# LLM Routing — a common ground for comparing routers

> **Routing:** you have several LLMs, some small and cheap, some large and expensive. For each question, a
> *router* decides which one should answer, so you get the best quality for the money.

In this repo the team re-implements LLM routing papers and compares them **fairly**: every router is trained
on the same frozen data, tested on the same test split, and scored with the same metrics.

It is a **learning project** at Sabancı University. The goal is to learn how routers and LLM scoring really
work in code, and how to run real GPU workloads on the Tosun cluster. That is why router code is kept
**small and readable**: we follow each paper's method, leave out its infrastructure, and write down every
place where we deviate ([CONTRIBUTING.md → Keep it simple](CONTRIBUTING.md#keep-it-simple)).

## How it works, in three steps

1. **Collect the data (on Tosun).** Pick a fixed set of questions, have 12 open models answer each one
   10 times, then score every answer and count its cost. The result is one frozen table that everyone
   uses.
2. **Train routers.** Each paper's router learns from the `train` and `val` splits of that table.
3. **Compare.** Every router is evaluated on the same `test` split with the same quality/cost metrics,
   and the numbers go into one leaderboard.

## Where we are now

The next collection is prepared as **`shared_v2`**: Falcon3-7B-Instruct replaces R1, all models get an
8192-token output ceiling and `top_p: 1.0`, and large models request one 96 GB RTX PRO 6000.
The real v2 GPU pilot is still pending. Existing v1 configs and data stay intact.
See [the collection walkthrough](docs/walkthrough/01_egitim_verisi.md#shared_v2-19-ekim-2026-hazırlığı).

| | status |
|---|---|
| Data collection pipeline (`shared_v1`) | ready and tested; the 30-query pilot on Tosun comes first |
| Routers | one baseline: *quality regression* (from RouterBench) |
| Planned paper routers | Hybrid LLM, RouteLLM, GraphRouter, MixLLM, UniRoute, BEST-Route, PILOT |
| Open decisions | cost proxy, mixed quality scales, shared accounts, GPUs for the large models, token limit for the reasoning model ([DECISIONS.md](DECISIONS.md)) |

| router | family | how it decides | code | explainer |
|---|---|---|---|---|
| *quality regression* (RouterBench baseline) | multi-model | predicts each model's quality, picks argmax(quality − λ·cost) | [`routers/baselines.py`](llm_routing/routers/baselines.py) | [docs](docs/concepts/comparison.md) |
| *next paper…* | | | | |

## Where to start

| I want to... | go to |
|---|---|
| understand what we did and why (Türkçe) | [docs/walkthrough/](docs/walkthrough/README.md): one write-up per finished step, with a real run. Step 1: [producing the training data](docs/walkthrough/01_egitim_verisi.md) |
| learn routing itself | [fundamentals](docs/concepts/routing_fundamentals.md) → [how routers are compared](docs/concepts/comparison.md) → [benchmarks & fairness rules](docs/concepts/benchmarks.md) → [metrics](docs/concepts/evaluation.md) → [leaderboard](results/leaderboard.md) |
| know how the team works (Türkçe) | [team guide](docs/team/team_workflow.md), [data protocol](docs/team/shared_protocol.md), [router contract](docs/team/router_contract.md) |
| try the code on my laptop | [Quick start](#quick-start-laptop-cpu) below |
| generate and score a model on Tosun | [cluster/tosun/README.md](cluster/tosun/README.md) |
| add a router (paper) | [CONTRIBUTING.md](CONTRIBUTING.md) and the [router contract](docs/team/router_contract.md) |
| work on it with an AI assistant | [AGENTS.md](AGENTS.md): map, commands, rules that must not break |

## Quick start (laptop, CPU)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # installs the repo itself plus pytest

pytest -q                                                            # all tests, a few seconds
llmr smoke-pipeline --output /tmp/llmr-smoke                         # the whole data pipeline with fake models
llmr routers                                                         # list the routers
llmr train configs/experiments/quality_regression__synthetic.yaml    # train a router, seconds, no downloads
llmr train configs/experiments/quality_regression__routerbench.yaml  # same on real data (downloads RouterBench)
llmr leaderboard                                                     # rebuild results/leaderboard.md
```

Any config value can be overridden on the command line, for example
`llmr train configs/experiments/quality_regression__synthetic.yaml seed=1 router.params.alpha=0.5`.

## On the cluster (Tosun)

Real generation and scoring only run on Tosun GPUs. We are three people, and each collects four of the
12 models ([`cluster/tosun/team.yaml`](cluster/tosun/team.yaml)). İbrahim G is the coordinator, who builds and
shares the dataset once and assembles everyone's bundles at the end. The steps are in the
[team guide (Türkçe)](docs/team/team_workflow.md#0-üç-kişiyiz-kim-ne-yapar).

Do the one-time setup in [cluster/tosun/README.md](cluster/tosun/README.md), then always start with the
30-query pilot:

```bash
bash cluster/tosun/submit_my_models.sh configs/benchmarks/shared_pilot.yaml <member> 1        # pilot, your 4 models
python cluster/tosun/inspect_shard.py configs/benchmarks/shared_pilot.yaml <model> <member>   # check each one
bash cluster/tosun/submit_my_models.sh configs/benchmarks/shared_v1.yaml <member> 4           # full run
```

Router training on Tosun uses `bash cluster/tosun/submit_train.sh <experiment.yaml> <num_seeds>`. There are
no `shared_v1` experiment configs yet; they come with the first paper router.

## The shared data (`shared_v1`)

- **Questions:** GSM8K, MMLU-Pro, BBH, OlympiadBench, MBPP, Alpaca, Multi-News
  ([`configs/benchmarks/shared_v1.yaml`](configs/benchmarks/shared_v1.yaml)), split once into
  train / val / test and frozen.
- **Models:** 12 open-weight models from 3B to 33B, pinned to exact Hugging Face commits
  ([`configs/model_pools/team.yaml`](configs/model_pools/team.yaml)). Each answers every question 10 times.
- **Quality**, scored three separate ways that are never mixed: ground truth (answer key, using the
  reference repo's audited evaluators), ArmoRM (a reward model) and BARTScore.
- **Cost:** a token proxy (parameters × tokens), **not dollars**.
- **No paid APIs.** Everything runs on Tosun.

Each team member generates their share of the data, packages it as a checked bundle, and the bundles are
assembled into one release with hashes. The details are in the [data protocol](docs/team/shared_protocol.md).

## What is where

```
llm_routing/
  core/            Router base classes, evaluation metrics, text classifiers
  routers/         one folder per paper + baselines.py
  data/            the shared Benchmark format and loaders (synthetic, MixInstruct, RouterBench)
  collection/      team data collection (shared_v1): questions -> answers -> scores -> release
  ground_truth/    answer-key checkers (math, multiple choice, code execution, ...)
  legacy_pipeline/ older single-user benchmark (ours_v1); kept so its configs still run
  runner.py        the one train + evaluate procedure
  cli.py           the `llmr` command
configs/           benchmarks/, model_pools/, routers/, experiments/
cluster/tosun/     SLURM setup and job scripts for Tosun
docs/
  concepts/        routing basics, comparison, benchmarks, metrics (English)
  team/            team workflow, data protocol, router contract (Türkçe)
  walkthrough/     how each project step was done and why (Türkçe)
  papers/          one explainer per paper
results/           leaderboard and plots
tests/             CPU tests, also run in CI
```

How one training run flows:

```
configs/benchmarks/*.yaml ──► frozen Benchmark (questions, scores, costs, fixed splits, fingerprint)
configs/routers/*.yaml    ──► Router.fit(train, val) ──► Router.sweep(test) ──► quality/cost curve
configs/experiments/*.yaml                                                          │
                                       runs/<benchmark>/<experiment>/<time>_s<seed>/ (+ W&B)
                                                                                    │
                                                    llmr leaderboard ──► results/leaderboard.md
```

## Rules that must not be broken

1. Routers never see the **test split** while training. Thresholds and checkpoints are chosen on `val`.
2. A benchmark that already has results is **never changed**. Make a new version (`shared_v2.yaml`) instead.
3. **No paid APIs.**
4. The three quality scores **stay separate**. A missing ground truth is never filled with another score.
5. Cost is a **token proxy**. Never report it as dollars.

The full list, with the reasons, is in [AGENTS.md](AGENTS.md#rules-that-must-not-be-broken).

## Results so far

Full table: [results/leaderboard.md](results/leaderboard.md). The paper routers will be trained on `shared_v1` next.

![](results/plots/quality_regression__routerbench.png)

**Quality regression** on RouterBench (11 models) beats every single model except GPT-4 at a fraction of
the cost, but it is still far below the oracle (the router that always picks the best model). There is a
lot of room for better routers.
