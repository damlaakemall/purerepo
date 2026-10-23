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

The next production protocol is [`shared_v6`](configs/benchmarks/shared_v6.yaml):
ArmoRM’s pinned tokenizer measures the full question/answer chat during generation;
a guarded engine stop keeps successful answers within its 8192-token context.
The producer ceiling is also 8192 new tokens. Fresh four-model pilots must pass
before full v6 collection; the completed GT2 contribution remains available.


Settled scoring revision: [`shared_v5_gt2`](configs/benchmarks/shared_v5_gt2.yaml)
adds a 600-second OlympiadBench check deadline for all 12 models. Timed-out GT
remains missing; frozen v5 answers are reused without regeneration.
See [the decision and migration procedure](docs/walkthrough/01_egitim_verisi.md#shared_v5_gt2-kesinleşen-matematik-süre-sınırı--22-ekim-2026).


The current scoring revision is **`shared_v5_gt2`** (İbrahim G’s four-model contribution validated; v3/v4/v5 stay frozen): 12 local open models with publicly verifiable, frozen
OpenRouter provider tariffs. Six v2 models are retained and six are replaced. Costs are **estimated
API USD**, computed from local input/output token counts, not a Tosun bill. Generation stays on Tosun,
with `top_p: 1.0`; v5 disables Qwen3 thinking across the pool and uses 16384 output tokens (Phi-4: 12288), reserving prompt room within model context; ArmoRM overflows remain missing. İbrahim G’s four v5 GPU pilots passed the collection checks; the other members’ eight models still need pilots. Existing v1/v2/v3/v4/v5 configs and data
stay intact. See [the pool, prices and procedure](docs/walkthrough/01_egitim_verisi.md#shared_v3-fiyatı-doğrulanabilen-model-havuzu).

| | status |
|---|---|
| Shared collection protocol | İbrahim G’s four-model GT2 contribution validated: 306080 responses in 16 bundles; the other eight models are still pending |
| Routers | one baseline: *quality regression* (from RouterBench) |
| Planned paper routers | Hybrid LLM, RouteLLM, GraphRouter, MixLLM, UniRoute, BEST-Route, PILOT |
| Open decisions | mixed quality scales, shared accounts, GPU pilots, Hybrid pair and UniRoute pool split ([DECISIONS.md](DECISIONS.md)) |

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
llmr train configs/experiments/quality_regression__synthetic_shared_v3.yaml   # same, on CPU data shaped like the shared release
llmr check-router configs/routers/quality_regression.yaml           # fairness checks a new router must pass
llmr train configs/experiments/quality_regression__routerbench.yaml  # same on real data (downloads RouterBench)
llmr leaderboard                                                     # rebuild results/leaderboard.md
```

Any config value can be overridden on the command line, for example
`llmr train configs/experiments/quality_regression__synthetic.yaml seed=1 router.params.alpha=0.5`.

## On the cluster (Tosun)

Real generation and scoring only run on Tosun GPUs. We are three people, and each collects four of the
12 models ([`cluster/tosun/team_v3.yaml`](cluster/tosun/team_v3.yaml)). İbrahim G is the coordinator, who builds and
shares the dataset once and assembles everyone's bundles at the end. The steps are in the
[team guide (Türkçe)](docs/team/team_workflow.md#0-üç-kişiyiz-kim-ne-yapar).

Do the one-time setup in [cluster/tosun/README.md](cluster/tosun/README.md), then always start with the
30-query v5 pilot:

```bash
bash cluster/tosun/submit_my_models.sh configs/benchmarks/shared_v5_gt2_pilot.yaml <member> 1        # pilot, your 4 models
python cluster/tosun/inspect_shard.py configs/benchmarks/shared_v5_gt2_pilot.yaml <model> <member>   # check each one
bash cluster/tosun/submit_my_models.sh configs/benchmarks/shared_v5_gt2.yaml <member> 4           # full run
```

Router training on Tosun uses `bash cluster/tosun/submit_train.sh <experiment.yaml> <num_seeds>`. The only
shared-data experiments so far are the baseline's (`quality_regression__shared_v1.yaml`, `__shared_v2.yaml`, `__shared_v3.yaml`);
paper routers add theirs. The [Hybrid LLM, GraphRouter and UniRoute training plan (Türkçe)](docs/team/uc_router_egitim_plani.md)
maps the papers and reference code to our data, with proposed `shared_v3` experiments. The shared evaluation
pieces they need are in place; the routers themselves are still planned ([how to add one](AGENTS.md#when-adding-a-router)).
The [training walkthrough (Türkçe)](docs/walkthrough/02_router_egitimi.md) records the checked path from
an assembled test-double release to training. The three paper implementations and real GPU runs are pending.

## The shared data (`shared_v5_gt2`)

- **Questions:** GSM8K, MMLU-Pro, BBH, OlympiadBench, MBPP, Alpaca, Multi-News
  ([`configs/benchmarks/shared_v5_gt2.yaml`](configs/benchmarks/shared_v5_gt2.yaml)), split once into
  train / val / test and frozen.
- **Models:** 12 open-weight models from 3B to 33B, pinned to exact Hugging Face commits
  ([`configs/model_pools/team_v5.yaml`](configs/model_pools/team_v5.yaml)). Each answers every question 10 times. V5 uses non-thinking Qwen3 and a 16384 output ceiling across the pool (Phi-4: 12288); prompt space is reserved within each model’s context. Actual tokens determine cost.
- **Quality**, scored three separate ways that are never mixed: ground truth (answer key, using the
  reference repo's audited evaluators), ArmoRM (a reward model) and BARTScore.
- **Cost:** frozen input/output tariffs × measured token counts = **estimated API USD**. V1/v2 retain their parameter token proxy. Hosted precision/revisions can differ from local weights; this is a tariff simulation.
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
2. A benchmark that already has results is **never changed**. Make a new version (as with `shared_v3.yaml`) instead.
3. **No paid APIs.**
4. The three quality scores **stay separate**. A missing ground truth is never filled with another score.
5. Keep cost definitions explicit: v1/v2 **token proxy** is never USD; v3 **estimated API USD** is never actual GPU spend or a provider bill.

The full list, with the reasons, is in [AGENTS.md](AGENTS.md#rules-that-must-not-be-broken).

## Results so far

Full table: [results/leaderboard.md](results/leaderboard.md). The paper routers will be trained on `shared_v3` next.

![](results/plots/quality_regression__routerbench.png)

**Quality regression** on RouterBench (11 models) beats every single model except GPT-4 at a fraction of
the cost, but it is still far below the oracle (the router that always picks the best model). There is a
lot of room for better routers.
