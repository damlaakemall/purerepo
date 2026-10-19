# Contributing: adding a router (paper) or a benchmark

## Keep it simple

This repo is for **learning** how routers work, so an implementation should read like the paper:

- Follow the paper's **method** exactly (labels, model, loss, decision rule). Leave out its **infrastructure**: own data loaders and eval scripts (we have the shared ones), ablation variants, distributed training, extra CLI options.
- Small functions with clear names; a short comment ties each step to the paper (e.g. `# Eq. 3: soft label r_prob`).
- If you simplify something on purpose (a smaller encoder, fewer epochs), say so in a comment and in the explainer's "paper vs. ours" table.
- **Optional companion notebook** (`notebooks/<paper>.ipynb`): walks through the paper step by step on a small example. It **imports** the router from `llm_routing/routers/<paper>/` and never copies its code, because the module is what Tosun jobs and the leaderboard run. Suggested outline: the problem → the key equation → a worked example with intermediate outputs → threshold vs. cost/quality plot → deviations from the paper.

## Add a router

1. **Pick the family** (see [`core/base.py`](llm_routing/core/base.py)):
   - `BinaryRouter`: small vs. large. Implement `score(queries) -> [0,1]`, where higher means the small model is fine.
   - `MultiModelRouter`: implement `predict_quality(queries) -> (N, M)`. The base class handles the cost trade-off.
   - `CascadeRouter`: implement `sweep(bench)` yourself and add up the cost of every model you call.
   - Anything else: subclass `Router` and implement `sweep(bench) -> list[Decision]`, cheapest operating point first.
2. **Create `llm_routing/routers/<paper>/`** with:
   - `labels.py` (or similar): the paper-specific idea, i.e. how training targets are built from the scores.
   - `router.py`: the router class, decorated with `@register_router("<paper>")`.
   - Reuse `core/classifiers.py` (TF-IDF or transformer, soft-label BCE; pass `val_texts`/`val_labels` to keep the best validation epoch) if the router is a text classifier, and `core/embeddings.py` for frozen query embeddings.
   - Put `label_manifest(...)` from `core/labels.py` in `describe()`, so every run records how its labels were built.
   - Dynamic pool (train on some LLMs, test on others): return them from `training_models()` and `evaluation_models()`. The runner then gives `fit` only those models' train labels and fits the baselines on val.
3. **Register the module** in `ROUTER_MODULES` in [`registry.py`](llm_routing/registry.py).
4. **Config**: add `configs/routers/<paper>.yaml` (a fast CPU variant) and, if the paper uses a large encoder, `<paper>_<encoder>.yaml` with the paper's exact setup.
5. **Experiments**: add `configs/experiments/<paper>__synthetic_shared_v3.yaml` (CPU data shaped like the shared release, with the team's model names) and `<paper>__shared_v3.yaml` (the real data); copy the `quality_regression__*` configs. A public benchmark (MixInstruct, RouterBench) is optional.
6. **Docs**: write `docs/papers/<paper>.md` using [`docs/papers/_template.md`](docs/papers/_template.md), and add a row to [docs/concepts/comparison.md](docs/concepts/comparison.md) and the README table.
7. **Tests**: add the router config to `test_check_router` in `tests/test_router_infra.py`, plus unit tests for the paper-specific parts in `tests/test_<paper>.py`. Then run `llmr check-router configs/routers/<paper>.yaml`: it perturbs test scores, costs, every `extras` outcome channel (public `prompt_tokens` stays fixed) and gold references; for a dynamic pool it also removes the new models' train scores/costs/extras. It checks that decisions stay unchanged and that `describe()` can be saved by the runner's JSON serializer. Return plain Python values there (e.g. NumPy arrays via `.tolist()`).

Tests should verify method calculations, information boundaries, data integrity and actual regressions.
Do not copy model lists, team assignments, dated prices or fixed config choices into assertions;
use small toy inputs for behavioral checks. Avoid duplicating checks already covered by `check-router`
or an existing integration test.

### Rules that keep the comparison fair

- `fit` may only use the `train` split, plus `val` for tuning or early stopping. **Never read `test`.** `llmr check-router` checks the part of this that can be checked automatically.
- At inference a router sees only query text, plus responses of models it has paid for if it's a cascade. Costs it needs are **estimated from train** (new-model profiles from val in a dynamic pool). Frozen tariffs and tokenized query length are public inputs; realized test output length/cost is an outcome.
- Use the paper's hyperparameters when they're reported; otherwise tune on `val` only, and write down what you tuned in the explainer.
- Report **≥ 3 seeds** for anything that goes into the leaderboard.

## Add a benchmark

- **Public dataset**: write `llm_routing/data/<name>.py` returning a `Benchmark`, add a branch in `data/loaders.py`, and add `configs/benchmarks/<name>.yaml`. Splits must be deterministic (seeded).
- **Our own data**: copy `configs/benchmarks/ours_v1.yaml` to a new version (`ours_v2.yaml`). **Never change a benchmark that already has results.** Make a new version instead, so old numbers stay valid (the fingerprint check enforces this).

## Workflow

- Work on a branch, open a PR, and wait for CI (tests plus a CPU smoke run) to pass before merging into `main`.
- Commit `results/leaderboard.md` and selected plots. Don't commit `runs/`, `data/` or `wandb/`; they're git-ignored.

## Shared collection and future papers

Use [the team workflow](docs/team/team_workflow.md) for member contributions and
[the router contract](docs/team/router_contract.md) for paper-native labels, architecture,
loss and inference. Shared artifacts never contain a universal router label.
`Decision.sample_index` identifies the returned sample; `Decision.cost` includes all
paid calls. Run `llmr smoke-pipeline --output /tmp/llmr-shared-check` before submitting.
