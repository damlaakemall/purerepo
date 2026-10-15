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
   - Reuse `core/classifiers.py` (TF-IDF or transformer, soft-label BCE) if the router is a text classifier.
3. **Register the module** in `ROUTER_MODULES` in [`registry.py`](llm_routing/registry.py).
4. **Config**: add `configs/routers/<paper>.yaml` (a fast CPU variant) and, if the paper uses a large encoder, `<paper>_<encoder>.yaml` with the paper's exact setup.
5. **Experiments**: add `configs/experiments/<paper>__<benchmark>.yaml` for synthetic and `shared_v1` (the main comparison); a public benchmark (MixInstruct, RouterBench) is optional.
6. **Docs**: write `docs/papers/<paper>.md` using [`docs/papers/_template.md`](docs/papers/_template.md), and add a row to [docs/comparison.md](docs/comparison.md) and the README table.
7. **Tests**: add the router config to `test_every_router_produces_valid_decisions` in `tests/test_framework.py`, plus unit tests for the paper-specific parts.

### Rules that keep the comparison fair

- `fit` may only use the `train` split, plus `val` for tuning or early stopping. **Never read `test`.**
- At inference a router sees only query text, plus responses of models it has paid for if it's a cascade. Costs it needs are **estimated from train**.
- Use the paper's hyperparameters when they're reported; otherwise tune on `val` only, and write down what you tuned in the explainer.
- Report **≥ 3 seeds** for anything that goes into the leaderboard.

## Add a benchmark

- **Public dataset**: write `llm_routing/data/<name>.py` returning a `Benchmark`, add a branch in `data/loaders.py`, and add `configs/benchmarks/<name>.yaml`. Splits must be deterministic (seeded).
- **Our own data**: copy `configs/benchmarks/ours_v1.yaml` to a new version (`ours_v2.yaml`). **Never change a benchmark that already has results.** Make a new version instead, so old numbers stay valid (the fingerprint check enforces this).

## Workflow

- Work on a branch, open a PR, and wait for CI (tests plus a CPU smoke run) to pass before merging into `main`.
- Commit `results/leaderboard.md` and selected plots. Don't commit `runs/`, `data/` or `wandb/`; they're git-ignored.

## Shared collection and future papers

Use [the team workflow](docs/team_workflow.md) for member contributions and
[the router contract](docs/router_contract.md) for paper-native labels, architecture,
loss and inference. Shared artifacts never contain a universal router label.
`Decision.sample_index` identifies the returned sample; `Decision.cost` includes all
paid calls. Run `llmr smoke-pipeline --output /tmp/llmr-shared-check` before submitting.
