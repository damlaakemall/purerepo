# Running on Tosun (Sabancı University HPC)

Tosun uses the **SLURM** scheduler: you submit jobs with `sbatch` from the login node. Your home directory is `/cta/users/<username>/`. Accounts are requested through SUTicket (FENS → HPC); see the [SU HPC QuickStart](https://su-hpc-tutorials.readthedocs.io/).

## 0. Site settings (already filled in)

[`config.sh`](config.sh) holds the settings SU Lab LLM-Routing-Project runs with on Tosun: partition and QOS `cuda`,
modules `python/3.12.9` and `cuda/12.8`, nodes `cn05,cn06` excluded, a Python venv (no conda). Override any of
them from the environment, e.g. `LLMR_GPU_TYPE=a100` to request a specific card. To see what each node has:

```bash
sinfo -p cuda -o "%20N %12G %8m %t"
```

Model weights for the 12-model pool take ~300 GB: point `HF_HOME` at scratch if your home quota is small.
No paid APIs are used; everything runs on Tosun GPUs.

## 1. One-time: environment

```bash
cd /cta/users/$USER && git clone https://github.com/damlaakemall/purerepo.git && cd purerepo
bash cluster/tosun/setup_env.sh      # venv, torch for CUDA 12.8, the package, vLLM
hf auth login                        # model downloads + pushing/pulling releases; accept the Llama and Gemma licenses on huggingface.co first
wandb login                          # experiment tracking (optional)
```

GPU jobs check `torch.cuda.is_available()` first (`llmr_require_gpu`): a torch built for a newer CUDA than the
driver would otherwise run on CPU, ~750x slower, and die on the wall clock hours later.

## 2. Build a benchmark with the older single-user pipeline (`ours_v1` / `ours_dev`)

> The main benchmark is now `shared_v1`, built by team members model by model: skip to [Independent shared-v1 member jobs](#independent-shared-v1-member-jobs). This section is kept for the earlier `ours_v1` draft and the quick `ours_dev` pipeline test. Check the open items in [DECISIONS.md](../../DECISIONS.md) first.

```bash
bash cluster/tosun/submit_pipeline.sh configs/benchmarks/ours_v1.yaml --push
squeue -u $USER             # watch jobs; logs in logs/
```

This runs `build-queries` on the login node, then for each model a **generate** job and a **score** job (SLURM arrays), then **assemble**, which freezes the data and pushes it to the HF Hub. Generation can resume: if a job hits its time limit, resubmit the same command and it continues where it stopped.

**Try it small first:** `configs/benchmarks/ours_dev.yaml` (50 queries per source, k=2, two small models) runs the whole pipeline in minutes: `bash cluster/tosun/submit_pipeline.sh configs/benchmarks/ours_dev.yaml`.

## 3. Train routers

```bash
bash cluster/tosun/submit_train.sh configs/experiments/<router>__shared_v1.yaml 3                 # 3 seeds
bash cluster/tosun/submit_train.sh configs/experiments/<router>__shared_v1.yaml 3 seed=7 name=<variant>   # overrides
```

Runs are written to `runs/` and to W&B. If compute nodes have no internet, set `WANDB_MODE=offline` and run `wandb sync` from the login node.

## 4. Collect results

```bash
llmr leaderboard            # -> results/leaderboard.md, commit it
```

## Interactive GPU session (debugging)

```bash
srun --partition=cuda --qos=cuda --gres=gpu:1 --mem=32G --time=01:00:00 --pty bash
```

## Independent shared-v1 member jobs

For the new protocol use `submit_member.sh CONFIG MODEL MEMBER NUM_SHARDS` after
freezing/sharing the dataset. It requests the model's `gpus` (2 for the 24B, 27B and 32B models), then runs
GT (CPU), ArmoRM (1 GPU) and BARTScore (1 GPU, if configured) scoring in parallel and packages the shard; it never submits router training.
`LLMR_BATCH_SIZE` (default 256) is how many requests go to vLLM at once. The older `submit_pipeline.sh` continues
to serve the original ours_v1/ours_dev configs. See [team workflow](../../docs/team_workflow.md).
