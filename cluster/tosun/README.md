# Running on Tosun (Sabancı University HPC)

Tosun uses the **SLURM** scheduler: you submit jobs with `sbatch` from the login node. Your home directory is `/cta/users/<username>/`. Accounts are requested through SUTicket (FENS → HPC); see the [SU HPC QuickStart](https://su-hpc-tutorials.readthedocs.io/).

## 0. One-time: fill in site settings

On the login node:

```bash
sinfo -o "%P %G %l %D"     # partitions, GPU types (gres), time limits, node counts
module avail               # available modules (anaconda / cuda names)
```

Then edit the `TODO`s in [`config.sh`](config.sh): the GPU partition, `--account`/`--qos` if required, the GPU `gres` string, module names, and the data and model-cache locations (use scratch if home space is small; models take tens of GB). Commit the file so the whole team shares it.

## 1. One-time: environment

```bash
cd /cta/users/$USER && git clone https://github.com/damlaakemall/purerepo.git && cd purerepo
bash cluster/tosun/setup_env.sh
conda activate llmr
wandb login                 # experiment tracking
huggingface-cli login       # gated models (Llama 3.1) + pushing/pulling benchmarks
```

Llama-3.1 is a gated model: accept its license on its Hugging Face page with the same account first.

## 2. Build our benchmark (once per benchmark version)

```bash
bash cluster/tosun/submit_pipeline.sh configs/benchmarks/ours_v1.yaml --push
squeue -u $USER             # watch jobs; logs in logs/
```

This runs `build-queries` on the login node, then for each model a **generate** job and a **score** job (SLURM arrays), then **assemble**, which freezes the data and pushes it to the HF Hub. Generation can resume: if a job hits its time limit, resubmit the same command and it continues where it stopped.

**Try it small first:** copy `ours_v1.yaml` to `ours_dev.yaml` with `n: 50` per source, `k: 2` and two models, and run the pipeline on that.

## 3. Train routers

```bash
bash cluster/tosun/submit_train.sh configs/experiments/hybrid_llm_deberta__ours_v1.yaml 3       # 3 seeds
bash cluster/tosun/submit_train.sh configs/experiments/hybrid_llm_deberta__ours_v1.yaml 3 router.params.mode=prob name=hybrid_llm_deberta_prob
```

Runs are written to `runs/` and to W&B. If compute nodes have no internet, set `WANDB_MODE=offline` and run `wandb sync` from the login node.

## 4. Collect results

```bash
llmr leaderboard            # -> results/leaderboard.md, commit it
```

## Interactive GPU session (debugging)

```bash
srun --partition=<gpu_partition> --gres=gpu:1 --mem=32G --time=01:00:00 --pty bash
```
