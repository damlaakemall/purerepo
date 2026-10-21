# Running on Tosun (Sabancı University HPC)

Tosun uses the **SLURM** scheduler: you submit jobs with `sbatch` from the login node. Your home directory is `/cta/users/<username>/`. Accounts are requested through SUTicket (FENS → HPC); see the [SU HPC QuickStart](https://su-hpc-tutorials.readthedocs.io/).

## 0. Site settings (already filled in)

[`config.sh`](config.sh) holds the settings SU Lab LLM-Routing-Project runs with on Tosun: partition and QOS `cuda`,
modules `python/3.12.9` and `cuda/12.8`, a Python venv (no conda). Nodes without bf16 (V100, K80) and cn14 (16 GB
A4000) are excluded. Override any of
them from the environment, e.g. `LLMR_GPU_TYPE=a100` to request a specific card. To see what each node has:

```bash
sinfo -p cuda -o "%20N %12G %8m %t"
```

Model weights for the 12-model pool take ~390 GB for v3 BF16 weights alone (allow extra cache/scorer space): point `HF_HOME` at scratch if your home quota is small.
No paid APIs are used; everything runs on Tosun GPUs.

## 1. One-time: environment

```bash
cd /cta/users/$USER && git clone https://github.com/damlaakemall/purerepo.git && cd purerepo
mkdir -p logs   # setup runs on a compute node: the login node's memory limit kills pip while installing torch
sbatch -p cuda --qos=cuda -c 8 --mem=32G -t 2:00:00 -o logs/setup_env-%j.out --wrap "bash cluster/tosun/setup_env.sh"
hf auth login                        # model downloads + pushing/pulling releases; accept the Llama and Gemma licenses on huggingface.co first
wandb login                          # experiment tracking (optional)
```

GPU jobs check `torch.cuda.is_available()` first (`llmr_require_gpu`): a torch built for a newer CUDA than the
driver would otherwise run on CPU, ~750x slower, and die on the wall clock hours later. Tosun's driver is 570
(CUDA 12.x), so `setup_env.sh` pins vLLM and torch builds for CUDA 12.9. Start with the 30-query pilot
(`configs/benchmarks/shared_v3_pilot.yaml`); the whole procedure, explained step by step with a real run, is in
[docs/walkthrough/01_egitim_verisi.md](../../docs/walkthrough/01_egitim_verisi.md) (Turkish).

## 2. Build a benchmark with the older single-user pipeline (`ours_v1` / `ours_dev`)

> The next main benchmark is `shared_v3`, built by team members model by model: skip to [Independent shared-v1 member jobs](#independent-shared-v1-member-jobs). This section is kept for the earlier `ours_v1` draft and the quick `ours_dev` pipeline test. Check the open items in [DECISIONS.md](../../DECISIONS.md) first.

```bash
bash cluster/tosun/submit_pipeline.sh configs/benchmarks/ours_v1.yaml --push
squeue -u $USER             # watch jobs; logs in logs/
```

This runs `build-queries` on the login node, then for each model a **generate** job and a **score** job (SLURM arrays), then **assemble**, which freezes the data and pushes it to the HF Hub. Generation can resume: if a job hits its time limit, resubmit the same command and it continues where it stopped.

**Try it small first:** `configs/benchmarks/ours_dev.yaml` (50 queries per source, k=2, two small models) runs the whole pipeline in minutes: `bash cluster/tosun/submit_pipeline.sh configs/benchmarks/ours_dev.yaml`.

## 3. Train routers

```bash
bash cluster/tosun/submit_train.sh configs/experiments/<router>__shared_v3.yaml 3                 # 3 seeds
bash cluster/tosun/submit_train.sh configs/experiments/<router>__shared_v3.yaml 3 seed=7 name=<variant>   # overrides
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

For the next collection use `configs/benchmarks/shared_v5_pilot.yaml`, then
`configs/benchmarks/shared_v5.yaml`. V5 uses non-thinking Qwen3 and 16384 output / 32768 total context across the pool, except Phi-4 (12288 / 16384); actual prompt room further limits output. Only known ArmoRM context overflows and absent final-answer reward scores may remain missing. The frozen v3 tariffs and member assignments are reused. Actual measured tokens determine cost. `collection_team` selects `team_v3.yaml`: four priced models per
member, with each 12B+ model requesting one `rtx_pro_6000` 96GB GPU. Generation is BF16 locally;
OpenRouter tariffs only simulate estimated API USD. Accept the Llama/Gemma licenses assigned to you.
İbrahim G’s four v5 pilots are validated (240 responses, two declared ArmoRM overflows); his four-model full collection has started, while the other members’ pilots remain pending; see the [walkthrough](../../docs/walkthrough/01_egitim_verisi.md#shared_v3-fiyatı-doğrulanabilen-model-havuzu).
V1/v2/v3 configs and their artifacts remain unchanged.

We are three members with four models each; the assignment is in [`team_v3.yaml`](team_v3.yaml) and the team
procedure (who builds the dataset, who assembles) in the [team workflow](../../docs/team/team_workflow.md#0-üç-kişiyiz-kim-ne-yapar).
After the dataset is frozen and shared, each member submits all their models with

```bash
bash cluster/tosun/submit_my_models.sh configs/benchmarks/shared_v5_pilot.yaml <member> 1   # pilot first
bash cluster/tosun/submit_my_models.sh configs/benchmarks/shared_v5.yaml <member> 4      # full run
```

which calls `submit_member.sh CONFIG MODEL MEMBER NUM_SHARDS` once per model (use that directly to resubmit one model). It requests the model's `gpus` (one 96GB card for v3 models with 12B+ parameters), then runs
GT (CPU), ArmoRM (1 GPU) and BARTScore (1 GPU, if configured) scoring sequentially (one shard writer lock) and packages the shard; it never submits router training.
`LLMR_BATCH_SIZE` (default 256) is how many requests go to vLLM at once. The older `submit_pipeline.sh` continues
to serve the original ours_v1/ours_dev configs. See [team workflow](../../docs/team/team_workflow.md).
