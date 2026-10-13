"""Experiment tracking: Weights & Biases when enabled, otherwise a no-op.

Every run ALSO writes local files (config.yaml, metrics.json, curve.csv, plot.png),
so results never depend on W&B being available (e.g. offline cluster nodes:
set WANDB_MODE=offline and `wandb sync` later).
"""

from __future__ import annotations


class Tracker:
    def __init__(self, cfg: dict, run_name: str, run_dir: str):
        w = cfg.get("wandb") or {}
        self.run = None
        if not w.get("enabled"):
            return
        try:
            import wandb
        except ImportError:
            print("[tracking] wandb not installed; `pip install wandb` or set wandb.enabled=false")
            return
        self.wandb = wandb
        self.run = wandb.init(project=w.get("project", "llm-routing"), entity=w.get("entity"), name=run_name,
                              group=w.get("group") or cfg["benchmark"]["name"], config=cfg, dir=run_dir,
                              tags=[cfg["benchmark"]["name"], cfg["router"]["type"]], reinit=True)

    def log_summary(self, metrics: dict):
        if self.run:
            for k, v in metrics.items():
                if isinstance(v, (int, float, str)) or v is None:
                    self.run.summary[k] = v

    def log_curve(self, name: str, rows: list[dict]):
        if self.run and rows:
            cols = list(rows[0])
            self.run.log({f"curve/{name}": self.wandb.Table(columns=cols, data=[[r[c] for c in cols] for r in rows])})

    def log_image(self, key: str, path: str):
        if self.run:
            self.run.log({key: self.wandb.Image(path)})

    def finish(self):
        if self.run:
            self.run.finish()
