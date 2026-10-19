"""Consistency between config files that CI cannot otherwise run (real collections, Tosun-only experiments)."""
import glob
import os

import yaml

from llm_routing.config import load_benchmark_config, load_experiment

ROOT = os.path.join(os.path.dirname(__file__), "..")


def test_every_collected_model_has_exactly_one_member():
    # Otherwise a model would only turn up missing at assembly, after the GPU work.
    checked = 0
    for path in sorted(glob.glob(os.path.join(ROOT, "configs", "benchmarks", "*.yaml"))):
        cfg = load_benchmark_config(path)
        if cfg.get("protocol") != "shared-v1" or cfg.get("synthetic"):
            continue
        team = yaml.safe_load(open(os.path.join(ROOT, cfg.get("collection_team", "cluster/tosun/team.yaml"))))["members"]
        assigned = [model for member in team for model in member["models"]]
        assert sorted(assigned) == sorted(m["name"] for m in cfg["models"]), path
        assert len({m["member"] for m in team}) == len(team), path
        checked += 1
    assert checked


def test_every_experiment_config_loads():
    # Experiments on real releases cannot be trained in CI; at least their YAML must resolve.
    paths = sorted(glob.glob(os.path.join(ROOT, "configs", "experiments", "*.yaml")))
    assert paths
    for path in paths:
        cfg = load_experiment(path, ["seed=3"])
        assert cfg["seed"] == 3 and "type" in cfg["router"] and "loader" in cfg["benchmark"], path
