import re

import pytest
import yaml

from llm_routing.config import load_benchmark_config


@pytest.mark.parametrize("benchmark", ["shared_v1", "shared_pilot", "shared_v2", "shared_v2_pilot", "shared_v3", "shared_v3_pilot"])
def test_every_pool_model_is_assigned_exactly_once(benchmark):
    cfg = load_benchmark_config(f"configs/benchmarks/{benchmark}.yaml")
    team = yaml.safe_load(open(cfg.get("collection_team", "cluster/tosun/team.yaml")))["members"]
    pool = [m["name"] for m in cfg["models"]]
    assigned = [model for m in team for model in m["models"]]
    assert sorted(assigned) == sorted(pool)
    # same rule as collect.job_spec: the member name becomes a directory name
    assert all(re.fullmatch(r"[A-Za-z0-9_.-]+", m["member"]) for m in team)
    assert len({m["member"] for m in team}) == len(team)
    assert sum(bool(m.get("coordinator")) for m in team) == 1
