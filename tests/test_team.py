import re

import yaml


def test_every_pool_model_is_assigned_exactly_once():
    team = yaml.safe_load(open("cluster/tosun/team.yaml"))["members"]
    pool = [m["name"] for m in yaml.safe_load(open("configs/model_pools/team.yaml"))["models"]]
    assigned = [model for m in team for model in m["models"]]
    assert sorted(assigned) == sorted(pool)
    # same rule as collect.job_spec: the member name becomes a directory name
    assert all(re.fullmatch(r"[A-Za-z0-9_.-]+", m["member"]) for m in team)
    assert len({m["member"] for m in team}) == len(team)
    assert sum(bool(m.get("coordinator")) for m in team) == 1
