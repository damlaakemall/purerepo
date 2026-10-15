"""Team collection protocol. Labels and router algorithms deliberately live elsewhere."""
PROTOCOL = "shared-v1"

# Quality channels a release can carry, each scored and stored separately (never mixed):
#   ground_truth  answer-key correctness: 0/1, or token F1 for free-text references
#   armorm        reward-model score of the response; every query
#   bartscore     BARTScore as SU Lab LLM-Routing-Project computes it: sum of log p(response | prompt)
#                 under facebook/bart-large; every query; negative, closer to 0 is better
# A benchmark uses the channels listed under its `scoring:` key.
CHANNELS = ("ground_truth", "armorm", "bartscore")


def channels(cfg):
    """The quality channels this benchmark config collects, in canonical order."""
    return [c for c in CHANNELS if c in (cfg.get("scoring") or {})]


def is_scored(channel, query):
    """Whether a channel scores this query at all. An unscored query stays null, never a fake label."""
    if channel == "ground_truth":
        return bool(query.get("answer_type"))
    return True
