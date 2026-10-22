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


class ArmoRMContextOverflow(ValueError):
    """A known scoring boundary, distinct from a broken scorer or GPU failure."""
    def __init__(self, tokens, limit):
        self.tokens, self.limit = tokens, limit
        super().__init__(f"ArmoRM context overflow: {tokens} > {limit}; no silent truncation")


class NoFinalAnswer(ValueError):
    """The model produced no text outside its reasoning trace."""


def allowed_missing_score(cfg, channel, record):
    """Only explicitly opted-in, measured boundary failures may remain missing."""
    details = record.get("details", {})
    tokens, limit = details.get("tokens"), details.get("limit")
    gt = cfg.get("scoring", {}).get("ground_truth", {})
    seconds = details.get("seconds")
    if (channel == "ground_truth" and gt.get("on_math_timeout") == "missing"
            and record.get("success") is False and record.get("score") is None
            and details.get("reason") == "math_timeout"
            and type(seconds) in (int, float) and seconds > 0
            and seconds == gt.get("math_timeout")):
        return True
    if (channel in ("armorm", "bartscore")
            and cfg.get("scoring", {}).get(channel, {}).get("text_policy") == "final_only"
            and record.get("success") is False and record.get("score") is None
            and details.get("reason") == "no_final_answer"):
        return True
    return (channel == "armorm"
            and cfg.get("scoring", {}).get(channel, {}).get("on_context_overflow") == "missing"
            and record.get("success") is False and record.get("score") is None
            and details.get("reason") == "context_overflow"
            and type(tokens) is int and type(limit) is int and tokens > limit > 0)
