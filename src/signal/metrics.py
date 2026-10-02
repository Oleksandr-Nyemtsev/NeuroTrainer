import math


def relaxation_score(state, min_beta=1.0, max_score=10.0):

    theta = state["Theta"]
    alpha = state["Alpha"]
    beta = state["Beta"]

    if not all(math.isfinite(v) for v in (theta, alpha, beta, min_beta, max_score)) or min_beta <= 0:
        raise ValueError("Score inputs must be finite and min_beta positive")

    safe_beta = max(beta, min_beta)

    score = (theta + alpha) / safe_beta

    if not math.isfinite(score):
        raise ValueError("Non-finite relaxation score")
    score = min(score, max_score)

    return score