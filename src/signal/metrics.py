def relaxation_score(state, min_beta=1.0, max_score=10.0):

    theta = state["Theta"]
    alpha = state["Alpha"]
    beta = state["Beta"]

    safe_beta = max(beta, min_beta)

    score = (theta + alpha) / safe_beta

    score = min(score, max_score)

    return score