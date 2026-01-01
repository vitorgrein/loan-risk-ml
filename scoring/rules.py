def calculate_score(prob_default: float) -> int:
    score = round(1000 * (1 - prob_default))
    return max(1, min(score, 1000))


def classify_risk(score: int) -> str:
    if score >= 700:
        return "BAIXO"
    elif score >= 400:
        return "MEDIO"
    return "ALTO"


def classify_priority(risk: str, loan_amnt: float) -> int:
    if loan_amnt >= 20000:
        bucket = "HIGH"
    elif loan_amnt >= 10000:
        bucket = "MEDIUM"
    else:
        bucket = "LOW"

    priority_map = {
        "ALTO": {"HIGH": 1, "MEDIUM": 2, "LOW": 3},
        "MEDIO": {"HIGH": 2, "MEDIUM": 3, "LOW": 4},
        "BAIXO": {"HIGH": 3, "MEDIUM": 4, "LOW": 5},
    }

    return priority_map[risk][bucket]