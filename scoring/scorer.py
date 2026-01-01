import joblib
import pandas as pd
import logging

from pipeline.features import build_features

# =========================
# Logging
# =========================
logger = logging.getLogger("riskflow.scorer")

# =========================
# Contrato de Features
# (input esperado da API)
# =========================
REQUIRED_FEATURES = [
    "loan_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "purpose",
    "dti",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
]

# =========================
# Carregar modelo
# =========================
MODEL_PATH = "models/credit_model.pkl"
model = joblib.load(MODEL_PATH)

# =========================
# Função de Scoring
# =========================
def score_loan(payload: dict) -> dict:
    """
    Recebe um dicionário com features do cliente e retorna:
    - prob_default (float)
    - score (1-1000)
    - risk (Baixo, Médio, Alto)
    - priority (1-5)
    """

    # -------------------------
    # 1. Validação de contrato
    # -------------------------
    missing = set(REQUIRED_FEATURES) - set(payload.keys())
    if missing:
        raise ValueError(f"columns are missing: {missing}")

    # -------------------------
    # 2. Input → DataFrame
    # -------------------------
    X = pd.DataFrame([payload])

    # -------------------------
    # 3. Feature Engineering
    # (MESMO pipeline do treino)
    # -------------------------
    X = build_features(X)

    # -------------------------
    # 4. Probabilidade de default
    # -------------------------
    prob_default = float(model.predict_proba(X)[:, 1][0])

    # -------------------------
    # 5. Score (1–1000)
    # -------------------------
    score = int(round(1000 * (1 - prob_default)))
    score = max(1, min(score, 1000))

    # -------------------------
    # 6. Faixa de risco
    # -------------------------
    if prob_default <= 0.2:
        risk = "Baixo"
        priority = 5
    elif prob_default <= 0.5:
        risk = "Médio"
        priority = 3
    else:
        risk = "Alto"
        priority = 1

    logger.info(
        f"Score generated | PD={prob_default:.4f} | Score={score} | Risk={risk}"
    )

    # -------------------------
    # 7. Resposta padronizada
    # -------------------------
    return {
        "prob_default": round(prob_default, 4),
        "score": score,
        "risk": risk,
        "priority": priority,
    }
