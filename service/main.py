import logging
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from scoring.scorer import score_loan, REQUIRED_FEATURES

app = FastAPI(
    title="RiskFlow Scoring API",
    version="1.0.0",
    openapi_url="/v1/openapi.json",
    docs_url="/v1/docs"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("riskflow")

# =========================
# Request
# =========================
class ScoreRequest(BaseModel):
    loan_amnt: float
    term: str
    int_rate: float
    installment: float

    grade: str
    sub_grade: str

    emp_length: Optional[str] = None
    home_ownership: str
    annual_inc: float
    verification_status: str
    purpose: str

    dti: float
    delinq_2yrs: int
    inq_last_6mths: int
    open_acc: int
    pub_rec: int

    revol_bal: float
    revol_util: float
    total_acc: int


# =========================
# Response
# =========================
class ScoreResponse(BaseModel):
    prob_default: float
    score: int
    risk: str
    priority: int


# =========================
# Features
# =========================
@app.get("/v1/features")
def get_features():
    return {
        "required_features": REQUIRED_FEATURES
    }


# =========================
# Scoring
# =========================
@app.post("/v1/score", response_model=ScoreResponse)
def score_endpoint(payload: ScoreRequest):
    logger.info("Scoring request received")

    try:
        return score_loan(payload.dict())
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected scoring error")
        raise HTTPException(status_code=500, detail="Internal scoring error")


# =========================
# Health
# =========================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True
    }