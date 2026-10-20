from fastapi import FastAPI, HTTPException
from api.schemas import LoanApplication, PredictionResponse
from api.utils import load_artifacts, run_prediction

app = FastAPI(
    title="Loan Default Risk API",
    description="Returns default probability and risk level for a loan application.",
    version="1.0.0",
)

# load once at startup — not on every request
model, scaler = load_artifacts()


@app.get("/")
def root():
    return {"status": "running", "model": "XGBoost Loan Default Predictor"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication):
    try:
        result = run_prediction(model, scaler, application.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
