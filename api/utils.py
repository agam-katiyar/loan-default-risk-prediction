import os
import joblib
import pandas as pd


MODEL_PATH = os.getenv("MODEL_PATH", "models/xgboost_final.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.pkl")

# must match exactly the column order used during training (after dropping monthly_debt & income_per_dependent)
FEATURE_ORDER = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
    "total_past_due",   # engineered: sum of all past-due columns
    "has_past_due",     # engineered: binary flag
]


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def get_risk_level(prob: float) -> str:
    if prob < 0.3:
        return "low"
    elif prob < 0.6:
        return "medium"
    return "high"


def run_prediction(model, scaler, data: dict) -> dict:
    # map API-friendly snake_case names → original training column names
    past_30_59 = data["past_due_30_59_days"]
    past_90    = data["times_90_days_late"]
    past_60_89 = data["past_due_60_89_days"]

    row = {
        "RevolvingUtilizationOfUnsecuredLines": data["revolving_utilization"],
        "age":                                  data["age"],
        "NumberOfTime30-59DaysPastDueNotWorse": past_30_59,
        "DebtRatio":                            data["debt_ratio"],
        "MonthlyIncome":                        data.get("monthly_income") or 0.0,
        "NumberOfOpenCreditLinesAndLoans":       data["open_credit_lines"],
        "NumberOfTimes90DaysLate":              past_90,
        "NumberRealEstateLoansOrLines":         data["real_estate_loans"],
        "NumberOfTime60-89DaysPastDueNotWorse": past_60_89,
        "NumberOfDependents":                   data.get("num_dependents") or 0,
        # engineered features — replicate notebook 02 logic
        "total_past_due": past_30_59 + past_90 + past_60_89,
        "has_past_due":   int((past_30_59 + past_90 + past_60_89) > 0),
    }

    df = pd.DataFrame([row])[FEATURE_ORDER]
    df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)

    prob = model.predict_proba(df_scaled)[:, 1][0]

    return {
        "default_probability": round(float(prob), 4),
        "prediction":          int(prob >= 0.5),
        "risk_level":          get_risk_level(prob),
    }
