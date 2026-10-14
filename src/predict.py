import joblib
import pandas as pd


def load_artifacts(model_path: str, scaler_path: str):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def preprocess_input(data: dict, scaler, feature_order: list) -> pd.DataFrame:
    df = pd.DataFrame([data])
    df = df[feature_order]   # make sure column order matches training
    df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)
    return df_scaled


def predict(model, input_df: pd.DataFrame) -> dict:
    prob = model.predict_proba(input_df)[:, 1][0]
    label = int(prob >= 0.5)
    return {
        "default_probability": round(float(prob), 4),
        "prediction": label,
    }
