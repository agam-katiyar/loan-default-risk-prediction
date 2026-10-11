import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier


def get_models() -> dict:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "xgboost": XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
    }


def cross_validate_models(models: dict, X, y, cv: int = 5) -> dict:
    # stratified CV to preserve class ratio across folds
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    results = {}

    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc")
        results[name] = {"mean_auc": round(scores.mean(), 4), "std": round(scores.std(), 4)}
        print(f"{name:25s} AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")

    return results


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def save_model(model, path: str):
    joblib.dump(model, path)
    print(f"saved → {path}")


def load_model(path: str):
    return joblib.load(path)
