import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE


def create_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    # loan-to-income: simple but useful risk signal
    if "loan_amnt" in df.columns and "annual_inc" in df.columns:
        df["loan_to_income"] = df["loan_amnt"] / (df["annual_inc"] + 1)

    # installment-to-income
    if "installment" in df.columns and "annual_inc" in df.columns:
        df["installment_to_income"] = df["installment"] / (df["annual_inc"] / 12 + 1)

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def scale_features(X_train, X_test):
    # fit only on train — avoids data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def handle_imbalance(X_train, y_train, random_state: int = 42):
    # SMOTE to handle class imbalance — defaults are rare in most datasets
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"After SMOTE — class counts: {dict(zip(*np.unique(y_res, return_counts=True)))}")
    return X_res, y_res
