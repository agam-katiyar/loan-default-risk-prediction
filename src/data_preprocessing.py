import pandas as pd
import numpy as np


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def handle_missing(df: pd.DataFrame, threshold: float = 0.4) -> pd.DataFrame:
    # drop columns where more than 40% of values are missing
    df = df.dropna(thresh=int((1 - threshold) * len(df)), axis=1)

    # fill numeric with median, categorical with mode
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


def remove_outliers(df: pd.DataFrame, col: str, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    # clip at percentile boundaries instead of hard-dropping
    q_low = df[col].quantile(lower)
    q_high = df[col].quantile(upper)
    return df[(df[col] >= q_low) & (df[col] <= q_high)]


def split_features_target(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
