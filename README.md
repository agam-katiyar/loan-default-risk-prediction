# loan-risk-ml

Predicting loan default risk — end-to-end, from raw data to a deployed API.

Built this to get hands-on with the full ML lifecycle: messy financial data, model comparison, and wrapping it all in a FastAPI service.

Dataset: [Give Me Some Credit — Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit)

---

## What this covers

- EDA — missing values, class imbalance, feature distributions
- Feature engineering — past-due aggregates, income ratios, imputation
- Model comparison — Logistic Regression vs Random Forest vs XGBoost
- Hyperparameter tuning with Optuna (50 trials, cross-validation)
- SHAP for feature explainability
- FastAPI REST endpoint + Docker deployment

## Stack

`Python` `Pandas` `Scikit-learn` `XGBoost` `SHAP` `Optuna` `FastAPI` `Docker`

## Results

| Model | CV AUC | Test AUC |
|---|---|---|
| Logistic Regression | 0.8910 | **0.8463** ✅ |
| Random Forest | 0.9882 | 0.8309 |
| XGBoost (tuned, 50 Optuna trials) | 0.9851 | 0.8133 |

LR wins on the held-out test set. Tree models overfit to SMOTE-inflated CV folds (data leakage — see notebook 03).

**Evaluation (LR @ threshold 0.5, test set = 30,000 rows):**
- AUC-ROC: 0.846 · Average Precision: 0.363 (random baseline: 0.067)
- Caught 1,404 / 2,005 actual defaults (70% recall)
- Top SHAP features: `total_past_due` › `RevolvingUtilizationOfUnsecuredLines` › `has_past_due`

## Progress

| Phase | Status |
|---|---|
| EDA | ✅ Done |
| Feature Engineering | ✅ Done |
| Modeling | ✅ Done |
| Evaluation + SHAP | ✅ Done |
| API + Docker | ✅ Done |

## Structure

```
loan-risk-ml/
├── data/
│   ├── raw/                  # original dataset (not tracked)
│   └── processed/            # cleaned + engineered features
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_evaluation.ipynb
├── src/                      # reusable modules
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── api/                      # FastAPI app
│   ├── main.py
│   ├── schemas.py
│   └── utils.py
├── models/                   # saved model files (not tracked)
├── reports/
│   └── figures/              # plots saved from notebooks
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/agam-katiyar/loan-default-risk-prediction.git
cd loan-risk-ml
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Download the dataset from Kaggle and place `cs-training.csv` in `data/raw/`.

## Run the API

```bash
# local
uvicorn api.main:app --reload

# docker
docker-compose up --build
```

API docs at `http://localhost:8000/docs`

---

*Work in progress — API and Docker testing next.*
