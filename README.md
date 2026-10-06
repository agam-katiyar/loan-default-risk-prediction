# loan-risk-ml

Predicting loan default risk — end-to-end, from raw data to a deployed API.

Built this to get hands-on with the full ML lifecycle: messy financial data, model comparison, and wrapping it all in a FastAPI service.

Dataset: [Give Me Some Credit — Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit)

---

## What this covers

- EDA — missing values, class imbalance, feature distributions
- Feature engineering — ratios, encoding, scaling
- Model comparison — Logistic Regression vs Random Forest vs XGBoost
- Hyperparameter tuning with Optuna + cross-validation
- SHAP for feature explainability
- FastAPI REST endpoint + Docker deployment

## Stack

`Python` `Pandas` `Scikit-learn` `XGBoost` `SHAP` `Optuna` `FastAPI` `Docker`

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

## Progress

| Phase | Status |
|---|---|
| EDA | 🔄 In progress |
| Feature Engineering | ⬜ Todo |
| Modeling | ⬜ Todo |
| Tuning + SHAP | ⬜ Todo |
| API + Docker | ⬜ Todo |

## Setup

```bash
git clone https://github.com/yourusername/loan-risk-ml.git
cd loan-risk-ml
pip install -r requirements.txt
```

Download the dataset from Kaggle and place it in `data/raw/`.

## Run the API

```bash
# local
uvicorn api.main:app --reload

# docker
docker-compose up --build
```

API docs available at `http://localhost:8000/docs`

---

*Work in progress — updating as I go.*
