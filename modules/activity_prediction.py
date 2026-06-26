"""
Module 4 — Activity Prediction (QSAR)
RF / XGBoost / LightGBM trained on Morgan fingerprints.
Mirrors your notebook exactly.
"""

import numpy as np
import pandas as pd
import pickle, os, warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from modules.features import morgan_fingerprint, featurise_dataframe

warnings.filterwarnings("ignore")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

_MODELS: dict = {}  # cache


def _model_path(name: str) -> str:
    return os.path.join(MODEL_DIR, f"{name}.pkl")


def train_models(df: pd.DataFrame) -> dict:
    """Train RF, XGBoost, LightGBM on the given dataframe. Returns metrics dict."""
    feat_df = featurise_dataframe(df)
    fp_cols = [c for c in feat_df.columns if c.startswith("morgan_")]
    X = feat_df[fp_cols].values
    y = feat_df["active"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model_defs = {
        "RandomForest": RandomForestClassifier(
            n_estimators=100, random_state=42,
            class_weight="balanced", n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, eval_metric="logloss",
            n_jobs=-1, verbosity=0
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=100, learning_rate=0.05, num_leaves=15,
            min_child_samples=5, random_state=42,
            n_jobs=-1, verbosity=-1
        ),
    }

    results = {}
    for name, model in model_defs.items():
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        pred = model.predict(X_test)
        auc  = roc_auc_score(y_test, prob)
        acc  = accuracy_score(y_test, pred)
        results[name] = {"roc_auc": round(auc, 4), "accuracy": round(acc, 4)}
        with open(_model_path(name), "wb") as f:
            pickle.dump(model, f)
        _MODELS[name] = model

    return results


def _load_model(name: str):
    if name in _MODELS:
        return _MODELS[name]
    p = _model_path(name)
    if os.path.exists(p):
        with open(p, "rb") as f:
            m = pickle.load(f)
        _MODELS[name] = m
        return m
    return None


def predict_single(smiles: str, model_name: str = "XGBoost") -> dict:
    """Predict activity for a single SMILES. Returns prob, label, uncertainty."""
    model = _load_model(model_name)
    if model is None:
        return {"error": f"Model {model_name} not trained yet. Go to Tab 2 and click Train."}

    fp = morgan_fingerprint(smiles)
    if fp is None:
        return {"error": "Invalid SMILES string."}

    X = fp.reshape(1, -1)
    prob = model.predict_proba(X)[0][1]
    label = "Active" if prob >= 0.5 else "Inactive"

    # Ensemble uncertainty across all loaded models
    probs = []
    for m in _MODELS.values():
        try:
            probs.append(m.predict_proba(X)[0][1])
        except Exception:
            pass
    uncertainty = float(np.std(probs)) if len(probs) > 1 else 0.0

    return {
        "model": model_name,
        "prob_active": round(float(prob), 4),
        "label": label,
        "uncertainty": round(uncertainty, 4),
        "confidence_pct": round(float(prob) * 100, 1),
    }


def predict_batch(smiles_list: list, model_name: str = "XGBoost") -> pd.DataFrame:
    """Predict activity for a list of SMILES strings."""
    rows = []
    for smi in smiles_list:
        r = predict_single(smi, model_name)
        r["smiles"] = smi
        rows.append(r)
    return pd.DataFrame(rows)


def models_trained() -> list:
    return [n for n in ["RandomForest", "XGBoost", "LightGBM"] if os.path.exists(_model_path(n))]
