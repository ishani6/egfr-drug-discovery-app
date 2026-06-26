"""
Module 9 — Active Learning Loop
Uncertainty-based acquisition: select the most informative molecules
from the unlabeled pool, simulate labeling, retrain.
"""

import numpy as np
import pandas as pd
import json, os, pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from modules.features import morgan_fingerprint
from modules.data_collection import load_and_clean

STATE_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "al_state.json")


def _load_state() -> dict:
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"iteration": 0, "labeled_indices": [], "auc_history": []}


def _save_state(state: dict):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f)


def _get_features(df: pd.DataFrame) -> np.ndarray:
    fps = []
    for smi in df["Smiles"]:
        fp = morgan_fingerprint(smi)
        fps.append(fp if fp is not None else np.zeros(2048))
    return np.vstack(fps)


def run_iteration(n_query: int = 10) -> dict:
    """
    Run one active learning iteration:
    1. Load dataset
    2. Split into labeled (seed) + unlabeled pool
    3. Train on labeled set
    4. Predict uncertainty on pool
    5. Select top-n most uncertain molecules
    6. Simulate labeling (use ground truth pIC50)
    7. Retrain and record AUC improvement
    """
    state = _load_state()
    iteration = state["iteration"] + 1

    df, _ = load_and_clean()
    X_all = _get_features(df)
    y_all = df["active"].values

    # Seed: first 60% or already-labeled indices
    n_seed = max(int(0.6 * len(df)), 10)
    labeled_idx = list(state.get("labeled_indices") or list(range(n_seed)))
    unlabeled_idx = [i for i in range(len(df)) if i not in set(labeled_idx)]

    if len(unlabeled_idx) < n_query:
        return {"message": "Pool exhausted — all molecules labeled.", "iteration": iteration}

    # Train on labeled set
    X_labeled = X_all[labeled_idx]
    y_labeled = y_all[labeled_idx]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_labeled)

    model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=iteration)
    model.fit(X_scaled, y_labeled)

    # Uncertainty on unlabeled pool
    X_pool = scaler.transform(X_all[unlabeled_idx])
    probs = model.predict_proba(X_pool)[:, 1]
    uncertainty = np.abs(probs - 0.5)           # lowest = most uncertain
    query_local = np.argsort(uncertainty)[:n_query]
    query_global = [unlabeled_idx[i] for i in query_local]

    # Simulate labeling + retrain
    new_labeled = labeled_idx + query_global
    X_new = scaler.fit_transform(X_all[new_labeled])
    y_new = y_all[new_labeled]
    model.fit(X_new, y_new)

    # Eval on held-out 20%
    held_out = [i for i in range(len(df)) if i not in set(new_labeled)]
    if len(held_out) > 5:
        X_held = scaler.transform(X_all[held_out])
        y_held = y_all[held_out]
        auc = round(roc_auc_score(y_held, model.predict_proba(X_held)[:, 1]), 4)
    else:
        auc = 0.5

    auc_history = state.get("auc_history", []) + [auc]

    new_state = {
        "iteration": iteration,
        "labeled_indices": new_labeled,
        "auc_history": auc_history,
    }
    _save_state(new_state)

    selected_smiles = df.iloc[query_global]["Smiles"].tolist()

    return {
        "iteration": iteration,
        "labeled_count": len(new_labeled),
        "unlabeled_remaining": len(df) - len(new_labeled),
        "queries_selected": n_query,
        "val_auc": auc,
        "auc_history": auc_history,
        "selected_smiles": selected_smiles[:5],  # preview
    }


def get_al_status() -> dict:
    state = _load_state()
    return {
        "iteration":      state["iteration"],
        "labeled_count":  len(state.get("labeled_indices", [])),
        "auc_history":    state.get("auc_history", []),
    }


def reset_loop():
    if os.path.exists(STATE_PATH):
        os.remove(STATE_PATH)
