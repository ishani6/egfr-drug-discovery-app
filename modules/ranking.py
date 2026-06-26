"""
Module 7 — Multi-Objective Ranking
Scoring formula from project spec:
  Final Score = 0.4*Potency + 0.2*Solubility + 0.2*Toxicity + 0.2*SA
Pareto front identification included.
"""

import numpy as np
import pandas as pd
from modules.activity_prediction import predict_single
from modules.admet import predict_admet


def score_molecule(
    smiles: str,
    w_activity: float = 0.4,
    w_solubility: float = 0.2,
    w_toxicity: float = 0.2,
    w_sa: float = 0.2,
    model_name: str = "XGBoost",
) -> dict:
    """Compute composite multi-objective score for a single molecule."""
    act = predict_single(smiles, model_name)
    adm = predict_admet(smiles)

    if "error" in act or "error" in adm:
        return {"error": act.get("error") or adm.get("error"), "smiles": smiles}

    potency_score  = act["prob_active"]
    solubility     = adm["solubility"]["score"]
    toxicity_inv   = adm["toxicity"]["score"]          # higher = less toxic = better
    sa_score       = adm.get("overall", 0.5)           # proxy for synthetic accessibility

    composite = (
        w_activity   * potency_score +
        w_solubility * solubility    +
        w_toxicity   * toxicity_inv  +
        w_sa         * sa_score
    )

    return {
        "smiles":         smiles,
        "composite":      round(float(composite), 4),
        "potency":        round(float(potency_score), 4),
        "solubility":     round(float(solubility), 4),
        "toxicity":       round(float(toxicity_inv), 4),
        "sa_score":       round(float(sa_score), 4),
        "admet_pass":     adm["overall_pass"],
        "prob_active":    act["prob_active"],
        "uncertainty":    act["uncertainty"],
    }


def rank_molecules(
    smiles_list: list[str],
    w_activity: float = 0.4,
    w_solubility: float = 0.2,
    w_toxicity: float = 0.2,
    w_sa: float = 0.2,
    model_name: str = "XGBoost",
) -> pd.DataFrame:
    """Score and rank a list of SMILES strings."""
    rows = []
    for smi in smiles_list:
        r = score_molecule(smi, w_activity, w_solubility, w_toxicity, w_sa, model_name)
        if "error" not in r:
            rows.append(r)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values("composite", ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = "rank"
    return df.reset_index()


def pareto_front(df: pd.DataFrame, obj1: str = "potency", obj2: str = "toxicity") -> pd.DataFrame:
    """Return the Pareto-optimal subset from a ranked dataframe."""
    if df.empty or obj1 not in df.columns or obj2 not in df.columns:
        return df

    points = df[[obj1, obj2]].values
    is_pareto = np.ones(len(points), dtype=bool)

    for i, p in enumerate(points):
        if is_pareto[i]:
            # Dominated if some other point is better on both objectives
            is_pareto[is_pareto] = ~np.all(
                points[is_pareto] >= p, axis=1
            ) | np.all(points[is_pareto] == p, axis=1)
            is_pareto[i] = True

    return df[is_pareto].copy()
