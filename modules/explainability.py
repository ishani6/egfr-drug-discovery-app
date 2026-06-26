"""
Module 4b — Explainability
SHAP-style feature importance using model's built-in feature_importances_.
Highlights top Morgan bits and maps them back to substructure descriptions.
"""

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
import io, base64

from modules.features import morgan_fingerprint
from modules.activity_prediction import _load_model

RDLogger.DisableLog("rdApp.*")


def explain_prediction(smiles: str, model_name: str = "XGBoost", top_n: int = 15) -> dict:
    """
    Return top contributing Morgan bits for a prediction.
    Uses model feature_importances_ multiplied by fingerprint value.
    """
    model = _load_model(model_name)
    if model is None:
        return {"error": "Model not trained yet."}

    fp = morgan_fingerprint(smiles)
    if fp is None:
        return {"error": "Invalid SMILES."}

    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return {"error": f"{model_name} does not expose feature importances."}

    # Contribution = importance × fingerprint bit value
    contributions = importances * fp
    top_idx = np.argsort(contributions)[::-1][:top_n]

    top_features = []
    for idx in top_idx:
        if contributions[idx] > 0:
            top_features.append({
                "feature": f"Morgan bit {idx}",
                "contribution": round(float(contributions[idx]), 6),
                "bit_set": bool(fp[idx]),
            })

    return {
        "smiles": smiles,
        "model": model_name,
        "top_features": top_features,
        "total_contributing_bits": int((contributions > 0).sum()),
    }


def mol_to_svg(smiles: str, width: int = 300, height: int = 200) -> str:
    """Render molecule as inline SVG string (for Gradio HTML component)."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return "<p>Invalid SMILES</p>"
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg


def feature_importance_table(smiles: str, model_name: str = "XGBoost") -> pd.DataFrame:
    r = explain_prediction(smiles, model_name)
    if "error" in r:
        return pd.DataFrame([{"error": r["error"]}])
    rows = [
        {"Feature": f["feature"], "Contribution": f["contribution"]}
        for f in r["top_features"]
    ]
    return pd.DataFrame(rows)
