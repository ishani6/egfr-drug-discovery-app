"""
Module 1 — Data Collection & Cleaning
Tries live ChEMBL API first; falls back to bundled CSV if fetch fails.
"""

import os
import pandas as pd
import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "egfr_sample.csv")
CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data/activity.json"


def fetch_from_chembl(
    target_chembl_id: str = "CHEMBL203",
    activity_type: str = "IC50",
    min_confidence: int = 6,
    limit: int = 500,
) -> tuple[pd.DataFrame | None, str]:
    """
    Fetch bioactivity data from ChEMBL REST API.
    Returns (DataFrame, status_message) — DataFrame is None on failure.
    Paginates automatically to reach the requested limit.
    """
    try:
        import requests

        limit = int(limit)          # Gradio sliders send floats
        page_size = min(limit, 1000)  # ChEMBL max per page is 1000
        fetched = []
        offset = 0

        while len(fetched) < limit:
            params = {
                "target_chembl_id":      target_chembl_id,
                "standard_type":         activity_type,
                # NOTE: assay_type removed — was silently dropping most records
                "confidence_score__gte": int(min_confidence),
                "standard_relation":     "=",
                "limit":                 min(page_size, limit - len(fetched)),
                "offset":                offset,
                "format":                "json",
            }
            resp = requests.get(CHEMBL_API, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            activities = data.get("activities", [])
            if not activities:
                break   # no more pages

            for a in activities:
                smi = a.get("canonical_smiles", "")
                val = a.get("standard_value")
                cid = a.get("molecule_chembl_id", "")
                units = (a.get("standard_units") or "nM").strip()
                if not smi or val is None:
                    continue
                try:
                    val_f = float(val)
                except (TypeError, ValueError):
                    continue
                # Normalise to nM
                if units == "uM":
                    val_f *= 1000
                elif units == "M":
                    val_f *= 1e9
                elif units == "pM":
                    val_f /= 1000
                fetched.append({
                    "Molecule ChEMBL ID": cid,
                    "Smiles":             smi,
                    "Standard Type":      a.get("standard_type", activity_type),
                    "Standard Value":     val_f,
                    "Standard Units":     "nM",
                })

            # Check if there are more pages
            total_count = data.get("page_meta", {}).get("total_count", 0)
            offset += len(activities)
            if offset >= total_count or len(activities) == 0:
                break

        if not fetched:
            return None, "ChEMBL returned 0 records for these filters"

        return pd.DataFrame(fetched), f"ChEMBL API — {len(fetched)} records fetched"

    except Exception as e:
        return None, f"ChEMBL fetch failed: {e}"


def _clean(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Shared cleaning pipeline."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    df = df.dropna(subset=["Smiles", "Standard Value"])
    df["Standard Value"] = pd.to_numeric(df["Standard Value"], errors="coerce")
    df = df.dropna(subset=["Standard Value"])

    if "Standard Type" in df.columns:
        df = df[df["Standard Type"].isin(["IC50", "Ki", "EC50"])]

    # pIC50: assume nM units
    if "pIC50" not in df.columns:
        df["pIC50"] = -np.log10(df["Standard Value"].clip(lower=1e-3) * 1e-9)

    df["pIC50"] = pd.to_numeric(df["pIC50"], errors="coerce")
    df = df.dropna(subset=["pIC50"])
    df["pIC50"] = df["pIC50"].clip(lower=3, upper=12)

    # Validate SMILES
    def valid_smiles(smi):
        try:
            return Chem.MolFromSmiles(str(smi)) is not None
        except Exception:
            return False

    df = df[df["Smiles"].apply(valid_smiles)].reset_index(drop=True)
    df["active"]   = (df["pIC50"] >= 7.0).astype(int)
    df["_source"]  = source
    return df


def load_and_clean(
    source: str = "auto",
    target_chembl_id: str = "CHEMBL203",
    activity_type: str = "IC50",
    min_confidence: int = 6,
    limit: int = 500,
) -> tuple[pd.DataFrame, str]:
    """
    Returns (cleaned_df, status_message).
    source='auto'   → try ChEMBL first, fall back to CSV
    source='chembl' → live ChEMBL only, error if fails
    source='csv'    → bundled CSV only
    """
    limit = int(limit)
    min_confidence = int(min_confidence)

    if source in ("auto", "chembl"):
        df, msg = fetch_from_chembl(target_chembl_id, activity_type, min_confidence, limit)
        if df is not None:
            return _clean(df, "chembl"), msg
        if source == "chembl":
            return pd.DataFrame(), f"❌ {msg}"
        # auto fallback
        csv_df = pd.read_csv(DATA_PATH)
        return _clean(csv_df, "csv"), f"⚠️ {msg} — using bundled CSV instead"

    # csv only
    csv_df = pd.read_csv(DATA_PATH)
    return _clean(csv_df, "csv"), "📁 Loaded bundled sample CSV"


def get_summary(df: pd.DataFrame) -> dict:
    source = df["_source"].iloc[0] if "_source" in df.columns and len(df) > 0 else "unknown"
    return {
        "total_molecules": len(df),
        "active":          int(df["active"].sum()) if "active" in df.columns else 0,
        "inactive":        int((df["active"] == 0).sum()) if "active" in df.columns else 0,
        "avg_pic50":       round(df["pIC50"].mean(), 2) if "pIC50" in df.columns else 0,
        "source":          source,
    }
