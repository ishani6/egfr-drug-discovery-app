import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from chembl_webresource_client.new_client import new_client
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize

def fetch_egfr_bioactivity_from_chembl():
    target_id = "CHEMBL203"
    activities = new_client.activity.filter(
        target_chembl_id=target_id,
        standard_type__in=["IC50", "Ki"]
    )
    df = pd.DataFrame(activities)
    keep_cols = [
        "activity_id", "assay_chembl_id", "document_chembl_id",
        "molecule_chembl_id", "standard_type", "standard_relation",
        "standard_value", "standard_units", "pchembl_value",
        "assay_type", "confidence_score", "canonical_smiles"
    ]
    df = df[[c for c in keep_cols if c in df.columns]].copy()
    df = df.dropna(subset=["canonical_smiles", "standard_value"])
    if "standard_relation" in df.columns:
        df = df[df["standard_relation"].fillna("=") == "="]
    if "standard_units" in df.columns:
        df = df[df["standard_units"].fillna("").str.lower().eq("nm")]
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df = df.dropna(subset=["standard_value"])
    df["pIC50"] = -np.log10(df["standard_value"] * 1e-9)
    df["clean_smiles"] = df["canonical_smiles"]
    return df.reset_index(drop=True)

def standardize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    parent = rdMolStandardize.FragmentParent(mol)
    try:
        Chem.SanitizeMol(parent)
    except:
        return None
    return Chem.MolToSmiles(parent, canonical=True)

def curate_egfr_dataset(df):
    out = df.copy()
    out["clean_smiles"] = out["canonical_smiles"].apply(standardize_smiles)
    out = out.dropna(subset=["clean_smiles"])
    out = out.sort_values("pIC50", ascending=False)
    out = out.drop_duplicates(subset=["clean_smiles"], keep="first")
    return out.reset_index(drop=True)

def calculate_rdkit_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return pd.Series([np.nan] * 5, index=["MolWt", "LogP", "TPSA", "HBD", "HBA"])
    return pd.Series([
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol)
    ], index=["MolWt", "LogP", "TPSA", "HBD", "HBA"])

def generate_morgan_fingerprint(smiles, n_bits=2048, radius=2):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.full(n_bits, np.nan)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def build_feature_table(df, n_bits=2048):
    out = df.copy()
    desc = out["clean_smiles"].apply(calculate_rdkit_descriptors).reset_index(drop=True)
    fps = out["clean_smiles"].apply(lambda s: generate_morgan_fingerprint(s, n_bits=n_bits))
    fp_df = pd.DataFrame(fps.tolist(), columns=[f"fp_{i}" for i in range(n_bits)])
    return pd.concat([out.reset_index(drop=True), desc, fp_df], axis=1)

def load_csv_if_exists(path):
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    return None

def load_prediction_model(path):
    path = Path(path)
    if path.exists():
        return joblib.load(path)
    return None

def get_model_feature_columns(df, model, base_cols=None):
    if base_cols is None:
        base_cols = ["MolWt", "LogP", "TPSA", "HBD", "HBA"]
    fp_cols = [c for c in df.columns if c.startswith("fp_")]
    cols = base_cols + fp_cols
    if hasattr(model, "feature_names_in_"):
        cols = [c for c in model.feature_names_in_ if c in df.columns]
    return cols

def predict_with_model(model, df, feature_cols):
    X = df[feature_cols].fillna(0)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)