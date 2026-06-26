"""
Module 2/3 — Molecular Features
Classical descriptors + Morgan fingerprints + MACCS keys from your notebook.
"""

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, Lipinski, MACCSkeys
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

_morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def compute_descriptors(smiles: str) -> dict | None:
    """Return classical descriptors for a single SMILES string."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return {
        "mol_wt":           round(Descriptors.MolWt(mol), 2),
        "tpsa":             round(rdMolDescriptors.CalcTPSA(mol), 2),
        "logp":             round(Crippen.MolLogP(mol), 2),
        "hba":              Lipinski.NumHAcceptors(mol),
        "hbd":              Lipinski.NumHDonors(mol),
        "rotatable_bonds":  Lipinski.NumRotatableBonds(mol),
        "ring_count":       rdMolDescriptors.CalcNumRings(mol),
        "fraction_csp3":    round(rdMolDescriptors.CalcFractionCSP3(mol), 3),
        "num_atoms":        mol.GetNumAtoms(),
        "num_heavy_atoms":  mol.GetNumHeavyAtoms(),
    }


def morgan_fingerprint(smiles: str) -> np.ndarray | None:
    """2048-bit Morgan fingerprint as numpy array."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    fp = _morgan_gen.GetFingerprint(mol)
    arr = np.zeros((2048,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def maccs_fingerprint(smiles: str) -> np.ndarray | None:
    """167-bit MACCS keys as numpy array."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros((167,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def featurise_dataframe(df: pd.DataFrame, smiles_col: str = "Smiles") -> pd.DataFrame:
    """Add descriptor columns to dataframe."""
    desc_rows = []
    morgan_rows = []

    for smi in df[smiles_col]:
        d = compute_descriptors(smi)
        fp = morgan_fingerprint(smi)
        desc_rows.append(d if d else {})
        morgan_rows.append(fp if fp is not None else np.zeros(2048, dtype=np.int8))

    desc_df = pd.DataFrame(desc_rows)
    morgan_df = pd.DataFrame(
        np.vstack(morgan_rows),
        columns=[f"morgan_{i}" for i in range(2048)]
    )

    result = pd.concat([df.reset_index(drop=True), desc_df, morgan_df], axis=1)
    return result


def lipinski_pass(smiles: str) -> dict:
    """Return Lipinski Ro5 pass/fail for each rule."""
    d = compute_descriptors(smiles)
    if d is None:
        return {}
    return {
        "MW ≤ 500":   d["mol_wt"] <= 500,
        "LogP ≤ 5":   d["logp"] <= 5,
        "HBA ≤ 10":   d["hba"] <= 10,
        "HBD ≤ 5":    d["hbd"] <= 5,
        "TPSA ≤ 140": d["tpsa"] <= 140,
    }
