"""
Module 10 (notebook) — Data Quality Audit
Checks duplicates, assay noise, activity cliffs. From your notebook.
"""

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

_morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)


def _canonical(smi: str):
    mol = Chem.MolFromSmiles(str(smi))
    return Chem.MolToSmiles(mol) if mol else None


def run_audit(df: pd.DataFrame, smiles_col: str = "Smiles", activity_col: str = "pIC50") -> dict:
    work = df.copy()
    work["canonical"] = work[smiles_col].apply(_canonical)
    work = work.dropna(subset=["canonical"])

    # ── Duplicates ──────────────────────────────────────────────────────
    dup_mask = work.duplicated(subset=["canonical"], keep=False)
    dup_df = work[dup_mask].copy()

    # ── Missing data ─────────────────────────────────────────────────────
    missing = work.isnull().sum().to_dict()

    # ── Assay noise (same scaffold, std > 1 log unit) ────────────────────
    noise_rows = []
    if activity_col in work.columns:
        work[activity_col] = pd.to_numeric(work[activity_col], errors="coerce")
        scaffold_groups = work.groupby("canonical")[activity_col]
        for scaffold, group in scaffold_groups:
            if len(group) > 1 and group.std() > 1.0:
                noise_rows.append({
                    "canonical": scaffold,
                    "n_assays": len(group),
                    "std_pIC50": round(group.std(), 3),
                })
    noise_df = pd.DataFrame(noise_rows)

    # ── Activity cliffs (high similarity, large activity diff) ────────────
    cliff_rows = []
    if activity_col in work.columns:
        act = work.dropna(subset=[activity_col]).copy()
        act = act.drop_duplicates(subset=["canonical"]).head(200)  # cap for speed
        fps = []
        for smi in act["canonical"]:
            mol = Chem.MolFromSmiles(smi)
            fps.append(_morgan_gen.GetFingerprint(mol) if mol else None)

        for i in range(len(act)):
            for j in range(i + 1, len(act)):
                if fps[i] is None or fps[j] is None:
                    continue
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                diff = abs(
                    act.iloc[i][activity_col] - act.iloc[j][activity_col]
                )
                if sim >= 0.80 and diff >= 2.0:
                    cliff_rows.append({
                        "mol_A": act.iloc[i]["canonical"],
                        "mol_B": act.iloc[j]["canonical"],
                        "similarity": round(sim, 3),
                        "activity_diff": round(diff, 3),
                    })

    cliff_df = pd.DataFrame(cliff_rows).sort_values(
        ["activity_diff", "similarity"], ascending=False
    ) if cliff_rows else pd.DataFrame()

    return {
        "total_rows":        len(df),
        "clean_rows":        len(work),
        "duplicate_count":   len(dup_df),
        "missing":           missing,
        "noise_cases":       len(noise_df),
        "activity_cliffs":   len(cliff_df),
        "duplicates_df":     dup_df,
        "noise_df":          noise_df,
        "cliffs_df":         cliff_df,
    }
