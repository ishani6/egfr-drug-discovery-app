"""
Module 6 — ADMET Prediction
Rule-based + heuristic ADMET panel derived from RDKit descriptors.
No external model checkpoint required.
"""

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, Lipinski

RDLogger.DisableLog("rdApp.*")


def predict_admet(smiles: str) -> dict:
    """
    Predict 6-property ADMET panel for a SMILES string.
    Returns a dict with numeric scores and pass/fail flags.
    """
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return {"error": "Invalid SMILES"}

    mw    = Descriptors.MolWt(mol)
    logp  = Crippen.MolLogP(mol)
    tpsa  = rdMolDescriptors.CalcTPSA(mol)
    hba   = Lipinski.NumHAcceptors(mol)
    hbd   = Lipinski.NumHDonors(mol)
    rb    = Lipinski.NumRotatableBonds(mol)
    rings = rdMolDescriptors.CalcNumRings(mol)
    fsp3  = rdMolDescriptors.CalcFractionCSP3(mol)
    arom  = rdMolDescriptors.CalcNumAromaticRings(mol)

    # ── Absorption (Caco-2 / oral bioavailability heuristic) ─────────────
    abs_score = 1.0
    if mw > 500:   abs_score -= 0.25
    if logp > 5:   abs_score -= 0.20
    if tpsa > 140: abs_score -= 0.30
    if hbd > 5:    abs_score -= 0.15
    abs_score = max(0.0, abs_score)
    abs_pass  = abs_score >= 0.5

    # ── Distribution (BBB penetration / Vd) ──────────────────────────────
    dist_score = 1.0
    if tpsa > 90:  dist_score -= 0.30
    if mw > 450:   dist_score -= 0.20
    if hba > 8:    dist_score -= 0.20
    dist_score = max(0.0, dist_score)
    dist_pass  = dist_score >= 0.5

    # ── Metabolism (CYP3A4 / CYP2D6 flag) ────────────────────────────────
    meta_score = 1.0
    if arom > 3:   meta_score -= 0.25
    if rings > 5:  meta_score -= 0.20
    if logp > 4:   meta_score -= 0.15
    meta_score = max(0.0, meta_score)
    meta_pass  = meta_score >= 0.5

    # ── Excretion (clearance / half-life) ────────────────────────────────
    excr_score = 1.0
    if mw > 400:   excr_score -= 0.20
    if logp < 1:   excr_score -= 0.15
    if rb > 10:    excr_score -= 0.15
    excr_score = max(0.0, excr_score)
    excr_pass  = excr_score >= 0.5

    # ── Toxicity (hERG, AMES heuristic) ──────────────────────────────────
    tox_score = 1.0
    if logp > 4.5: tox_score -= 0.25
    if arom > 3:   tox_score -= 0.20
    if mw < 200:   tox_score -= 0.15
    tox_score = max(0.0, tox_score)
    tox_pass  = tox_score >= 0.5

    # ── Solubility (ESOL-like heuristic) ─────────────────────────────────
    sol_score = max(0.0, 1.0 - 0.1 * logp - 0.001 * mw + 0.01 * fsp3)
    sol_score = min(1.0, sol_score)
    sol_pass  = sol_score >= 0.5

    overall = np.mean([abs_score, dist_score, meta_score, excr_score, tox_score, sol_score])

    return {
        "absorption":   {"score": round(abs_score, 2),  "pass": abs_pass},
        "distribution": {"score": round(dist_score, 2), "pass": dist_pass},
        "metabolism":   {"score": round(meta_score, 2), "pass": meta_pass},
        "excretion":    {"score": round(excr_score, 2), "pass": excr_pass},
        "toxicity":     {"score": round(tox_score, 2),  "pass": tox_pass},
        "solubility":   {"score": round(sol_score, 2),  "pass": sol_pass},
        "overall":      round(float(overall), 2),
        "overall_pass": float(overall) >= 0.6,
        "descriptors": {
            "MW": round(mw, 1), "LogP": round(logp, 2),
            "TPSA": round(tpsa, 1), "HBA": hba, "HBD": hbd,
        }
    }


def admet_summary_table(smiles_list: list) -> list[dict]:
    """Return list of ADMET result dicts for a list of SMILES."""
    rows = []
    for smi in smiles_list:
        r = predict_admet(smi)
        if "error" in r:
            continue
        rows.append({
            "smiles":       smi,
            "absorption":   "✅" if r["absorption"]["pass"]   else "❌",
            "distribution": "✅" if r["distribution"]["pass"] else "❌",
            "metabolism":   "✅" if r["metabolism"]["pass"]   else "❌",
            "excretion":    "✅" if r["excretion"]["pass"]    else "❌",
            "toxicity":     "✅" if r["toxicity"]["pass"]     else "❌",
            "solubility":   "✅" if r["solubility"]["pass"]   else "❌",
            "overall":      r["overall"],
            "pass":         "✅ Pass" if r["overall_pass"] else "❌ Fail",
        })
    return rows
