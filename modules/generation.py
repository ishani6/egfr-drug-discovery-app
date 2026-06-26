"""
Module 8 — Molecule Generation
Generates novel molecules via SMILES-based structural mutations.
Uses RDKit fragment operations — no large model checkpoint required on HF Spaces.
Falls back gracefully if moses is available.
"""

import random
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolDescriptors, Crippen, Lipinski, Descriptors
from rdkit.Chem.rdchem import RWMol

RDLogger.DisableLog("rdApp.*")

# Common substituents used in EGFR inhibitor medicinal chemistry
_SUBSTITUENTS = [
    "F", "Cl", "Br", "C", "CC", "OC", "N", "NC", "CF",
    "C(F)(F)F", "OCC", "NCC", "c1ccccc1", "C1CCNCC1",
    "C(=O)N", "S(=O)(=O)N",
]

_EGFR_SCAFFOLDS = [
    "c1cc2ncnc(N)c2cc1",           # quinazoline core
    "c1cnc2ccccc2c1",              # quinoline
    "c1cc2[nH]cnc2cc1",            # 7-azaindole
    "c1ccc2[nH]ccc2c1",            # indole
    "c1cncc(N)c1",                 # aminopyridine
    "COc1cc2ncnc(N)c2cc1OC",       # gefitinib scaffold
    "C#Cc1cccc(N)c1",              # erlotinib-like
]


def _mutate_smiles(smiles: str, seed: int = 0) -> str | None:
    """Apply a random structural mutation to a valid SMILES."""
    random.seed(seed)
    np.random.seed(seed)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Strategy: append a random substituent to a random atom
    rw = RWMol(mol)
    atoms = [a for a in rw.GetAtoms() if a.GetAtomicNum() == 6]
    if not atoms:
        return None

    try:
        target_atom = random.choice(atoms)
        sub_smi = random.choice(_SUBSTITUENTS)
        sub_mol = Chem.MolFromSmiles(sub_smi)
        if sub_mol is None:
            return None

        combined = Chem.RWMol(Chem.CombineMols(rw, sub_mol))
        n_orig = rw.GetNumAtoms()
        combined.AddBond(target_atom.GetIdx(), n_orig, Chem.BondType.SINGLE)
        Chem.SanitizeMol(combined)
        return Chem.MolToSmiles(combined)
    except Exception:
        return None


def generate_molecules(
    seed_smiles: list[str],
    n_generate: int = 20,
    objectives: list[str] = None,
) -> pd.DataFrame:
    """
    Generate novel candidate molecules by mutating seed SMILES.
    Returns a dataframe with SMILES + basic properties.
    """
    if objectives is None:
        objectives = ["activity", "admet"]

    candidates = []
    seen = set(seed_smiles)
    attempt = 0

    while len(candidates) < n_generate and attempt < n_generate * 10:
        seed = random.choice(seed_smiles + _EGFR_SCAFFOLDS)
        new_smi = _mutate_smiles(seed, seed=attempt)
        attempt += 1

        if new_smi is None or new_smi in seen:
            continue

        mol = Chem.MolFromSmiles(new_smi)
        if mol is None:
            continue

        seen.add(new_smi)

        mw   = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        hbd  = Lipinski.NumHDonors(mol)
        hba  = Lipinski.NumHAcceptors(mol)

        # Quick Lipinski filter
        ro5_pass = (mw <= 500 and logp <= 5 and hba <= 10 and hbd <= 5)

        # Novelty: 1 if not in seed set
        novelty = 1.0 if new_smi not in set(seed_smiles) else 0.0

        # Synthetic accessibility heuristic (lower ring count + lower MW = easier)
        rings = rdMolDescriptors.CalcNumRings(mol)
        sa_score = max(0.0, 1.0 - 0.1 * rings - 0.001 * max(0, mw - 200))

        candidates.append({
            "smiles":    new_smi,
            "mol_wt":   round(mw, 1),
            "logp":     round(logp, 2),
            "tpsa":     round(tpsa, 1),
            "ro5_pass": ro5_pass,
            "novelty":  round(novelty, 2),
            "sa_score": round(sa_score, 2),
        })

    return pd.DataFrame(candidates)
