---
title: EGFR Molecule Discovery Platform
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# 🧬 EGFR Molecule Discovery Platform

An end-to-end AI-assisted drug discovery pipeline for EGFR inhibitor discovery.

## Pipeline stages

| Tab | Stage |
|-----|-------|
| 01 | Data collection — bundled ChEMBL EGFR dataset |
| 02 | Activity prediction — RF / XGBoost / LightGBM on Morgan fingerprints |
| 03 | ADMET prediction — rule-based 6-property panel |
| 04 | Explainability — feature attribution via model importances |
| 05 | Molecule generation — RDKit structural mutation |
| 06 | Multi-objective ranking — composite scoring |
| 07 | Active learning loop — uncertainty-based acquisition |
| 08 | Experiment tracking — lightweight run log |
| 09 | Data quality audit — duplicates, noise, activity cliffs |

## Usage

1. Open **Tab 01** → click **Load & Clean Dataset**
2. Open **Tab 02** → click **Train All Models** (required before prediction/ranking)
3. Work through the tabs in order, or jump to any stage
