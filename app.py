import os
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="EGFR Drug Discovery Pipeline", layout="wide")

st.title("End-to-End AI-Assisted Drug Discovery Pipeline for EGFR Inhibitors")
st.caption("Demo dashboard for EGFR/NSCLC data processing, QSAR, ADMET, ranking, validation, uncertainty, and generative outputs.")

st.markdown(
    """
This app is designed for same-day deployment. It presents the full project workflow using uploaded or precomputed CSV outputs from your Colab modules.

Expected workflow: ChEMBL collection -> curation -> structure enrichment -> descriptors/fingerprints -> QSAR -> ADMET -> ranking -> audit -> validation -> uncertainty -> generation/optimization.
"""
)

DATA_DIR = Path("data")
DEFAULT_FILES = {
    "Raw EGFR activity": "egfr_ic50_full.csv",
    "NSCLC filtered activity": "egfr_nsclc_only.csv",
    "Unique pIC50 set": "egfr_nsclc_unique_pIC50_clean.csv",
    "SMILES/InChI set": "egfr_nsclc_unique_pIC50_smiles_inchi.csv",
    "Descriptors + fingerprints": "descriptors_with_fingerprints.csv",
    "Module 4 ready set": "module4_ready.csv",
    "Main model metrics": "module4_main_model_metrics.csv",
    "Comparison model metrics": "module4_comparison_model_metrics.csv",
    "Test predictions": "module4_test_predictions.csv",
    "ADMET raw": "module6_admet.csv",
    "ADMET summary": "module6_output.csv",
    "Ranking tracking": "module7_score_tracking.csv",
    "Ranked molecules": "module7_ranked_molecules.csv",
    "Activity cliffs": "module10_activity_cliffs_audit.csv",
    "Activity cliff summary": "module10_activity_cliff_summary.csv",
    "Validation results": "module11_validation_results.csv",
    "Ensemble uncertainty": "module12_ensemble_uncertainty.csv",
    "MC uncertainty": "module12_mc_dropout_uncertainty.csv",
    "Generated molecules": "module8_transvae_generated_molecules.csv",
    "Optimized molecules": "module9_optimized_output.csv",
}

@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def resolve_file(filename: str):
    p = DATA_DIR / filename
    return p if p.exists() else None

def show_df(title: str, filename: str, key_prefix: str):
    st.subheader(title)
    uploaded = st.file_uploader(
        f"Upload {filename}", type=["csv"], key=f"uploader_{key_prefix}"
    )

    df = None
    source = None
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        source = "uploaded"
    else:
        p = resolve_file(filename)
        if p:
            df = load_csv(str(p))
            source = str(p)

    if df is None:
        st.info(f"No file loaded for {filename}. Upload it or place it in ./data/{filename}")
        return None

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(df))
    c2.metric("Columns", len(df.columns))
    c3.caption(f"Source: {source}")

    st.dataframe(df.head(200), use_container_width=True)
    st.download_button(
        label=f"Download {filename}",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=f"download_{key_prefix}",
    )
    return df

with st.sidebar:
    st.header("App guide")
    st.markdown(
        """
- Use uploaded CSVs from Colab, or place them in a `data/` folder.
- This version is optimized for deployment stability.
- Heavy steps are shown from precomputed outputs rather than rerunning all chemistry jobs live.
"""
    )
    st.subheader("Expected files")
    st.code("\n".join(DEFAULT_FILES.values()), language="text")

overview, data_tab, model_tab, admet_tab, quality_tab, gen_tab = st.tabs([
    "Overview", "Data", "Modeling", "ADMET & Ranking", "Quality & Validation", "Generation"
])

with overview:
    st.markdown(
        """
### Project scope
This dashboard maps to the final project requirements:
1. Collect molecules from ChEMBL
2. Predict EGFR activity
3. Predict ADMET
4. Rank molecules with multiple objectives
5. Audit data quality and activity cliffs
6. Compare validation strategies
7. Estimate prediction uncertainty
8. Show generated and optimized molecules

### Deployment note
This app is intentionally lightweight for Streamlit deployment. Precomputed CSV outputs make it much more reliable than rerunning every module online.
"""
    )

with data_tab:
    show_df("Raw EGFR activity", DEFAULT_FILES["Raw EGFR activity"], "raw")
    show_df("NSCLC filtered activity", DEFAULT_FILES["NSCLC filtered activity"], "nsclc")
    show_df("Unique pIC50 set", DEFAULT_FILES["Unique pIC50 set"], "pic50")
    show_df("SMILES and InChI", DEFAULT_FILES["SMILES/InChI set"], "smiles")
    show_df("Descriptors and fingerprints", DEFAULT_FILES["Descriptors + fingerprints"], "fp")

with model_tab:
    main_df = show_df("Main model metrics", DEFAULT_FILES["Main model metrics"], "main_metrics")
    comp_df = show_df("Comparison model metrics", DEFAULT_FILES["Comparison model metrics"], "comp_metrics")
    pred_df = show_df("Test predictions", DEFAULT_FILES["Test predictions"], "preds")

    if main_df is not None and not main_df.empty and "roc_auc" in main_df.columns:
        best = main_df.sort_values("roc_auc", ascending=False).iloc[0]
        st.success(f"Best main model: {best.get('model', 'N/A')} | ROC-AUC: {best.get('roc_auc', 'N/A')}")

with admet_tab:
    show_df("ADMET raw output", DEFAULT_FILES["ADMET raw"], "admet_raw")
    admet_summary = show_df("ADMET summary", DEFAULT_FILES["ADMET summary"], "admet_summary")
    ranking = show_df("Ranked molecules", DEFAULT_FILES["Ranked molecules"], "ranked")

    if ranking is not None and not ranking.empty and "final_score" in ranking.columns:
        st.subheader("Top candidates")
        cols = [c for c in ["molecule_chembl_id", "smiles", "pic50", "final_score"] if c in ranking.columns]
        st.dataframe(ranking[cols].head(10), use_container_width=True)

with quality_tab:
    show_df("Activity cliffs", DEFAULT_FILES["Activity cliffs"], "cliffs")
    show_df("Activity cliff summary", DEFAULT_FILES["Activity cliff summary"], "cliff_summary")
    validation = show_df("Validation results", DEFAULT_FILES["Validation results"], "validation")
    show_df("Ensemble uncertainty", DEFAULT_FILES["Ensemble uncertainty"], "ens_unc")
    show_df("MC uncertainty", DEFAULT_FILES["MC uncertainty"], "mc_unc")

    if validation is not None and not validation.empty:
        st.markdown("### Validation insight")
        st.dataframe(validation, use_container_width=True)

with gen_tab:
    show_df("Generated molecules", DEFAULT_FILES["Generated molecules"], "gen_mols")
    show_df("Optimized molecules", DEFAULT_FILES["Optimized molecules"], "opt_mols")

st.markdown("---")
st.markdown("Built for deployment-focused presentation of the EGFR inhibitor discovery pipeline.")