"""
EGFR Molecule Discovery Platform
9-tab Gradio app — deploy-ready for Hugging Face Docker Space.
"""

import gradio as gr
import pandas as pd
import numpy as np
import json, os, sys

# ── ensure modules are importable ────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from modules.data_collection   import load_and_clean, get_summary
from modules.features          import compute_descriptors, lipinski_pass
from modules.activity_prediction import train_models, predict_single, predict_batch, models_trained
from modules.admet             import predict_admet, admet_summary_table
from modules.explainability    import explain_prediction, mol_to_svg, feature_importance_table
from modules.generation        import generate_molecules
from modules.ranking           import rank_molecules, score_molecule
from modules.active_learning   import run_iteration, get_al_status, reset_loop
from modules.tracking          import log_run, get_runs_df, clear_runs
from modules.data_quality      import run_audit

# ─────────────────────────────────────────────────────────────────────────────
# Shared state (loaded once)
# ─────────────────────────────────────────────────────────────────────────────
_df_cache: dict = {}

def get_df():
    if "df" not in _df_cache:
        df, _ = load_and_clean()
        _df_cache["df"] = df
    return _df_cache["df"]


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Data Collection
# ─────────────────────────────────────────────────────────────────────────────
def tab1_load(source="auto", target="CHEMBL203", activity_type="IC50", min_conf=6, limit=500):
    df, status_msg = load_and_clean(source, target, activity_type, int(min_conf), int(limit))
    _df_cache["df"] = df
    s = get_summary(df)
    src_label = "🌐 Live ChEMBL API" if s["source"] == "chembl" else "📁 Bundled sample CSV"
    summary_md = f"""
### Dataset loaded ✅
**Status:** {status_msg}

| Metric | Value |
|--------|-------|
| Total molecules | **{s['total_molecules']}** |
| Active (pIC50 ≥ 7) | **{s['active']}** |
| Inactive | **{s['inactive']}** |
| Average pIC50 | **{s['avg_pic50']}** |
| Source | {src_label} |
"""
    cols = ["Smiles", "pIC50", "active"]
    if "Molecule ChEMBL ID" in df.columns:
        cols = ["Molecule ChEMBL ID"] + cols
    preview = df[cols].head(10) if not df.empty else df
    return summary_md, preview


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Activity Prediction
# ─────────────────────────────────────────────────────────────────────────────
def tab2_train():
    df = get_df()
    results = train_models(df)
    log_run("activity_prediction", "RF+XGB+LGBM", "v1",
            metrics={k: v["roc_auc"] for k, v in results.items()},
            tags=["training"])
    rows = [{"Model": k, "ROC-AUC": v["roc_auc"], "Accuracy": v["accuracy"]}
            for k, v in results.items()]
    md = "### Training complete ✅\n"
    return md, pd.DataFrame(rows)


def tab2_predict(smiles: str, model_name: str):
    if not smiles.strip():
        return "Please enter a SMILES string."
    r = predict_single(smiles.strip(), model_name)
    if "error" in r:
        return f"❌ {r['error']}"
    return (
        f"**Molecule:** `{smiles}`\n\n"
        f"| Property | Value |\n|---|---|\n"
        f"| Model | {r['model']} |\n"
        f"| Predicted label | **{r['label']}** |\n"
        f"| Probability active | {r['prob_active']} |\n"
        f"| Confidence | {r['confidence_pct']}% |\n"
        f"| Uncertainty (ensemble) | ±{r['uncertainty']} |"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — ADMET
# ─────────────────────────────────────────────────────────────────────────────
def tab3_admet(smiles: str):
    if not smiles.strip():
        return "Enter a SMILES string.", None
    r = predict_admet(smiles.strip())
    if "error" in r:
        return f"❌ {r['error']}", None

    props = ["absorption", "distribution", "metabolism", "excretion", "toxicity", "solubility"]
    rows = [{
        "Property":  p.capitalize(),
        "Score":     r[p]["score"],
        "Result":    "✅ Pass" if r[p]["pass"] else "❌ Fail",
    } for p in props]

    flag = "✅ ADMET PASS" if r["overall_pass"] else "❌ ADMET FAIL"
    md = (
        f"### {flag} — Overall score: **{r['overall']}**\n\n"
        f"**Descriptors:** MW={r['descriptors']['MW']} · "
        f"LogP={r['descriptors']['LogP']} · "
        f"TPSA={r['descriptors']['TPSA']} · "
        f"HBA={r['descriptors']['HBA']} · HBD={r['descriptors']['HBD']}"
    )
    return md, pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 — Explainability
# ─────────────────────────────────────────────────────────────────────────────
def tab4_explain(smiles: str, model_name: str):
    if not smiles.strip():
        return "Enter a SMILES string.", "<p>No molecule</p>", None
    svg  = mol_to_svg(smiles.strip())
    tbl  = feature_importance_table(smiles.strip(), model_name)
    r    = explain_prediction(smiles.strip(), model_name)
    if "error" in r:
        return f"❌ {r['error']}", svg, None
    md = (
        f"### Feature attribution — {model_name}\n\n"
        f"Top contributing Morgan fingerprint bits shown below.\n"
        f"Total contributing bits: **{r['total_contributing_bits']}**"
    )
    return md, svg, tbl


# ─────────────────────────────────────────────────────────────────────────────
# Tab 5 — Molecule Generation
# ─────────────────────────────────────────────────────────────────────────────
def tab5_generate(seed_smiles_text: str, n_generate: int):
    seeds = [s.strip() for s in seed_smiles_text.strip().splitlines() if s.strip()]
    if not seeds:
        df = get_df()
        seeds = df["Smiles"].head(5).tolist()
    gen_df = generate_molecules(seeds, n_generate=int(n_generate))
    log_run("generation", "SMILES-mutation", "v1",
            metrics={"generated": len(gen_df), "ro5_pass": int(gen_df["ro5_pass"].sum())},
            tags=["generation"])
    md = f"### Generated {len(gen_df)} candidate molecules ✅\n\nRo5 passing: **{gen_df['ro5_pass'].sum()}**"
    return md, gen_df


# ─────────────────────────────────────────────────────────────────────────────
# Tab 6 — Multi-Objective Ranking
# ─────────────────────────────────────────────────────────────────────────────
def tab6_rank(smiles_text: str, w_act: float, w_sol: float, w_tox: float, w_sa: float, model_name: str):
    smiles_list = [s.strip() for s in smiles_text.strip().splitlines() if s.strip()]
    if not smiles_list:
        df = get_df()
        smiles_list = df["Smiles"].head(8).tolist()
    ranked = rank_molecules(smiles_list, w_act, w_sol, w_tox, w_sa, model_name)
    if ranked.empty:
        return "❌ No molecules could be scored. Train the model first (Tab 2).", None
    log_run("ranking", model_name, "v1",
            metrics={"n_ranked": len(ranked), "top_score": float(ranked["composite"].max())},
            tags=["ranking"])
    md = f"### Ranked {len(ranked)} molecules ✅\n\nTop composite score: **{ranked['composite'].max():.3f}**"
    return md, ranked[["rank", "smiles", "composite", "potency", "solubility", "toxicity", "admet_pass"]]


# ─────────────────────────────────────────────────────────────────────────────
# Tab 7 — Active Learning Loop
# ─────────────────────────────────────────────────────────────────────────────
def tab7_status():
    s = get_al_status()
    if s["iteration"] == 0:
        return "No iterations run yet. Click **Run Iteration** to start.", None
    hist = s["auc_history"]
    rows = [{"Iteration": i+1, "Val AUC": v} for i, v in enumerate(hist)]
    md = (
        f"### Active Learning Status\n\n"
        f"| | |\n|---|---|\n"
        f"| Current iteration | **{s['iteration']}** |\n"
        f"| Labeled molecules | **{s['labeled_count']}** |\n"
        f"| Best AUC | **{max(hist):.4f}** |"
    )
    return md, pd.DataFrame(rows)


def tab7_run(n_query: int):
    result = run_iteration(n_query=int(n_query))
    if "message" in result:
        return result["message"], None
    log_run("active_learning", "GBM", "v1",
            metrics={"val_auc": result["val_auc"], "iteration": result["iteration"]},
            tags=["active-learning"])
    md = (
        f"### Iteration {result['iteration']} complete ✅\n\n"
        f"| | |\n|---|---|\n"
        f"| Queries selected | **{result['queries_selected']}** |\n"
        f"| Labeled count | **{result['labeled_count']}** |\n"
        f"| Unlabeled remaining | **{result['unlabeled_remaining']}** |\n"
        f"| Validation AUC | **{result['val_auc']}** |"
    )
    hist = result["auc_history"]
    rows = [{"Iteration": i+1, "Val AUC": v} for i, v in enumerate(hist)]
    return md, pd.DataFrame(rows)


def tab7_reset():
    reset_loop()
    return "Loop reset ✅", None


# ─────────────────────────────────────────────────────────────────────────────
# Tab 8 — Experiment Tracking
# ─────────────────────────────────────────────────────────────────────────────
def tab8_runs():
    df = get_runs_df()
    if df.empty:
        return "No runs logged yet. Run any pipeline stage to log a run.", None
    md = f"### {len(df)} experiment runs logged"
    return md, df


def tab8_clear():
    clear_runs()
    return "All runs cleared ✅", None


# ─────────────────────────────────────────────────────────────────────────────
# Tab 9 — Data Quality Audit
# ─────────────────────────────────────────────────────────────────────────────
def tab9_audit():
    df = get_df()
    r  = run_audit(df)
    md = (
        f"### Data Quality Report\n\n"
        f"| Check | Result |\n|---|---|\n"
        f"| Total rows | **{r['total_rows']}** |\n"
        f"| Clean rows | **{r['clean_rows']}** |\n"
        f"| Duplicate SMILES | **{r['duplicate_count']}** |\n"
        f"| Assay noise cases | **{r['noise_cases']}** |\n"
        f"| Activity cliffs | **{r['activity_cliffs']}** |"
    )
    cliff_preview = r["cliffs_df"].head(10) if not r["cliffs_df"].empty else pd.DataFrame({"message": ["No cliffs found"]})
    return md, cliff_preview


# ─────────────────────────────────────────────────────────────────────────────
# Build Gradio UI
# ─────────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="EGFR Molecule Discovery", theme=gr.themes.Soft()) as demo:  # type: ignore
    gr.Markdown(
        "# 🧬 EGFR Molecule Discovery Platform\n"
        "End-to-end AI-assisted drug discovery pipeline · "
        "EGFR inhibitor discovery · 9-stage workflow"
    )

    # ── Tab 1: Data Collection ────────────────────────────────────────────
    with gr.Tab("01 · Data Collection"):
        gr.Markdown(
            "### ChEMBL EGFR Dataset\n"
            "Fetches live data from the ChEMBL REST API. "
            "Falls back to the bundled sample CSV if the API is unreachable."
        )
        with gr.Row():
            src1  = gr.Radio(["auto", "chembl", "csv"], value="auto", label="Source")
            tgt1  = gr.Textbox(value="CHEMBL203", label="Target ChEMBL ID")
            act1  = gr.Dropdown(["IC50", "Ki", "EC50"], value="IC50", label="Activity type")
        with gr.Row():
            conf1 = gr.Slider(1, 9, value=6, step=1, label="Min. confidence score")
            lim1  = gr.Slider(100, 2000, value=500, step=100, label="Max records to fetch")
        btn1 = gr.Button("Fetch & Clean Dataset", variant="primary")
        out1_md  = gr.Markdown()
        out1_tbl = gr.Dataframe(label="Sample rows (first 10)")
        btn1.click(tab1_load, inputs=[src1, tgt1, act1, conf1, lim1], outputs=[out1_md, out1_tbl])

    # ── Tab 2: Activity Prediction ────────────────────────────────────────
    with gr.Tab("02 · Activity Prediction"):
        gr.Markdown("### QSAR Models — RF / XGBoost / LightGBM\nTrain on Morgan fingerprints, then predict individual molecules.")
        with gr.Row():
            btn2_train = gr.Button("Train All Models", variant="primary")
        out2_train_md  = gr.Markdown()
        out2_train_tbl = gr.Dataframe(label="Model metrics")
        btn2_train.click(tab2_train, outputs=[out2_train_md, out2_train_tbl])

        gr.Markdown("---\n#### Predict a single molecule")
        with gr.Row():
            smi2   = gr.Textbox(label="SMILES", placeholder="COc1cc2ncnc(Nc3cccc(Br)c3)c2cc1OC", scale=3)
            mdl2   = gr.Dropdown(["XGBoost", "RandomForest", "LightGBM"], value="XGBoost", label="Model")
        btn2_pred = gr.Button("Predict")
        out2_pred = gr.Markdown()
        btn2_pred.click(tab2_predict, inputs=[smi2, mdl2], outputs=out2_pred)

    # ── Tab 3: ADMET ──────────────────────────────────────────────────────
    with gr.Tab("03 · ADMET Prediction"):
        gr.Markdown("### ADMET Property Screening\nAbsorption · Distribution · Metabolism · Excretion · Toxicity · Solubility")
        smi3     = gr.Textbox(label="SMILES", placeholder="COc1cc2ncnc(Nc3cccc(Br)c3)c2cc1OC")
        btn3     = gr.Button("Predict ADMET", variant="primary")
        out3_md  = gr.Markdown()
        out3_tbl = gr.Dataframe(label="ADMET breakdown")
        btn3.click(tab3_admet, inputs=smi3, outputs=[out3_md, out3_tbl])

    # ── Tab 4: Explainability ─────────────────────────────────────────────
    with gr.Tab("04 · Explainability"):
        gr.Markdown("### Prediction Explainability\nTop contributing Morgan fingerprint bits + 2D structure view.")
        with gr.Row():
            smi4 = gr.Textbox(label="SMILES", placeholder="COc1cc2ncnc(Nc3cccc(Br)c3)c2cc1OC", scale=3)
            mdl4 = gr.Dropdown(["XGBoost", "RandomForest", "LightGBM"], value="XGBoost", label="Model")
        btn4     = gr.Button("Explain", variant="primary")
        out4_md  = gr.Markdown()
        with gr.Row():
            out4_svg = gr.HTML(label="2D Structure")
            out4_tbl = gr.Dataframe(label="Feature contributions")
        btn4.click(tab4_explain, inputs=[smi4, mdl4], outputs=[out4_md, out4_svg, out4_tbl])

    # ── Tab 5: Molecule Generation ────────────────────────────────────────
    with gr.Tab("05 · Molecule Generation"):
        gr.Markdown("### Generate Novel Candidates\nRDKit-based structural mutation of seed EGFR inhibitors.")
        smi5 = gr.Textbox(
            label="Seed SMILES (one per line — leave blank to use dataset seeds)",
            lines=4,
            placeholder="COc1cc2ncnc(Nc3cccc(Br)c3)c2cc1OC\nCCOc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCC"
        )
        n5   = gr.Slider(5, 50, value=20, step=5, label="Molecules to generate")
        btn5     = gr.Button("Generate", variant="primary")
        out5_md  = gr.Markdown()
        out5_tbl = gr.Dataframe(label="Generated candidates")
        btn5.click(tab5_generate, inputs=[smi5, n5], outputs=[out5_md, out5_tbl])

    # ── Tab 6: Ranking ────────────────────────────────────────────────────
    with gr.Tab("06 · Multi-Objective Ranking"):
        gr.Markdown(
            "### Multi-Objective Ranking\n"
            "Score = w_activity × potency + w_solubility × sol + w_toxicity × tox + w_sa × SA"
        )
        smi6 = gr.Textbox(
            label="SMILES to rank (one per line — leave blank to use dataset)",
            lines=5,
        )
        with gr.Row():
            w_act = gr.Slider(0.0, 1.0, value=0.4, step=0.05, label="w_activity")
            w_sol = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="w_solubility")
            w_tox = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="w_toxicity")
            w_sa  = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="w_SA")
        mdl6  = gr.Dropdown(["XGBoost", "RandomForest", "LightGBM"], value="XGBoost", label="Model")
        btn6      = gr.Button("Rank Molecules", variant="primary")
        out6_md   = gr.Markdown()
        out6_tbl  = gr.Dataframe(label="Ranked shortlist")
        btn6.click(tab6_rank, inputs=[smi6, w_act, w_sol, w_tox, w_sa, mdl6], outputs=[out6_md, out6_tbl])

    # ── Tab 7: Active Learning ────────────────────────────────────────────
    with gr.Tab("07 · Active Learning Loop"):
        gr.Markdown(
            "### Active Learning Discovery Loop\n"
            "Train → Predict uncertainty on pool → Select informative molecules → Simulate labeling → Retrain"
        )
        n7       = gr.Slider(5, 30, value=10, step=5, label="Queries per iteration")
        with gr.Row():
            btn7_run    = gr.Button("Run Next Iteration", variant="primary")
            btn7_status = gr.Button("Show Status")
            btn7_reset  = gr.Button("Reset Loop", variant="stop")
        out7_md  = gr.Markdown()
        out7_tbl = gr.Dataframe(label="AUC history")
        btn7_run.click(tab7_run,    inputs=n7, outputs=[out7_md, out7_tbl])
        btn7_status.click(tab7_status,         outputs=[out7_md, out7_tbl])
        btn7_reset.click(tab7_reset,           outputs=[out7_md, out7_tbl])

    # ── Tab 8: Experiment Tracking ────────────────────────────────────────
    with gr.Tab("08 · Experiment Tracking"):
        gr.Markdown("### Experiment Run Log\nAll pipeline runs are logged automatically.")
        with gr.Row():
            btn8_view  = gr.Button("View All Runs", variant="primary")
            btn8_clear = gr.Button("Clear Runs", variant="stop")
        out8_md  = gr.Markdown()
        out8_tbl = gr.Dataframe(label="Run log")
        btn8_view.click(tab8_runs,  outputs=[out8_md, out8_tbl])
        btn8_clear.click(tab8_clear, outputs=[out8_md, out8_tbl])

    # ── Tab 9: Data Quality ───────────────────────────────────────────────
    with gr.Tab("09 · Data Quality Audit"):
        gr.Markdown("### Data Quality Audit\nDuplicate detection · Assay noise · Activity cliffs")
        btn9     = gr.Button("Run Audit", variant="primary")
        out9_md  = gr.Markdown()
        out9_tbl = gr.Dataframe(label="Activity cliffs (top 10)")
        btn9.click(tab9_audit, outputs=[out9_md, out9_tbl])

demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
