import gradio as gr
import numpy as np
from pathlib import Path
from utils import (
    fetch_egfr_bioactivity_from_chembl,
    curate_egfr_dataset,
    build_feature_table,
    load_csv_if_exists,
    load_prediction_model,
    get_model_feature_columns,
    predict_with_model,
)

DATA_DIR = Path("data")
MODEL_DIR = Path("models")

def load_raw_data():
    df = fetch_egfr_bioactivity_from_chembl()
    return df, f"Loaded {len(df)} raw EGFR records from ChEMBL."

def curate_data(raw_df):
    if raw_df is None or len(raw_df) == 0:
        return None, "No raw data loaded."
    curated = curate_egfr_dataset(raw_df)
    return curated, f"Curated dataset contains {len(curated)} unique molecules."

def make_features(curated_df):
    if curated_df is None or len(curated_df) == 0:
        return None, "No curated data available."
    feat = build_feature_table(curated_df)
    return feat, f"Feature table built with {len(feat)} molecules."

def predict_activity(feature_df):
    if feature_df is None or len(feature_df) == 0:
        return None, "No feature table available."
    df = feature_df.copy()
    model = load_prediction_model(MODEL_DIR / "egfr_activity_model.joblib")
    if model is not None:
        feature_cols = get_model_feature_columns(df, model)
        df["pred_pIC50"] = predict_with_model(model, df, feature_cols)
        msg = "Predicted EGFR activity using saved model."
    else:
        df["pred_pIC50"] = df["pIC50"] if "pIC50" in df.columns else 0.0
        msg = "Saved activity model not found; copied pIC50 as fallback."
    df["activity_prob"] = 1 / (1 + 10 ** (df["pred_pIC50"] - 6))
    return df, msg

def predict_admet(activity_df):
    if activity_df is None or len(activity_df) == 0:
        return None, "No activity predictions available."
    out = activity_df.copy()
    model = load_prediction_model(MODEL_DIR / "egfr_admet_model.joblib")
    if model is not None:
        feature_cols = get_model_feature_columns(out, model)
        out["admet_score"] = predict_with_model(model, out, feature_cols)
        out["solubility_class"] = np.where(out["admet_score"] > 0.5, "Good", "Poor")
        out["toxicity_class"] = np.where(out["admet_score"] > 0.5, "Low", "High")
        out["clearance_class"] = np.where(out["admet_score"] > 0.5, "Medium", "Low")
        out["half_life_class"] = np.where(out["admet_score"] > 0.5, "Moderate", "Short")
        msg = "Predicted ADMET using saved model."
    else:
        out["solubility_class"] = "Medium"
        out["toxicity_class"] = "Low"
        out["clearance_class"] = "Medium"
        out["half_life_class"] = "Moderate"
        msg = "Saved ADMET model not found; used fallback labels."
    return out, msg

def rank_molecules(admet_df):
    if admet_df is None or len(admet_df) == 0:
        return None, "No ADMET results available."
    out = admet_df.copy()
    out["final_score"] = (
        out["pred_pIC50"].rank(pct=True) * 0.5 +
        out["activity_prob"].rank(pct=True) * 0.2 +
        (out["toxicity_class"] == "Low").astype(int) * 0.2 +
        (out["solubility_class"] != "Poor").astype(int) * 0.1
    )
    out = out.sort_values("final_score", ascending=False)
    return out, f"Ranked {len(out)} molecules."

def load_generated():
    gen = load_csv_if_exists(DATA_DIR / "module8_transvae_generated_molecules.csv")
    opt = load_csv_if_exists(DATA_DIR / "module9_optimized_output.csv")
    if gen is None and opt is None:
        return None, "No generated/optimized CSV files found in data/."
    return {"generated": gen, "optimized": opt}, "Loaded generated and optimized molecule tables."

with gr.Blocks(title="EGFR Drug Discovery Pipeline") as demo:
    gr.Markdown("# EGFR Drug Discovery Pipeline")
    gr.Markdown("Hugging Face Docker Space for sequential EGFR drug discovery workflow.")

    raw_state = gr.State()
    curated_state = gr.State()
    feature_state = gr.State()
    activity_state = gr.State()
    admet_state = gr.State()
    ranked_state = gr.State()
    gen_state = gr.State()

    status = gr.Markdown("Click the first button to begin.")

    with gr.Row():
        btn_load = gr.Button("Get molecules from ChEMBL", variant="primary")
        btn_curate = gr.Button("Curate dataset")
        btn_feat = gr.Button("Build descriptors and fingerprints")
        btn_act = gr.Button("Predict activity against EGFR")
        btn_admet = gr.Button("Predict ADMET")
        btn_rank = gr.Button("Rank molecules")
        btn_gen = gr.Button("Load generated / optimized molecules")

    with gr.Tabs():
        with gr.TabItem("Raw ChEMBL"):
            raw_table = gr.Dataframe(label="Raw EGFR bioactivity", interactive=False)
        with gr.TabItem("Curated"):
            curated_table = gr.Dataframe(label="Curated EGFR dataset", interactive=False)
        with gr.TabItem("Features"):
            feature_table = gr.Dataframe(label="Feature table", interactive=False)
        with gr.TabItem("Activity"):
            activity_table = gr.Dataframe(label="Activity predictions", interactive=False)
        with gr.TabItem("ADMET"):
            admet_table = gr.Dataframe(label="ADMET predictions", interactive=False)
        with gr.TabItem("Ranking"):
            ranked_table = gr.Dataframe(label="Ranked molecules", interactive=False)
        with gr.TabItem("Generated / Optimized"):
            generated_table = gr.Dataframe(label="Generated molecules", interactive=False)
            optimized_table = gr.Dataframe(label="Optimized molecules", interactive=False)

    btn_load.click(load_raw_data, [], [raw_state, status]).then(lambda df: df, raw_state, raw_table)
    btn_curate.click(curate_data, [raw_state], [curated_state, status]).then(lambda df: df, curated_state, curated_table)
    btn_feat.click(make_features, [curated_state], [feature_state, status]).then(lambda df: df, feature_state, feature_table)
    btn_act.click(predict_activity, [feature_state], [activity_state, status]).then(lambda df: df, activity_state, activity_table)
    btn_admet.click(predict_admet, [activity_state], [admet_state, status]).then(lambda df: df, admet_state, admet_table)
    btn_rank.click(rank_molecules, [admet_state], [ranked_state, status]).then(lambda df: df, ranked_state, ranked_table)
    btn_gen.click(load_generated, [], [gen_state, status]).then(lambda d: d["generated"] if d else None, gen_state, generated_table).then(lambda d: d["optimized"] if d else None, gen_state, optimized_table)

demo.launch(server_name="0.0.0.0", server_port=7860)