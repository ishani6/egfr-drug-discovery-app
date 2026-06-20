import gradio as gr
import numpy as np
import pandas as pd
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
GEN_PATH = DATA_DIR / "module8_transvae_generated_molecules.csv"
OPT_PATH = DATA_DIR / "module9_optimized_output.csv"


def normalize_columns(df):
    if df is None:
        return None
    out = df.copy()
    out.columns = [c.strip().lower() for c in out.columns]
    return out


def load_raw_data():
    df = fetch_egfr_bioactivity_from_chembl()
    df = normalize_columns(df)
    msg = f"Loaded {len(df)} raw EGFR records from ChEMBL." if df is not None else "No raw data loaded."
    return df, msg


def curate_data(raw_df):
    if raw_df is None or len(raw_df) == 0:
        return None, "No raw data loaded."
    curated = curate_egfr_dataset(raw_df)
    curated = normalize_columns(curated)
    return curated, f"Curated dataset contains {len(curated)} unique molecules."


def make_features(curated_df):
    if curated_df is None or len(curated_df) == 0:
        return None, "No curated data available."
    feat = build_feature_table(curated_df)
    feat = normalize_columns(feat)
    return feat, f"Feature table built with {len(feat)} molecules."


def _activity_target_col(df):
    for c in ["pic50", "p_ic50", "pic_50"]:
        if c in df.columns:
            return c
    return None


def predict_activity(feature_df):
    if feature_df is None or len(feature_df) == 0:
        return None, "No feature table available."
    df = normalize_columns(feature_df)
    model = load_prediction_model(MODEL_DIR / "egfr_activity_model.joblib")
    if model is not None:
        feature_cols = get_model_feature_columns(df, model)
        df["pred_pic50"] = predict_with_model(model, df, feature_cols)
        msg = "Predicted EGFR activity using saved model."
    else:
        target_col = _activity_target_col(df)
        if target_col is not None:
            df["pred_pic50"] = pd.to_numeric(df[target_col], errors="coerce").fillna(0.0)
            msg = "Saved activity model not found; copied measured pIC50 as fallback."
        else:
            df["pred_pic50"] = 0.0
            msg = "Saved activity model not found; no pIC50 column available."
    df["activity_prob"] = 1 / (1 + 10 ** (df["pred_pic50"] - 6))
    return df, msg


def predict_admet(activity_df):
    if activity_df is None or len(activity_df) == 0:
        return None, "No activity predictions available."
    out = normalize_columns(activity_df)
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
        out["admet_score"] = 0.5
        out["solubility_class"] = "Medium"
        out["toxicity_class"] = "Low"
        out["clearance_class"] = "Medium"
        out["half_life_class"] = "Moderate"
        msg = "Saved ADMET model not found; used fallback labels."
    return out, msg


def rank_molecules(admet_df):
    if admet_df is None or len(admet_df) == 0:
        return None, "No ADMET results available."
    out = normalize_columns(admet_df)
    if "pred_pic50" not in out.columns:
        out["pred_pic50"] = 0.0
    if "activity_prob" not in out.columns:
        out["activity_prob"] = 0.5
    out["final_score"] = (
        out["pred_pic50"].rank(pct=True) * 0.5 +
        out["activity_prob"].rank(pct=True) * 0.2 +
        (out.get("toxicity_class", pd.Series(["Low"] * len(out))) == "Low").astype(int) * 0.2 +
        (out.get("solubility_class", pd.Series(["Good"] * len(out))) != "Poor").astype(int) * 0.1
    )
    out = out.sort_values("final_score", ascending=False)
    return out, f"Ranked {len(out)} molecules."


def load_generated():
    gen = load_csv_if_exists(GEN_PATH)
    opt = load_csv_if_exists(OPT_PATH)
    gen = normalize_columns(gen) if gen is not None else None
    opt = normalize_columns(opt) if opt is not None else None
    if gen is None and opt is None:
        return None, None, "No generated/optimized CSV files found."
    return gen, opt, "Loaded generated and optimized molecule tables."


with gr.Blocks(title="EGFR Drug Discovery Pipeline") as demo:
    gr.Markdown("# EGFR Drug Discovery Pipeline")
    gr.Markdown("Sequential EGFR drug discovery workflow.")

    raw_state = gr.State()
    curated_state = gr.State()
    feature_state = gr.State()
    activity_state = gr.State()
    admet_state = gr.State()
    ranked_state = gr.State()
    gen_state = gr.State()
    opt_state = gr.State()

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
    btn_gen.click(load_generated, [], [gen_state, opt_state, status]).then(lambda x: x[0], gen_state, generated_table).then(lambda x: x[0] if isinstance(x, tuple) else x, opt_state, optimized_table)

demo.launch(server_name="0.0.0.0", server_port=7860)
