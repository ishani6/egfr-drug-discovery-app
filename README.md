# EGFR Drug Discovery Pipeline Streamlit App

This Streamlit app is a deployment-friendly dashboard for the final project:
End-to-End AI-Assisted Drug Discovery Pipeline for EGFR Inhibitors.

## Files
- `app.py` - Streamlit dashboard
- `requirements.txt` - minimal deployment dependencies
- `data/` - optional folder for precomputed CSV outputs from Colab

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
Push these files to GitHub and deploy on Streamlit Community Cloud.
Upload your Colab CSV outputs through the UI or place them in a `data/` folder.