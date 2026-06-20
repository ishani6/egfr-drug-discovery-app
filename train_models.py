from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score, roc_auc_score

DATA_PATH = Path("data/egfr_features.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

if (MODEL_DIR / "egfr_activity_model.joblib").exists() and (MODEL_DIR / "egfr_admet_model.joblib").exists():
    print("models already exist")
    raise SystemExit(0)

df = pd.read_csv(DATA_PATH)

if "pIC50" not in df.columns:
    raise ValueError("Training data must contain a pIC50 column")

base_cols = [c for c in ["MolWt", "LogP", "TPSA", "HBA", "HBD"] if c in df.columns]
fp_cols = [c for c in df.columns if c.lower().startswith(("morgan", "maccs", "fp_"))]
feature_cols = base_cols + fp_cols

if not feature_cols:
    raise ValueError("No feature columns found")

X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
y_reg = pd.to_numeric(df["pIC50"], errors="coerce")
mask = y_reg.notna()
X = X.loc[mask].reset_index(drop=True)
y_reg = y_reg.loc[mask].reset_index(drop=True)
y_clf = (y_reg >= 6.0).astype(int)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_clf, test_size=0.2, random_state=42)

activity_model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
activity_model.fit(X_train_r, y_train_r)
print("Activity MAE:", mean_absolute_error(y_test_r, activity_model.predict(X_test_r)))
joblib.dump(activity_model, MODEL_DIR / "egfr_activity_model.joblib")

admet_model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
admet_model.fit(X_train_c, y_train_c)
pred_c = admet_model.predict(X_test_c)
proba_c = admet_model.predict_proba(X_test_c)[:, 1]
print("ADMET Accuracy:", accuracy_score(y_test_c, pred_c))
print("ADMET ROC-AUC:", roc_auc_score(y_test_c, proba_c))
joblib.dump(admet_model, MODEL_DIR / "egfr_admet_model.joblib")

print("Saved models to models/")