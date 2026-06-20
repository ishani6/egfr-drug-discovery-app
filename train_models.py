from pathlib import Path
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score, roc_auc_score

DATA_PATH = Path("data/egfr_features.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

feature_cols = ["MolWt", "LogP", "TPSA", "HBD", "HBA"] + [c for c in df.columns if c.startswith("fp_")]
X = df[feature_cols].fillna(0)

y_reg = df["pIC50"].astype(float)
y_clf = (df["pIC50"] >= 6.0).astype(int)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_clf, test_size=0.2, random_state=42)

activity_model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
activity_model.fit(X_train_r, y_train_r)
pred_r = activity_model.predict(X_test_r)
print("Activity MAE:", mean_absolute_error(y_test_r, pred_r))
joblib.dump(activity_model, MODEL_DIR / "egfr_activity_model.joblib")

admet_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
admet_model.fit(X_train_c, y_train_c)
pred_c = admet_model.predict(X_test_c)
proba_c = admet_model.predict_proba(X_test_c)[:, 1]
print("ADMET Accuracy:", accuracy_score(y_test_c, pred_c))
print("ADMET ROC-AUC:", roc_auc_score(y_test_c, proba_c))
joblib.dump(admet_model, MODEL_DIR / "egfr_admet_model.joblib")

print("Saved models to models/")