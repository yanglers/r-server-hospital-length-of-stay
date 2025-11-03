import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from math import sqrt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
# -----------------------------
# Config (match your training script)
# -----------------------------
RANDOM_STATE = 42
TRAIN_PCT = 0.70
DATA_FILE = Path("/Users/austinyang/r-server-hospital-length-of-stay/Data/LengthOfStay.csv")
MODEL_DIR = Path(__file__).resolve().parent / "models_sklearn_los"

TARGET = "lengthofstay"
ID_COL = "eid"
DATE_COLS = ["vdate", "discharged"]
DROP_FROM_FEATURES = ["eid", "vdate", "discharged", "facid"]

CONTINUOUS_TO_STANDARDIZE = [
    "hematocrit", "neutrophils", "sodium", "glucose", "bloodureanitro",
    "creatinine", "bmi", "pulse", "respiration"
]

ISSUE_INDICATORS = [
    "hemo", "dialysisrenalendstage", "asthma", "irondef", "pneum",
    "substancedependence", "psychologicaldisordermajor", "depress",
    "psychother", "fibrosisandother", "malnutrition"
]

# -----------------------------
# Data prep logic (mirrors training script)
# -----------------------------
def load_raw():
    df = pd.read_csv(DATA_FILE)
    for c in DATE_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    # Indicator columns numeric
    for c in [c for c in ISSUE_INDICATORS if c in df.columns]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def clean_fill(df: pd.DataFrame) -> pd.DataFrame:
    protected = {ID_COL, TARGET, *DATE_COLS}
    cols_to_consider = [c for c in df.columns if c not in protected]

    num_cols = df[cols_to_consider].select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df[cols_to_consider].select_dtypes(include=["object", "category"]).columns.tolist()

    if num_cols:
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean(numeric_only=True))

    for c in cat_cols:
        if df[c].isna().any():
            mode_val = df[c].mode(dropna=True)
            df[c] = df[c].fillna(mode_val.iloc[0] if not mode_val.empty else "UNKNOWN")

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    return df

def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    present_cont = [c for c in CONTINUOUS_TO_STANDARDIZE if c in df.columns]
    for c in present_cont:
        mean, std = df[c].mean(), df[c].std(ddof=0)
        df[c] = (df[c] - mean) / (std if std else 1.0)

    present_issue = [c for c in ISSUE_INDICATORS if c in df.columns]
    if present_issue:
        df["number_of_issues"] = (
            df[present_issue]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0).sum(axis=1).astype(int).astype(str)
        )
    else:
        df["number_of_issues"] = "0"
    return df

def prepare(df: pd.DataFrame):
    df = clean_fill(df.copy())
    df = feature_engineer(df)
    feature_cols = [c for c in df.columns if c not in set(DROP_FROM_FEATURES + [TARGET])]
    X = df[feature_cols].copy()
    y = df[TARGET].copy()
    return df, X, y, feature_cols

# -----------------------------
# Load models
# -----------------------------
@st.cache_resource
def load_models():
    models = {}
    if MODEL_DIR.exists():
        for p in MODEL_DIR.glob("*.joblib"):
            models[p.stem.replace("_model","")] = joblib.load(p)
    return models

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Hospital Length of Stay Predictor", layout="wide")
st.title("🏥 Hospital Length of Stay — Demo UI")

models = load_models()
if not models:
    st.error("No models found in ./models_sklearn_los. Train and save them first.")
    st.stop()

df_raw = load_raw()
df_prep, X_all, y_all, feature_cols = prepare(df_raw)

# Train/test split to show metrics similar to your notebook
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_all, y_all, df_prep.index, train_size=TRAIN_PCT, random_state=RANDOM_STATE
)

model_key = st.sidebar.selectbox("Model", sorted(models.keys()), index=0)
pipe = models[model_key]

st.sidebar.markdown("### Actions")
mode = st.sidebar.radio("Mode", ["Pick a patient", "Upload CSV", "Diagnostics"], index=0)

def score_and_render(base_df: pd.DataFrame, X: pd.DataFrame):
    pred = pipe.predict(X)
    pred_round = np.rint(pred).astype(int)
    out = base_df.copy()
    if "vdate" in out.columns:
        out["discharged_Pred"] = out["vdate"] + pd.to_timedelta(pred_round, unit="D")
    else:
        out["discharged_Pred"] = pd.NaT
    out["lengthofstay_Pred"] = pred
    out["lengthofstay_Pred_Rounded"] = pred_round
    return out

if mode == "Pick a patient":
    # Pick from test set only (to avoid leakage in metrics)
    pick_idx = st.selectbox("Select eid from test set:", df_prep.loc[idx_test, ID_COL].tolist())
    row_idx = df_prep.index[df_prep[ID_COL] == pick_idx][0]
    row_X = X_all.loc[[row_idx]]
    row_df = df_prep.loc[[row_idx], [ID_COL, "vdate", TARGET]]

    scored = score_and_render(row_df, row_X)
    st.subheader("Prediction")
    st.write(scored[[ID_COL, "vdate", TARGET, "lengthofstay_Pred", "lengthofstay_Pred_Rounded", "discharged_Pred"]])

elif mode == "Upload CSV":
    st.write("Upload a CSV with the **same schema** as the training file (LengthOfStay.csv).")
    up = st.file_uploader("CSV", type=["csv"])
    if up:
        df_u = pd.read_csv(up)
        # Apply same prep
        for c in DATE_COLS:
            if c in df_u.columns:
                df_u[c] = pd.to_datetime(df_u[c], errors="coerce")
        df_u_prep, X_u, _, _ = prepare(df_u)
        base_cols = [c for c in ["eid", "vdate", TARGET] if c in df_u_prep.columns]
        scored = score_and_render(df_u_prep[base_cols], X_u)
        st.success(f"Scored {len(scored)} rows.")
        st.dataframe(scored.head(20))
        st.download_button(
            "Download predictions CSV",
            scored.to_csv(index=False).encode("utf-8"),
            file_name="LoS_Predictions.csv",
            mime="text/csv"
        )

else:  # Diagnostics
    st.subheader("Test-set metrics")
    y_pred = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    st.write(
        pd.DataFrame([{"Model": model_key, "MAE": mae, "RMSE": rmse, "R2": r2}])
    )

    st.subheader("Residuals")
    resid = y_test - y_pred
    st.line_chart(pd.DataFrame({"residuals": resid}).reset_index(drop=True))
    st.subheader("Prediction vs. Actual (sample)")
    show = pd.DataFrame({"actual": y_test.values, "pred": y_pred}).head(500)
    st.scatter_chart(show)
