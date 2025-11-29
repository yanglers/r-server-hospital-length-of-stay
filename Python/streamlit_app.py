# streamlit_app.py
import os
from pathlib import Path
from math import sqrt

import numpy as np
import pandas as pd
import streamlit as st
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor

# -----------------------------
# Config (match training where possible)
# -----------------------------
RANDOM_STATE = 100
TRAIN_PCT = 0.70
TARGET = "lengthofstay"
ID_COL = "eid"
DATE_COLS = ["vdate", "discharged"]
DROP_FROM_FEATURES = ["eid", "vdate", "discharged", "facid"]

CONTINUOUS_TO_STANDARDIZE = [
    "hematocrit",
    "neutrophils",
    "sodium",
    "glucose",
    "bloodureanitro",
    "creatinine",
    "bmi",
    "pulse",
    "respiration",
]

ISSUE_INDICATORS = [
    "hemo",
    "dialysisrenalendstage",
    "asthma",
    "irondef",
    "pneum",
    "substancedependence",
    "psychologicaldisordermajor",
    "depress",
    "psychother",
    "fibrosisandother",
    "malnutrition",
]

# Candidates for procedure column
PROCEDURE_COL_CANDIDATES = [
    "procedure",
    "primaryprocedure",
    "surgery",
    "aprdrgdescription",
]

APP_DIR = Path(__file__).resolve().parent
REPO_DIR = APP_DIR.parent
MODEL_DIR = APP_DIR / "models_sklearn_los"

# -----------------------------
# Robust data path resolution
# -----------------------------
CANDIDATES = []
env_path = os.getenv("LOS_DATA")
if env_path:
    CANDIDATES.append(Path(env_path))

CANDIDATES += [
    REPO_DIR / "Data" / "LengthOfStay.csv",
    APP_DIR / "Data" / "LengthOfStay.csv",
    Path("Data/LengthOfStay.csv"),
    Path("LengthOfStay.csv"),
]

DATA_FILE = next((p for p in CANDIDATES if p.exists()), None)


def load_raw() -> pd.DataFrame:
    """Load CSV from repo or let the user upload it."""
    if DATA_FILE is not None and DATA_FILE.exists():
        st.info(f"Using data file: {DATA_FILE}")
        return pd.read_csv(DATA_FILE)

    st.warning("Couldn't find **Data/LengthOfStay.csv** in the repo. Please upload it.")
    up = st.file_uploader("Upload LengthOfStay.csv", type=["csv"])
    if not up:
        st.stop()
    return pd.read_csv(up)


# -----------------------------
# Cleaning & feature engineering
# -----------------------------
def clean_fill(df: pd.DataFrame) -> pd.DataFrame:
    # parse dates
    for c in DATE_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

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

    # ensure indicators numeric
    for c in [c for c in ISSUE_INDICATORS if c in df.columns]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    return df


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    # z-score selected continuous columns (dataset stats)
    present_cont = [c for c in CONTINUOUS_TO_STANDARDIZE if c in df.columns]
    for c in present_cont:
        mean, std = df[c].mean(), df[c].std(ddof=0)
        df[c] = (df[c] - mean) / (std if std else 1.0)

    # number_of_issues = sum of indicators, cast to string
    present_issue = [c for c in ISSUE_INDICATORS if c in df.columns]
    if present_issue:
        df["number_of_issues"] = (
            df[present_issue]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .sum(axis=1)
            .astype(int)
            .astype(str)
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
# Model loading & first-run fallback training
# -----------------------------
@st.cache_resource
def load_models():
    models = {}
    if MODEL_DIR.exists():
        for p in MODEL_DIR.glob("*.joblib"):
            models[p.stem.replace("_model", "")] = joblib.load(p)
    return models


def quick_train_and_save(df_raw: pd.DataFrame) -> float:
    """Train a simple GBT model with the same prep, save to MODEL_DIR, return test RMSE."""
    df_prep, X_all, y_all, _ = prepare(df_raw)

    Xtr, Xte, ytr, yte = train_test_split(
        X_all, y_all, train_size=TRAIN_PCT, random_state=RANDOM_STATE
    )

    cat = Xtr.select_dtypes(include=["object", "category"]).columns
    num = Xtr.select_dtypes(exclude=["object", "category"]).columns
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    prep = ColumnTransformer(
        [("cat", ohe, cat), ("num", "passthrough", num)], remainder="drop"
    )
    pipe = Pipeline(
        [
            ("prep", prep),
            (
                "model",
                GradientBoostingRegressor(
                    n_estimators=40, learning_rate=0.3, random_state=9
                ),
            ),
        ]
    )

    pipe.fit(Xtr, ytr)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_DIR / "GBT_model.joblib")

    rmse = float(np.sqrt(mean_squared_error(yte, pipe.predict(Xte))))
    return rmse


# -----------------------------
# UI helpers
# -----------------------------
def score_and_render(base_df: pd.DataFrame, X: pd.DataFrame, pipe) -> pd.DataFrame:
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


def explain_prediction(pipe, X_row: pd.DataFrame):
    """Show either local contributions (for linear) or global importances (trees)."""
    st.subheader("Feature contributions / importance")

    model = pipe.named_steps.get("model")
    prep = pipe.named_steps.get("prep")
    if model is None or prep is None:
        st.info("Cannot introspect this model pipeline.")
        return

    Xt = prep.transform(X_row)
    try:
        feat_names = prep.get_feature_names_out()
    except Exception:
        feat_names = np.array([f"f{i}" for i in range(Xt.shape[1])])

    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_).ravel()
        intercept = float(model.intercept_)
        contrib = Xt.reshape(-1) * coef
        df_contrib = pd.DataFrame(
            {
                "feature_encoded": feat_names,
                "contribution": contrib,
            }
        )
        df_contrib["abs_contribution"] = df_contrib["contribution"].abs()
        df_contrib = df_contrib.sort_values(
            "abs_contribution", ascending=False
        ).head(15)
        st.write(
            "Local contributions from a linear-style model (prediction ≈ intercept + Σ feature · coef)."
        )
        st.dataframe(df_contrib[["feature_encoded", "contribution"]])
    elif hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
        df_imp = pd.DataFrame(
            {
                "feature_encoded": feat_names,
                "importance": importances,
            }
        ).sort_values("importance", ascending=False)
        st.write("Global feature importances from this tree-based model.")
        st.dataframe(df_imp.head(20))
    else:
        st.info(
            "This model type does not expose per-feature contributions or importances."
        )


def show_procedure_context(
    df_prep: pd.DataFrame,
    proc_col: str,
    proc_stats: pd.DataFrame,
    row_idx,
    scored_row: pd.DataFrame,
):
    if proc_col is None or proc_col not in df_prep.columns:
        st.info("No procedure column found in data; cannot show procedure-specific stats.")
        return

    proc_val = df_prep.loc[row_idx, proc_col]
    st.subheader("Procedure context")
    st.write(f"Procedure: **{proc_val}**")

    if proc_val not in proc_stats.index:
        st.info("No historical stats available for this procedure.")
        return

    stats = proc_stats.loc[proc_val]
    mean_los = stats["mean"]
    std_los = stats["std"]
    count_los = int(stats["count"])

    st.write(
        f"Average historical length of stay for this procedure: {mean_los:.2f} days "
        f"(n={count_los})."
    )

    if "lengthofstay_Pred_Rounded" in scored_row.columns:
        pred_los = float(scored_row["lengthofstay_Pred_Rounded"].iloc[0])
    else:
        pred_los = float(scored_row["lengthofstay_Pred"].iloc[0])

    if pd.notnull(std_los) and std_los > 0:
        z = (pred_los - mean_los) / std_los
        st.write(f"Prediction is {z:+.2f} standard deviations from the procedure mean.")
        if abs(z) > 2:
            st.warning(
                "This predicted stay looks unusual for this procedure (|z| > 2)."
            )

    # Distribution plot
    data_proc = df_prep[df_prep[proc_col] == proc_val][TARGET].dropna()
    if len(data_proc) >= 5:
        bins = min(20, max(5, int(len(data_proc) ** 0.5)))
        counts, edges = np.histogram(data_proc, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        hist_df = pd.DataFrame({"LoS_bin": centers, "Count": counts}).set_index(
            "LoS_bin"
        )
        st.write("Distribution of length of stay for this procedure:")
        st.bar_chart(hist_df)
    else:
        st.info("Not enough historical data to show a distribution for this procedure.")


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Hospital Length of Stay Predictor", layout="wide")
st.title("Hospital Length of Stay")

models = load_models()
if not models:
    st.info("No saved models found in ./models_sklearn_los.")
    df_raw = load_raw()
    if st.button("Train a quick GBT model now"):
        rmse = quick_train_and_save(df_raw)
        st.success(
            f"Model trained and saved. Test RMSE ≈ {rmse:.3f}. Click **Rerun** to load it."
        )
        st.stop()
    else:
        st.stop()

# Data for inference/diagnostics
df_raw = load_raw()
df_prep, X_all, y_all, feature_cols = prepare(df_raw)

# Procedure stats for UI
PROC_COL = next((c for c in PROCEDURE_COL_CANDIDATES if c in df_prep.columns), None)
if PROC_COL:
    proc_stats = df_prep.groupby(PROC_COL)[TARGET].agg(["mean", "std", "count"])
else:
    proc_stats = None

# Train/test split to show metrics
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_all, y_all, df_prep.index, train_size=TRAIN_PCT, random_state=RANDOM_STATE
)

model_key = st.sidebar.selectbox("Model", sorted(models.keys()), index=0)
pipe = models[model_key]

st.sidebar.markdown("### Actions")
mode = st.sidebar.radio(
    "Mode", ["Pick a patient", "Upload CSV", "Diagnostics", "Manual input"], index=0
)

# -----------------------------
# Mode: Pick a patient
# -----------------------------
if mode == "Pick a patient":
    if ID_COL in df_prep.columns:
        pick_from = df_prep.loc[idx_test, ID_COL].tolist()
        pick_idx = st.selectbox("Select eid from test set:", pick_from)
        row_idx = df_prep.index[df_prep[ID_COL] == pick_idx][0]
    else:
        st.warning(f"`{ID_COL}` not found; selecting by row index instead.")
        pick_from = idx_test.tolist()
        row_idx = st.selectbox("Select row from test set:", pick_from)

    row_X = X_all.loc[[row_idx]]
    base_cols = [c for c in [ID_COL, "vdate", TARGET] if c in df_prep.columns]
    row_df = df_prep.loc[[row_idx], base_cols]

    scored = score_and_render(row_df, row_X, pipe)

    st.subheader("Prediction")
    show_cols = [
        c
        for c in [
            ID_COL,
            "vdate",
            TARGET,
            "lengthofstay_Pred",
            "lengthofstay_Pred_Rounded",
            "discharged_Pred",
        ]
        if c in scored.columns
    ]
    st.write(scored[show_cols])

    # Highlight expected discharge date
    if "discharged_Pred" in scored.columns:
        discharge_date = scored["discharged_Pred"].iloc[0]
        if pd.notnull(discharge_date):
            st.metric(
                "Expected discharge date",
                value=str(discharge_date.date()),
            )

    # Procedure-level context & distribution
    if PROC_COL and proc_stats is not None:
        show_procedure_context(df_prep, PROC_COL, proc_stats, row_idx, scored)

    # Feature contributions / importances
    explain_prediction(pipe, row_X)

# -----------------------------
# Mode: Upload CSV
# -----------------------------
elif mode == "Upload CSV":
    st.write(
        "Upload a CSV with the **same schema** as the training file (LengthOfStay.csv)."
    )
    up = st.file_uploader("CSV", type=["csv"])
    if up:
        df_u = pd.read_csv(up)
        for c in DATE_COLS:
            if c in df_u.columns:
                df_u[c] = pd.to_datetime(df_u[c], errors="coerce")
        df_u_prep, X_u, _, _ = prepare(df_u)
        base_cols = [c for c in [ID_COL, "vdate", TARGET] if c in df_u_prep.columns]
        scored = score_and_render(df_u_prep[base_cols], X_u, pipe)
        st.success(f"Scored {len(scored)} rows.")
        st.dataframe(scored.head(20))
        st.download_button(
            "Download predictions CSV",
            scored.to_csv(index=False).encode("utf-8"),
            file_name="LoS_Predictions.csv",
            mime="text/csv",
        )

# -----------------------------
# Mode: Diagnostics
# -----------------------------
elif mode == "Diagnostics":
    st.subheader("Test-set metrics")
    y_pred = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)
    st.write(
        pd.DataFrame(
            [
                {"Model": model_key, "MAE": mae, "RMSE": rmse, "R2": r2},
            ]
        )
    )

    st.subheader("Residuals")
    resid = y_test - y_pred
    st.line_chart(pd.DataFrame({"residuals": resid}).reset_index(drop=True))

    st.subheader("Prediction vs. Actual (sample)")
    show = pd.DataFrame({"actual": y_test.values, "pred": y_pred}).head(500)
    st.scatter_chart(show)

    st.subheader("Global feature importance / contributions")
    # Use one row just to trigger explain_prediction; trees will show global importances
    explain_prediction(pipe, X_test.head(1))

# -----------------------------
# Mode: Manual input (demo mode)
# -----------------------------
else:  # "Manual input"
    st.write(
        "Manually specify factors to get a predicted length of stay. "
        "Defaults come from typical values in the dataset."
    )

    # Visit date + optional ID
    visit_date = st.date_input("Visit date", value=pd.Timestamp.today().date())
    base_row = {}
    base_row["vdate"] = pd.to_datetime(visit_date)
    if ID_COL in df_raw.columns:
        base_row[ID_COL] = st.text_input("Patient ID (optional)", value="DEMO-1")

    # We collect manual values for all feature columns except number_of_issues
    manual_feature_cols = [c for c in feature_cols if c != "number_of_issues"]

    num_cols = df_prep[manual_feature_cols].select_dtypes(
        exclude=["object", "category"]
    ).columns
    cat_cols = df_prep[manual_feature_cols].select_dtypes(
        include=["object", "category"]
    ).columns

    st.markdown("#### Numeric features")
    manual_values = {}
    for col in num_cols:
        col_median = float(df_prep[col].median())
        manual_values[col] = st.number_input(col, value=col_median)

    st.markdown("#### Categorical / indicator features")
    for col in cat_cols:
        # use observed levels for this column
        options = sorted(df_prep[col].dropna().astype(str).unique().tolist())
        if not options:
            manual_values[col] = ""
        else:
            manual_values[col] = st.selectbox(col, options, key=f"manual_{col}")

    if st.button("Predict length of stay"):
        # Build a one-row raw dataframe with manual values
        manual_row_dict = {c: np.nan for c in df_raw.columns}
        for k, v in base_row.items():
            if k in manual_row_dict:
                manual_row_dict[k] = v
        for k, v in manual_values.items():
            if k in manual_row_dict:
                manual_row_dict[k] = v

        df_manual_raw = pd.DataFrame([manual_row_dict])

        # Append to raw dataset so that cleaning/standardization uses dataset stats
        df_aug = pd.concat([df_raw, df_manual_raw], ignore_index=True)
        df_aug_prep, X_aug, _, _ = prepare(df_aug)

        manual_idx = df_aug_prep.index[-1]
        X_manual = X_aug.loc[[manual_idx]]

        base_cols = [c for c in [ID_COL, "vdate", TARGET] if c in df_aug_prep.columns]
        base_df_manual = df_aug_prep.loc[[manual_idx], base_cols]

        scored_manual = score_and_render(base_df_manual, X_manual, pipe)

        st.subheader("Manual prediction")
        show_cols = [
            c
            for c in [
                ID_COL,
                "vdate",
                "lengthofstay_Pred",
                "lengthofstay_Pred_Rounded",
                "discharged_Pred",
            ]
            if c in scored_manual.columns
        ]
        st.write(scored_manual[show_cols])

        if "discharged_Pred" in scored_manual.columns:
            discharge_date = scored_manual["discharged_Pred"].iloc[0]
            if pd.notnull(discharge_date):
                st.metric(
                    "Expected discharge date",
                    value=str(discharge_date.date()),
                )

        # Show procedure context if we have a procedure
        if PROC_COL and proc_stats is not None and PROC_COL in df_aug_prep.columns:
            # Use the procedure value on the manual row
            show_procedure_context(
                df_aug_prep,
                PROC_COL,
                proc_stats,
                manual_idx,
                scored_manual,
            )

        explain_prediction(pipe, X_manual)
