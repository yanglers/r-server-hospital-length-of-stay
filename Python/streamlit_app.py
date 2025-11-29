# streamlit_app.py
import os
from math import sqrt
from pathlib import Path
from typing import Optional

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
from sklearn.inspection import permutation_importance

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

PROCEDURE_COL_CANDIDATES = [
    "procedure",
    "primaryprocedure",
    "surgery",
    "aprdrgdescription",
]
SITE_COL_CANDIDATES = [
    "facid",
    "site",
    "clinic",
    "hospital",
]

AT_RISK_THRESHOLD_DAYS = 5  # for dashboard risk band

# Friendly labels for UI
FEATURE_LABELS = {
    "hematocrit": "Hematocrit (red blood cell % of blood)",
    "neutrophils": "Neutrophil count (white blood cells)",
    "sodium": "Sodium (blood level)",
    "glucose": "Glucose (blood sugar)",
    "bloodureanitro": "Blood urea nitrogen (kidney function)",
    "creatinine": "Creatinine (kidney function)",
    "bmi": "Body mass index (kg/m²)",
    "pulse": "Heart rate (beats per minute)",
    "respiration": "Respiratory rate (breaths per minute)",
    "hemo": "Anemia / low hemoglobin",
    "dialysisrenalendstage": "Dialysis / end-stage renal disease",
    "asthma": "Asthma",
    "irondef": "Iron deficiency",
    "pneum": "Pneumonia",
    "substancedependence": "Substance dependence",
    "psychologicaldisordermajor": "Major psychological disorder",
    "depress": "Depression",
    "psychother": "Receiving psychotherapy",
    "fibrosisandother": "Pulmonary fibrosis / other lung disease",
    "malnutrition": "Malnutrition",
    "gender": "Gender",
    "rcount": "Readmission count",
    "secondarydiagnosisnonicd9": "Secondary diagnosis (non-ICD9)",
    "number_of_issues": "Number of comorbid conditions",
    "facid": "Clinic / hospital site",
    "age": "Age (years)",
}

def pretty_label(col: str) -> str:
    if col in FEATURE_LABELS:
        return FEATURE_LABELS[col]
    return col.replace("_", " ").title()


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
    cat_cols = df[cols_to_consider].select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

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
    # z-score selected continuous columns
    present_cont = [c for c in CONTINUOUS_TO_STANDARDIZE if c in df.columns]
    for c in present_cont:
        mean, std = df[c].mean(), df[c].std(ddof=0)
        df[c] = (df[c] - mean) / (std if std else 1.0)

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
# Model loading & fallback training
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
    joblib.dump(pipe, MODEL_DIR / "GBT_40_model.joblib")

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


def explain_prediction(
    pipe,
    X_row: pd.DataFrame,
    X_background: Optional[pd.DataFrame] = None,
    y_background: Optional[pd.Series] = None,
    max_features: int = 10,
):
    """
    Show either:
    - local contributions for linear models (coef_),
    - global feature importances for tree-based models (feature_importances_),
    - or permutation importance for any fallback model (e.g., neural net).
    """
    st.subheader("Top contributing factors")

    model = pipe.named_steps.get("model")
    prep = pipe.named_steps.get("prep")
    if model is None or prep is None:
        st.info("Cannot introspect this model pipeline.")
        return

    Xt_row = prep.transform(X_row)
    try:
        feat_names = prep.get_feature_names_out()
    except Exception:
        feat_names = np.array([f"f{i}" for i in range(Xt_row.shape[1])])

    # 1) Linear models: local contributions
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_).ravel()
        contrib = Xt_row.reshape(-1) * coef
        df_contrib = pd.DataFrame(
            {"feature_encoded": feat_names, "contribution": contrib}
        )
        df_contrib["abs_contribution"] = df_contrib["contribution"].abs()
        df_contrib = df_contrib.sort_values(
            "abs_contribution", ascending=False
        ).head(max_features)

        st.caption(
            "Local contributions from a linear model "
            "(prediction ≈ intercept + Σ feature · coefficient)."
        )
        st.dataframe(df_contrib[["feature_encoded", "contribution"]])
        return

    # 2) Tree-based models: global feature_importances_
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
        df_imp = pd.DataFrame(
            {"feature_encoded": feat_names, "importance": importances}
        ).sort_values("importance", ascending=False)

        st.caption("Global feature importances from a tree-based model.")
        st.dataframe(df_imp.head(max_features))
        return

    # 3) Fallback: permutation importance
    if X_background is None or y_background is None:
        st.info(
            "This model does not expose coefficients or feature importances, "
            "and no background data was provided to compute permutation importance."
        )
        return

    X_bg = X_background.copy()
    y_bg = y_background.copy()
    if len(X_bg) > 2000:
        sample_idx = np.random.choice(len(X_bg), size=2000, replace=False)
        X_bg = X_bg.iloc[sample_idx]
        y_bg = y_bg.iloc[sample_idx]

    Xt_bg = prep.transform(X_bg)
    result = permutation_importance(
        model, Xt_bg, y_bg, n_repeats=5, random_state=0, n_jobs=-1
    )

    df_perm = pd.DataFrame(
        {
            "feature_encoded": feat_names,
            "importance": result.importances_mean,
        }
    ).sort_values("importance", ascending=False)

    st.caption("Global permutation importance (works for any model).")
    st.dataframe(df_perm.head(max_features))


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
    st.write(f"**Procedure**: {proc_val}")

    if proc_val not in proc_stats.index:
        st.info("No historical stats available for this procedure.")
        return

    stats = proc_stats.loc[proc_val]
    mean_los = stats["mean"]
    std_los = stats["std"]
    count_los = int(stats["count"])

    st.write(
        f"Average length of stay for this procedure: **{mean_los:.2f} days** "
        f"(n={count_los})."
    )

    if "lengthofstay_Pred_Rounded" in scored_row.columns:
        pred_los = float(scored_row["lengthofstay_Pred_Rounded"].iloc[0])
    else:
        pred_los = float(scored_row["lengthofstay_Pred"].iloc[0])

    if pd.notnull(std_los) and std_los > 0:
        z = (pred_los - mean_los) / std_los
        st.write(f"Prediction is **{z:+.2f}** standard deviations from the procedure mean.")
        if abs(z) > 2:
            st.warning(
                "This predicted stay looks unusual for this procedure (|z| > 2)."
            )

    data_proc = df_prep[df_prep[proc_col] == proc_val][TARGET].dropna()
    if len(data_proc) >= 5:
        bins = min(20, max(5, int(len(data_proc) ** 0.5)))
        counts, edges = np.histogram(data_proc, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        hist_df = pd.DataFrame({"LoS_bin": centers, "Count": counts}).set_index(
            "LoS_bin"
        )
        st.write("Historical LOS distribution for this procedure:")
        st.bar_chart(hist_df)
    else:
        st.info("Not enough historical data to show a distribution for this procedure.")


def format_model_name(key: str, pipe) -> str:
    model = pipe.named_steps.get("model")
    cls = model.__class__.__name__ if model is not None else "Unknown"
    return f"{key} ({cls})"


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Length of Stay Dashboard", layout="wide")
st.title("Length of Stay Dashboard")

models = load_models()
if not models:
    st.info("No saved models found in ./models_sklearn_los.")
    df_raw_tmp = load_raw()
    if st.button("Train a quick baseline GBT model now"):
        rmse = quick_train_and_save(df_raw_tmp)
        st.success(
            f"Model trained and saved. Test RMSE ≈ {rmse:.3f}. Click **Rerun** to load it."
        )
        st.stop()
    else:
        st.stop()

# Data for inference / diagnostics
df_raw = load_raw()
df_prep, X_all, y_all, feature_cols = prepare(df_raw)

PROC_COL = next((c for c in PROCEDURE_COL_CANDIDATES if c in df_prep.columns), None)
SITE_COL = next((c for c in SITE_COL_CANDIDATES if c in df_prep.columns), None)

if PROC_COL:
    proc_stats = df_prep.groupby(PROC_COL)[TARGET].agg(["mean", "std", "count"])
else:
    proc_stats = None

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_all, y_all, df_prep.index, train_size=TRAIN_PCT, random_state=RANDOM_STATE
)

model_keys = sorted(models.keys())
model_key = st.sidebar.selectbox(
    "Model",
    model_keys,
    index=0,
    format_func=lambda k: format_model_name(k, models[k]),
)
pipe = models[model_key]

st.sidebar.markdown("### View")
mode = st.sidebar.radio(
    "Mode", ["Pick a patient", "Upload CSV", "Diagnostics", "Manual input"], index=0
)

# -----------------------------
# Mode: Pick a patient (dashboard)
# -----------------------------
if mode == "Pick a patient":
    filters_col, main_panel = st.columns([1, 3])

    # Left: filters
    with filters_col:
        st.markdown("#### Filters")

        if PROC_COL and PROC_COL in df_prep.columns:
            proc_options = ["All procedures"] + sorted(
                df_prep[PROC_COL].dropna().astype(str).unique().tolist()
            )
            proc_filter = st.selectbox("Procedure", proc_options)
        else:
            proc_filter = "All procedures"

        if SITE_COL and SITE_COL in df_prep.columns:
            site_options = ["All sites"] + sorted(
                df_prep[SITE_COL].dropna().astype(str).unique().tolist()
            )
            site_filter = st.selectbox(pretty_label(SITE_COL), site_options)
        else:
            site_filter = "All sites"

        st.markdown("##### Comorbidities")
        issue_filters = {}
        present_issue_cols = [c for c in ISSUE_INDICATORS if c in df_prep.columns]
        for col in present_issue_cols:
            label = pretty_label(col)
            issue_filters[col] = st.checkbox(label, value=False, key=f"filter_{col}")

        st.markdown("##### Search")
        search_text = st.text_input("Search patient / case #", value="")

    subset = df_prep.loc[idx_test].copy()

    if PROC_COL and proc_filter != "All procedures":
        subset = subset[subset[PROC_COL].astype(str) == proc_filter]

    if SITE_COL and site_filter != "All sites":
        subset = subset[subset[SITE_COL].astype(str) == site_filter]

    for col, required in issue_filters.items():
        if required and col in subset.columns:
            subset = subset[subset[col].fillna(0) > 0]

    if search_text and ID_COL in subset.columns:
        subset = subset[
            subset[ID_COL].astype(str).str.contains(search_text, case=False, na=False)
        ]

    if subset.empty:
        with main_panel:
            st.warning("No patients match the current filters.")
        st.stop()

    # Score filtered cohort
    X_subset = X_all.loc[subset.index]
    base_cols_queue = [c for c in [ID_COL, "vdate", PROC_COL, TARGET] if c in df_prep.columns]
    scored_subset = score_and_render(
        df_prep.loc[subset.index, base_cols_queue], X_subset, pipe
    )

    median_los = float(np.median(scored_subset["lengthofstay_Pred"]))
    at_risk_cases = int(
        (scored_subset["lengthofstay_Pred_Rounded"] >= AT_RISK_THRESHOLD_DAYS).sum()
    )
    total_cases = len(scored_subset)

    scored_subset["Risk band"] = pd.cut(
        scored_subset["lengthofstay_Pred"],
        bins=[-np.inf, 3, AT_RISK_THRESHOLD_DAYS, np.inf],
        labels=["Low", "Medium", "High"],
    )

    with main_panel:
        st.markdown("#### Overview (filtered cohort)")
        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            st.metric("Median predicted LOS", f"{median_los:.1f} days")
        with kpi2:
            st.metric(
                f"At-risk cases (≥ {AT_RISK_THRESHOLD_DAYS} days)", f"{at_risk_cases}"
            )
        with kpi3:
            st.metric("Cases in view", f"{total_cases}")

        st.markdown("---")

        # Patient queue
        st.markdown("#### Patient queue")
        queue_cols = []
        if ID_COL in scored_subset.columns:
            queue_cols.append(ID_COL)
        if PROC_COL and PROC_COL in scored_subset.columns:
            queue_cols.append(PROC_COL)
        queue_cols += [
            c
            for c in [
                "vdate",
                "lengthofstay_Pred_Rounded",
                "Risk band",
                "discharged_Pred",
            ]
            if c in scored_subset.columns
        ]

        queue_display = scored_subset[queue_cols].copy()
        if "lengthofstay_Pred_Rounded" in queue_display.columns:
            queue_display.rename(
                columns={"lengthofstay_Pred_Rounded": "Predicted LOS (days)"},
                inplace=True,
            )
        if "discharged_Pred" in queue_display.columns:
            queue_display["discharged_Pred"] = queue_display["discharged_Pred"].dt.date

        st.dataframe(queue_display, height=350)

        # Select case for detail
        if ID_COL in scored_subset.columns:
            case_choices = scored_subset[ID_COL].astype(str).tolist()
            selected_case = st.selectbox("Select case for detailed view", case_choices)
            row_idx = scored_subset.index[
                scored_subset[ID_COL].astype(str) == selected_case
            ][0]
        else:
            row_idx = scored_subset.index[0]

        row_X = X_all.loc[[row_idx]]
        base_cols_detail = [
            c
            for c in [
                ID_COL,
                "vdate",
                TARGET,
                PROC_COL,
                "gender",
                "age",
                "bmi",
            ]
            if c in df_prep.columns
        ]
        row_base = df_prep.loc[[row_idx], base_cols_detail]
        scored_row = score_and_render(row_base, row_X, pipe)

        st.markdown("---")
        case_left, case_right = st.columns([1, 1])

        with case_left:
            st.markdown("#### Selected case")
            pred_los = float(scored_row["lengthofstay_Pred_Rounded"].iloc[0])
            st.markdown(
                f"<h2 style='margin-bottom:0'>{pred_los:.1f} days</h2>"
                "<p style='color:gray;margin-top:0'>Predicted length of stay</p>",
                unsafe_allow_html=True,
            )

            if "discharged_Pred" in scored_row.columns:
                discharge_date = scored_row["discharged_Pred"].iloc[0]
                if pd.notnull(discharge_date):
                    st.metric(
                        "Expected discharge date",
                        value=str(discharge_date.date()),
                    )

            st.markdown("##### Patient snapshot")
            info_rows = []
            for col in ["age", "gender", "bmi", PROC_COL]:
                if col in scored_row.columns:
                    info_rows.append((pretty_label(col), str(scored_row[col].iloc[0])))
            if info_rows:
                info_df = pd.DataFrame(
                    info_rows, columns=["Attribute", "Value"]
                ).set_index("Attribute")
                st.table(info_df)

        with case_right:
            if PROC_COL and proc_stats is not None:
                show_procedure_context(df_prep, PROC_COL, proc_stats, row_idx, scored_row)

            explain_prediction(
                pipe, row_X, X_background=X_train, y_background=y_train
            )

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
    explain_prediction(
        pipe, X_test.head(1), X_background=X_test, y_background=y_test
    )

# -----------------------------
# Mode: Manual input (demo mode)
# -----------------------------
else:  # "Manual input"
    st.write(
        "Manually specify factors to get a predicted length of stay. "
        "Defaults come from typical values in the dataset."
    )

    form_col, result_col = st.columns([2, 1])

    with form_col:
        visit_date = st.date_input(
            "Admission date", value=pd.Timestamp.today().date()
        )
        base_row = {"vdate": pd.to_datetime(visit_date)}
        if ID_COL in df_raw.columns:
            base_row[ID_COL] = st.text_input("Patient ID", value="DEMO-1")

        manual_feature_cols = [c for c in feature_cols if c != "number_of_issues"]

        num_cols = df_prep[manual_feature_cols].select_dtypes(
            exclude=["object", "category"]
        ).columns
        cat_cols = df_prep[manual_feature_cols].select_dtypes(
            include=["object", "category"]
        ).columns

        st.markdown("#### Clinical measurements")
        manual_values = {}
        for col in num_cols:
            col_median = float(df_prep[col].median())
            manual_values[col] = st.number_input(
                pretty_label(col), value=col_median, key=f"num_{col}"
            )

        st.markdown("#### Diagnoses and other factors")
        for col in cat_cols:
            options = sorted(df_prep[col].dropna().astype(str).unique().tolist())
            label = pretty_label(col)
            if not options:
                manual_values[col] = ""
            else:
                manual_values[col] = st.selectbox(
                    label, options, key=f"cat_{col}"
                )

        predict_clicked = st.button("Predict length of stay")

    if predict_clicked:
        manual_row_dict = {c: np.nan for c in df_raw.columns}
        for k, v in base_row.items():
            if k in manual_row_dict:
                manual_row_dict[k] = v
        for k, v in manual_values.items():
            if k in manual_row_dict:
                manual_row_dict[k] = v

        df_manual_raw = pd.DataFrame([manual_row_dict])
        df_aug = pd.concat([df_raw, df_manual_raw], ignore_index=True)
        df_aug_prep, X_aug, _, _ = prepare(df_aug)

        manual_idx = df_aug_prep.index[-1]
        X_manual = X_aug.loc[[manual_idx]]

        base_cols = [c for c in [ID_COL, "vdate", TARGET, PROC_COL] if c in df_aug_prep.columns]
        base_df_manual = df_aug_prep.loc[[manual_idx], base_cols]
        scored_manual = score_and_render(base_df_manual, X_manual, pipe)

        with result_col:
            st.subheader("Manual prediction")
            pred_los = float(scored_manual["lengthofstay_Pred_Rounded"].iloc[0])
            st.markdown(
                f"<h2 style='margin-bottom:0'>{pred_los:.1f} days</h2>"
                "<p style='color:gray;margin-top:0'>Predicted length of stay</p>",
                unsafe_allow_html=True,
            )

            if "discharged_Pred" in scored_manual.columns:
                discharge_date = scored_manual["discharged_Pred"].iloc[0]
                if pd.notnull(discharge_date):
                    st.metric(
                        "Expected discharge date",
                        value=str(discharge_date.date()),
                    )

            if PROC_COL and proc_stats is not None and PROC_COL in df_aug_prep.columns:
                show_procedure_context(
                    df_aug_prep,
                    PROC_COL,
                    proc_stats,
                    manual_idx,
                    scored_manual,
                )

            explain_prediction(
                pipe, X_manual, X_background=X_all, y_background=y_all
            )
