import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Badminton In-Match Prediction Dashboard",
    page_icon="🏸",
    layout="wide",
)

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

CAL_MODEL = MODELS_DIR / "badminton_best_model_calibrated.pkl"
BASE_MODEL = MODELS_DIR / "badminton_best_model_base.pkl"
META_PATH = MODELS_DIR / "model_metadata.json"

EDA_FILES = [
    "eda_missing_values.png",
    "eda_overview.png",
    "eda_feature_distributions.png",
    "eda_boxplot_g1_score_diff.png",
]

PERFORMANCE_FILES = [
    "roc_curves_test.png",
    "confusion_matrix_final.png",
    "learning_curve_tuned_model.png",
    "calibration_curve_validation.png",
]

EXPLAINABILITY_FILES = [
    "feature_importance_model.png",
    "shap_summary_bar.png",
    "shap_summary_beeswarm.png",
    "shap_local_win_case.png",
    "shap_local_loss_case.png",
    "shap_waterfall_win.png",
    "lime_win_case.png",
    "lime_loss_case.png",
]

CSV_FILES = [
    "baseline_validation_results.csv",
    "final_test_metrics.csv",
    "shap_feature_importance.csv",
]


@st.cache_resource
def load_model():
    if CAL_MODEL.exists():
        return joblib.load(CAL_MODEL), "Calibrated model"
    if BASE_MODEL.exists():
        return joblib.load(BASE_MODEL), "Base model"
    return None, None


@st.cache_data
def load_metadata():
    if META_PATH.exists():
        with open(META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def available_image(path: Path):
    return path.exists() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}


def available_csv(path: Path):
    return path.exists() and path.suffix.lower() == ".csv"


def make_default_input_row(features):
    defaults = {
        "g1_t1": 21.0,
        "g1_t2": 18.0,
        "g1_score_diff": 3.0,
        "g1_total": 39.0,
        "t1_win_pct_g1": 21 / 39,
        "round_num": 6.0,
        "tournament_tier": 4.0,
        "team_one_most_consecutive_points_game_1": 4.0,
        "team_two_most_consecutive_points_game_1": 2.0,
        "team_one_game_points_game_1": 1.0,
        "team_two_game_points_game_1": 0.0,
        "consec_g1_diff": 2.0,
        "game_pts_g1_diff": 1.0,
    }
    return {f: defaults.get(f, 0.0) for f in features}


def compute_derived_fields(row):
    if "g1_t1" in row and "g1_t2" in row:
        row["g1_score_diff"] = row["g1_t1"] - row["g1_t2"]
        row["g1_total"] = row["g1_t1"] + row["g1_t2"]
        row["t1_win_pct_g1"] = (row["g1_t1"] / row["g1_total"]) if row["g1_total"] else 0.0

    if (
        "team_one_most_consecutive_points_game_1" in row
        and "team_two_most_consecutive_points_game_1" in row
    ):
        row["consec_g1_diff"] = (
            row["team_one_most_consecutive_points_game_1"]
            - row["team_two_most_consecutive_points_game_1"]
        )

    if (
        "team_one_game_points_game_1" in row
        and "team_two_game_points_game_1" in row
    ):
        row["game_pts_g1_diff"] = (
            row["team_one_game_points_game_1"]
            - row["team_two_game_points_game_1"]
        )

    return row


def compute_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "g1_t1" in df.columns and "g1_t2" in df.columns:
        df["g1_score_diff"] = df["g1_t1"] - df["g1_t2"]
        df["g1_total"] = df["g1_t1"] + df["g1_t2"]
        safe_total = df["g1_total"].replace(0, pd.NA)
        df["t1_win_pct_g1"] = (df["g1_t1"] / safe_total).fillna(0.0)

    if (
        "team_one_most_consecutive_points_game_1" in df.columns
        and "team_two_most_consecutive_points_game_1" in df.columns
    ):
        df["consec_g1_diff"] = (
            df["team_one_most_consecutive_points_game_1"]
            - df["team_two_most_consecutive_points_game_1"]
        )

    if (
        "team_one_game_points_game_1" in df.columns
        and "team_two_game_points_game_1" in df.columns
    ):
        df["game_pts_g1_diff"] = (
            df["team_one_game_points_game_1"]
            - df["team_two_game_points_game_1"]
        )

    return df


st.title("🏸 Badminton In-Match Prediction Dashboard")
st.caption(
    "Predict the final match winner after Game 1 and present the full model story in one hosted Streamlit app."
)

model, model_label = load_model()
meta = load_metadata()

with st.sidebar:
    st.header("Project status")
    st.write(f"**Model loaded:** {model_label or 'Not found'}")
    st.write(f"**Metadata loaded:** {'Yes' if meta else 'No'}")
    st.write(f"**Outputs folder:** {'Found' if OUTPUTS_DIR.exists() else 'Missing'}")
    st.markdown("---")
    st.subheader("Required files")
    st.code(
        "models/badminton_best_model_base.pkl\n"
        "models/badminton_best_model_calibrated.pkl  # optional, preferred\n"
        "models/model_metadata.json\n"
        "outputs/*.png\n"
        "outputs/*.csv"
    )

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Single Prediction", "Batch Prediction", "Model Results", "Explainability & EDA"]
)

with tab1:
    left, right = st.columns([1.1, 1])

    with left:
        st.subheader("What this app does")
        st.markdown(
            """
            This dashboard combines:
            - **live match prediction after Game 1**
            - **model performance reporting**
            - **calibration and explainability visuals**
            - **EDA charts for presentation and defense**
            """
        )

        if meta:
            st.subheader("Model summary")
            c1, c2, c3 = st.columns(3)
            c1.metric("Task", meta.get("task_definition", "N/A"))
            c2.metric("Target", meta.get("target", "N/A"))
            c3.metric("Feature count", len(meta.get("features", [])))
            st.json(meta)
        else:
            st.info("Add `models/model_metadata.json` to display task metadata.")

    with right:
        metrics_path = OUTPUTS_DIR / "final_test_metrics.csv"
        if available_csv(metrics_path):
            st.subheader("Final test metrics")
            st.dataframe(pd.read_csv(metrics_path), use_container_width=True)
        else:
            st.info("Add `outputs/final_test_metrics.csv` to show final evaluation metrics.")
with tab2:
    st.subheader("Single match prediction")

    if model is None or meta is None:
        st.warning("Add the model files and metadata first.")
    else:
        features = meta["features"]
        defaults = make_default_input_row(features)

        # Feature labels (user-friendly)
        feature_labels = {
            "g1_t1": "Team 1 Score (Game 1)",
            "g1_t2": "Team 2 Score (Game 1)",
            "g1_score_diff": "Score Difference (Team1 - Team2)",
            "g1_total": "Total Points in Game 1",
            "t1_win_pct_g1": "Team 1 Win % in Game 1 (0–1)",
            "round_num": "Tournament Round (e.g., 6 = Quarterfinal)",
            "tournament_tier": "Tournament Tier (1–6)",
            "team_one_most_consecutive_points_game_1": "Team 1 Max Consecutive Points",
            "team_two_most_consecutive_points_game_1": "Team 2 Max Consecutive Points",
            "team_one_game_points_game_1": "Team 1 Game Points",
            "team_two_game_points_game_1": "Team 2 Game Points",
            "consec_g1_diff": "Consecutive Points Difference",
            "game_pts_g1_diff": "Game Points Difference",
        }

        with st.form("predict_form"):
            cols = st.columns(2)
            user_row = {}

            for i, feat in enumerate(features):
                with cols[i % 2]:
                    step_value = 0.01 if "pct" in feat else 1.0
                    label = feature_labels.get(feat, feat)

                    user_row[feat] = st.number_input(
                        label,
                        value=float(defaults.get(feat, 0.0)),
                        step=step_value,
                    )

            # ✅ THIS MUST ALIGN WITH for-loop (NOT inside it)
            auto_derive = st.checkbox(
                "Auto-recompute derived Game 1 fields", value=True
            )

            submitted = st.form_submit_button("Predict")

        if submitted:
            row = dict(user_row)

            if auto_derive:
                row = compute_derived_fields(row)

            x = pd.DataFrame([row])[features]
            prob = float(model.predict_proba(x)[0, 1])
            pred = int(prob >= 0.5)

            c1, c2, c3 = st.columns(3)
            c1.metric("Team 1 win probability", f"{prob:.2%}")
            c2.metric("Predicted class", pred)
            c3.metric("Prediction label", "Team 1 wins" if pred == 1 else "Team 2 wins")

            st.dataframe(pd.DataFrame([row]), use_container_width=True)

with tab3:
    st.subheader("Batch prediction from CSV")
    st.write("Upload a CSV containing the required feature columns.")

    if model is None or meta is None:
        st.warning("Add the model files and metadata first.")
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])

        sample_path = ROOT / "sample_batch_input.csv"
        if sample_path.exists():
            st.download_button(
                "Download sample input CSV",
                data=sample_path.read_bytes(),
                file_name="sample_batch_input.csv",
                mime="text/csv",
            )

        if uploaded is not None:
            df = pd.read_csv(uploaded)
            df = compute_derived_columns(df)

            features = meta["features"]
            missing = [c for c in features if c not in df.columns]

            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                probs = model.predict_proba(df[features])[:, 1]
                preds = (probs >= 0.5).astype(int)

                out = df.copy()
                out["team_1_win_probability"] = probs
                out["predicted_class"] = preds
                out["prediction_label"] = out["predicted_class"].map(
                    {1: "Team 1 wins", 0: "Team 2 wins"}
                )out["confidence"] = pd.cut(
    out["team_1_win_probability"],
    bins=[0, 0.55, 0.7, 1],
    labels=["Low", "Medium", "High"]
)

                st.dataframe(out, use_container_width=True)

                st.download_button(
                    "Download predictions CSV",
                    data=out.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv",
                )

with tab4:
    st.subheader("Model results")

    for fname in CSV_FILES:
        path = OUTPUTS_DIR / fname
        if available_csv(path):
            st.markdown(f"**{fname}**")
            st.dataframe(pd.read_csv(path), use_container_width=True)

    for fname in PERFORMANCE_FILES:
        path = OUTPUTS_DIR / fname
        if available_image(path):
            st.image(
                str(path),
                caption=fname.replace("_", " ").replace(".png", "").title(),
                use_container_width=True,
            )

with tab5:
    st.subheader("Explainability and EDA")
    st.markdown("Use this section for project defense, report screenshots, and dashboard storytelling.")

    st.markdown("### Explainability")
    for fname in EXPLAINABILITY_FILES:
        path = OUTPUTS_DIR / fname
        if available_image(path):
            st.image(
                str(path),
                caption=fname.replace("_", " ").replace(".png", "").title(),
                use_container_width=True,
            )

    shap_dep_files = sorted(OUTPUTS_DIR.glob("shap_dependence_*.png"))
    for path in shap_dep_files:
        st.image(str(path), caption=path.name, use_container_width=True)

    st.markdown("### EDA")
    for fname in EDA_FILES:
        path = OUTPUTS_DIR / fname
        if available_image(path):
            st.image(
                str(path),
                caption=fname.replace("_", " ").replace(".png", "").title(),
                use_container_width=True,
            )
