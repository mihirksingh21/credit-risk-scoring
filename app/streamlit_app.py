import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import sys
import os
import matplotlib.pyplot as plt
import shap

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.report import generate_pdf_report
from src.utils import compute_ks, compute_gini, population_stability_index
from src.fairness import demographic_parity, equal_opportunity
from src.eval import tune_threshold

ARTIFACT_DIR = Path("artifacts")

st.set_page_config(page_title="Credit Risk Scorecard", layout="wide")
st.title("Credit Risk Scoring - Responsible AI")

@st.cache_resource
def load_artifacts():
    model = joblib.load(ARTIFACT_DIR / "model.joblib")
    woe = joblib.load(ARTIFACT_DIR / "woe.joblib")
    return model, woe

@st.cache_resource
def load_raw_model_and_explainer():
    raw_model_path = ARTIFACT_DIR / "raw_lgbm.joblib"
    if raw_model_path.exists():
        raw_model = joblib.load(raw_model_path)
        explainer = shap.TreeExplainer(raw_model)
        return raw_model, explainer
    return None, None

with st.sidebar:
    st.header("Artifacts")
    if not (ARTIFACT_DIR / "model.joblib").exists():
        st.warning("No model found. Run training: python -m src.model")
    else:
        st.success("Model and encoder loaded")
        st.caption(str(ARTIFACT_DIR))

if (ARTIFACT_DIR / "model.joblib").exists():
    model, woe = load_artifacts()
    raw_model, explainer = load_raw_model_and_explainer()
    
    st.subheader("Batch Scoring")
    upload = st.file_uploader("Upload CSV (no target column)", type=["csv"])
    if upload is not None:
        data = pd.read_csv(upload)
        X_woe = woe.transform(data)
        scores = model.predict_proba(X_woe)[:, 1]
        st.write("Preview:")
        st.dataframe(pd.concat([data.head(), pd.Series(scores[:5], name="score")], axis=1))
        st.download_button("Download scores.csv", pd.Series(scores, name="score").to_csv(index=False), "scores.csv")

        # Optional: PSI vs validation scores if available
        val_scores_path = ARTIFACT_DIR / "val_scores.csv"
        if val_scores_path.exists():
            val_scores = pd.read_csv(val_scores_path).iloc[:, 0]
            psi_upload_vs_val = population_stability_index(val_scores, pd.Series(scores))
            st.info({"psi_upload_vs_val": float(psi_upload_vs_val)})

        # Optional: demographic parity on uploaded predictions by group column
        cols = list(data.columns)
        group_choice = st.selectbox("Optional group column for fairness (Demographic Parity)", ["(none)"] + cols)
        th = st.slider("Decision threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        if group_choice != "(none)":
            y_pred_upload = (scores >= th).astype(int)
            dp_metrics = demographic_parity(y_pred_upload, data[group_choice])
            st.write({"demographic_parity": dp_metrics})
            st.caption("Equal Opportunity needs true labels and is not computed for uploaded, unlabeled data.")

        # Single-record SHAP waterfall and drivers
        if raw_model is not None and explainer is not None:
            st.subheader("Single Record Explainability")
            idx = st.number_input("Row index to explain", min_value=0, max_value=len(data) - 1, value=0, step=1)
            x_row_woe = X_woe.iloc[[idx]]
            try:
                shap_values = explainer.shap_values(x_row_woe)
                # handle list vs array depending on TreeExplainer output
                if isinstance(shap_values, list):
                    # binary: index 1 for positive class
                    shap_row = shap_values[1][0]
                    base_value = explainer.expected_value[1]
                else:
                    shap_row = shap_values[0]
                    base_value = explainer.expected_value
                contrib = pd.Series(shap_row, index=x_row_woe.columns).sort_values(key=np.abs, ascending=False)
                st.write("Top drivers:")
                st.dataframe(pd.DataFrame({"feature": contrib.index, "contribution": contrib.values}).head(10))
                # Waterfall
                fig, ax = plt.subplots(figsize=(8, 6))
                shap.plots._waterfall.waterfall_legacy(base_value, shap_row, feature_names=list(x_row_woe.columns), max_display=15, show=False)
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"SHAP explanation unavailable: {e}")
    
    st.subheader("Metrics & Report")
    # Show KS/Gini based on stored test scores if present
    metrics_file = ARTIFACT_DIR / "test_scores.csv"
    val_scores_file = ARTIFACT_DIR / "val_scores.csv"
    val_labels_file = ARTIFACT_DIR / "val_labels.csv"
    if metrics_file.exists():
        s = pd.read_csv(metrics_file).iloc[:, 0].values
        st.write({"ks": compute_ks((s > s.mean()).astype(int), s), "gini": compute_gini((s > s.mean()).astype(int), s)})
    # Show PSI between stored validation and test scores if both exist
    if val_scores_file.exists() and metrics_file.exists():
        s_val = pd.read_csv(val_scores_file).iloc[:, 0]
        s_test = pd.read_csv(metrics_file).iloc[:, 0]
        psi_val_test = population_stability_index(s_val, s_test)
        st.write({"psi_val_to_test": float(psi_val_test)})

    # Cost-sensitive threshold optimization on validation
    if val_scores_file.exists() and val_labels_file.exists():
        st.subheader("Policy: Cost-sensitive Threshold")
        fp_cost = st.number_input("Cost of False Positive", min_value=0.0, value=1.0, step=0.5)
        fn_cost = st.number_input("Cost of False Negative", min_value=0.0, value=5.0, step=0.5)
        y_val = pd.read_csv(val_labels_file).iloc[:, 0].values.astype(int)
        s_val = pd.read_csv(val_scores_file).iloc[:, 0].values
        best_th, th_stats = tune_threshold(y_val, s_val, fp_cost=fp_cost, fn_cost=fn_cost)
        st.write({"recommended_threshold": float(best_th), **th_stats})

    # SHAP feature importance if available
    shap_path = ARTIFACT_DIR / "shap_importance.csv"
    if shap_path.exists():
        st.subheader("Top Feature Importance (SHAP | validation/test set)")
        shap_imp_df = pd.read_csv(shap_path, index_col=0)
        shap_series = shap_imp_df.iloc[:, 0]
        st.bar_chart(shap_series.head(20))

    # Feature stability (IV & PSI)
    stability_path = ARTIFACT_DIR / "feature_stability.csv"
    if stability_path.exists():
        st.subheader("Feature Stability (IV & PSI val→test)")
        st.dataframe(pd.read_csv(stability_path))
    
    if st.button("Export PDF Report"):
        pdf_path = ARTIFACT_DIR / "credit_policy_report.pdf"
        metrics = {"note": 1.0}
        fairness = {"note": 1.0}
        generate_pdf_report(str(pdf_path), "Credit Risk Model Report", metrics, fairness)
        st.success(f"Saved {pdf_path}")
else:
    st.info("Train a model first to enable the app features.")
