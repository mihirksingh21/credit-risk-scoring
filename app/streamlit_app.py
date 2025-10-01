import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.report import generate_pdf_report
from src.utils import compute_ks, compute_gini, population_stability_index

ARTIFACT_DIR = Path("artifacts")

st.set_page_config(page_title="Credit Risk Scorecard", layout="wide")
st.title("Credit Risk Scoring - Responsible AI")

@st.cache_resource
def load_artifacts():
    model = joblib.load(ARTIFACT_DIR / "model.joblib")
    woe = joblib.load(ARTIFACT_DIR / "woe.joblib")
    return model, woe

with st.sidebar:
    st.header("Artifacts")
    if not (ARTIFACT_DIR / "model.joblib").exists():
        st.warning("No model found. Run training: python -m src.model")
    else:
        st.success("Model and encoder loaded")
        st.caption(str(ARTIFACT_DIR))

if (ARTIFACT_DIR / "model.joblib").exists():
    model, woe = load_artifacts()
    
    st.subheader("Batch Scoring")
    upload = st.file_uploader("Upload CSV (no target column)", type=["csv"])
    if upload is not None:
        data = pd.read_csv(upload)
        X_woe = woe.transform(data)
        scores = model.predict_proba(X_woe)[:, 1]
        st.write("Preview:")
        st.dataframe(pd.concat([data.head(), pd.Series(scores[:5], name="score")], axis=1))
        st.download_button("Download scores.csv", pd.Series(scores, name="score").to_csv(index=False), "scores.csv")
    
    st.subheader("Metrics & Report")
    metrics_file = ARTIFACT_DIR / "test_scores.csv"
    if metrics_file.exists():
        s = pd.read_csv(metrics_file).iloc[:, 0].values
        st.write({"ks": compute_ks((s > s.mean()).astype(int), s), "gini": compute_gini((s > s.mean()).astype(int), s)})
    
    if st.button("Export PDF Report"):
        pdf_path = ARTIFACT_DIR / "credit_policy_report.pdf"
        metrics = {"note": 1.0}
        fairness = {"note": 1.0}
        generate_pdf_report(str(pdf_path), "Credit Risk Model Report", metrics, fairness)
        st.success(f"Saved {pdf_path}")
else:
    st.info("Train a model first to enable the app features.")
