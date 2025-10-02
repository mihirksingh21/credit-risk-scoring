import argparse
import os
from pathlib import Path
from typing import List, Optional
import joblib
import numpy as np
import pandas as pd
import mlflow
import shap
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report
import lightgbm as lgb

from .data import load_csv, create_sample_german_credit
from .features import WoETransformer
from .eval import evaluate_binary, tune_threshold
from .fairness import demographic_parity, equal_opportunity
from .drift import psi_report


ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser()
	p.add_argument("--data_path", type=str, default="data/sample_german_credit.csv")
	p.add_argument("--target", type=str, default="target")
	p.add_argument("--group_col", type=str, default="sex")
	p.add_argument("--test_size", type=float, default=0.2)
	p.add_argument("--val_size", type=float, default=0.2)
	p.add_argument("--random_state", type=int, default=42)
	return p.parse_args()


def infer_categorical_columns(df: pd.DataFrame, target: str) -> List[str]:
	candidates = []
	for col in df.columns:
		if col == target:
			continue
		if df[col].dtype == "object" or df[col].nunique() <= 12:
			candidates.append(col)
	return candidates


def train():
	args = parse_args()
	if not Path(args.data_path).exists():
		create_sample_german_credit(args.data_path)
	df = load_csv(args.data_path)
	assert args.target in df.columns, f"Missing target column {args.target}"
	X = df.drop(columns=[args.target])
	y = df[args.target].astype(int).values
	categorical = infer_categorical_columns(df, args.target)
	
	X_trainval, X_test, y_trainval, y_test = train_test_split(
		X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
	)
	X_train, X_val, y_train, y_val = train_test_split(
		X_trainval, y_trainval, test_size=args.val_size, random_state=args.random_state, stratify=y_trainval
	)
	
	woe = WoETransformer(max_bins=10)
	woe.fit(X_train, pd.Series(y_train), categorical)
	X_tr_woe = woe.transform(X_train)
	X_val_woe = woe.transform(X_val)
	X_te_woe = woe.transform(X_test)
	
	# Monotonic constraints heuristic: assume larger amount/duration increase risk, age decreases
	mono_map = {col: 0 for col in X_tr_woe.columns}
	for col in X_tr_woe.columns:
		if any(k in col for k in ["amount", "duration"]):
			mono_map[col] = 1
		elif "age" in col:
			mono_map[col] = -1
	monotone_constraints = [mono_map[c] for c in X_tr_woe.columns]
	
	lgbm = lgb.LGBMClassifier(
		n_estimators=500,
		learning_rate=0.03,
		max_depth=-1,
		num_leaves=31,
		min_child_samples=50,
		subsample=0.9,
		colsample_bytree=0.9,
		reg_lambda=5.0,
		monotone_constraints=monotone_constraints,
		random_state=args.random_state,
		n_jobs=-1,
	)
	lgbm.fit(X_tr_woe, y_train, eval_set=[(X_val_woe, y_val)], eval_metric="auc")
	
	# Calibrate probabilities
	calibrated = CalibratedClassifierCV(lgbm, cv="prefit", method="isotonic")
	calibrated.fit(X_val_woe, y_val)
	
	# Scores
	val_scores = calibrated.predict_proba(X_val_woe)[:, 1]
	test_scores = calibrated.predict_proba(X_te_woe)[:, 1]
	
	metrics = evaluate_binary(y_val, val_scores)
	best_th, th_stats = tune_threshold(y_val, val_scores)
	y_test_pred = (test_scores >= best_th).astype(int)
	
	# Fairness
	fairness_metrics = {}
	if args.group_col in X_test.columns:
		group_vals = X_test[args.group_col]
		fairness_metrics.update(demographic_parity(y_test_pred, group_vals))
		fairness_metrics.update(equal_opportunity(y_test, y_test_pred, group_vals))
	
	# Drift PSI between val and test scores
	from .utils import population_stability_index
	psi_score = population_stability_index(pd.Series(val_scores), pd.Series(test_scores))
	
	# SHAP explanations
	explainer = shap.TreeExplainer(lgbm)
	shap_vals = explainer.shap_values(X_te_woe)[1] if isinstance(explainer.shap_values(X_te_woe), list) else explainer.shap_values(X_te_woe)
	shap_importance = pd.Series(np.abs(shap_vals).mean(axis=0), index=X_te_woe.columns).sort_values(ascending=False)
	
	# Compute feature IV on training data and feature PSI between val and test
	def _compute_iv_for_feature(col: str) -> float:
		# derive bins for original space then use WoE map
		if col in woe.bin_edges_:
			# numeric
			edges = woe.bin_edges_[col]
			bins = np.digitize(X_train[col].replace([np.inf, -np.inf], np.nan).fillna(X_train[col].median()).values, edges, right=False) - 1
			woe_map = woe.woe_map_[col]
		elif col in woe.categorical_levels_:
			levels = woe.categorical_levels_[col]
			cats = pd.Categorical(X_train[col].astype(str).fillna("missing"), categories=levels)
			bins = cats.codes
			woe_map = woe.woe_map_[col]
		else:
			return 0.0
		bin_ids = np.unique(bins)
		pos_total = max(pd.Series(y_train).sum(), 1)
		neg_total = max(len(y_train) - pd.Series(y_train).sum(), 1)
		iv = 0.0
		for b in bin_ids:
			mask = bins == b
			pos = np.sum((mask) & (y_train == 1))
			neg = np.sum((mask) & (y_train == 0))
			pos_rate = max(pos / pos_total, 1e-10)
			neg_rate = max(neg / neg_total, 1e-10)
			woe_val = float(woe_map.get(int(b), 0.0))
			iv += (pos_rate - neg_rate) * woe_val
		return float(iv)
	
	feature_rows = []
	for col in X_tr_woe.columns:
		iv = _compute_iv_for_feature(col)
		psi_feat = population_stability_index(X_val_woe[col], X_te_woe[col])
		feature_rows.append({"feature": col, "iv": iv, "psi_val_test": float(psi_feat)})
	feature_stability_df = pd.DataFrame(feature_rows).sort_values(["iv"], ascending=False)
	
	# Save artifacts
	ARTIFACT_DIR.mkdir(exist_ok=True)
	joblib.dump(woe, ARTIFACT_DIR / "woe.joblib")
	joblib.dump(calibrated, ARTIFACT_DIR / "model.joblib")
	joblib.dump(lgbm, ARTIFACT_DIR / "raw_lgbm.joblib")
	shap_importance.to_csv(ARTIFACT_DIR / "shap_importance.csv")
	pd.Series(val_scores).to_csv(ARTIFACT_DIR / "val_scores.csv", index=False)
	pd.Series(y_val).to_csv(ARTIFACT_DIR / "val_labels.csv", index=False)
	pd.DataFrame(X_val_woe).to_csv(ARTIFACT_DIR / "X_val_woe.csv", index=False)
	feature_stability_df.to_csv(ARTIFACT_DIR / "feature_stability.csv", index=False)
	pd.Series(test_scores).to_csv(ARTIFACT_DIR / "test_scores.csv", index=False)
	
	# MLflow logging
	mlflow.set_tracking_uri("file:" + str(Path("mlruns").absolute()))
	with mlflow.start_run(run_name="credit_risk_model"):
		for k, v in metrics.items():
			mlflow.log_metric(k, v)
		mlflow.log_metric("best_threshold", best_th)
		mlflow.log_metric("psi_val_test", psi_score)
		for k, v in fairness_metrics.items():
			# only log numeric
			if isinstance(v, (int, float)):
				mlflow.log_metric(f"fair_{k}", float(v))
		mlflow.log_artifact(str(ARTIFACT_DIR / "woe.joblib"))
		mlflow.log_artifact(str(ARTIFACT_DIR / "model.joblib"))
		mlflow.log_artifact(str(ARTIFACT_DIR / "raw_lgbm.joblib"))
		mlflow.log_artifact(str(ARTIFACT_DIR / "shap_importance.csv"))
		mlflow.log_artifact(str(ARTIFACT_DIR / "feature_stability.csv"))
	
	print("Validation metrics:", metrics)
	print("Best threshold:", best_th, th_stats)
	print("Test classification report:\n", classification_report(y_test, y_test_pred))
	print("Fairness:", fairness_metrics)
	print("PSI (val->test):", psi_score)


if __name__ == "__main__":
	train() 