from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, brier_score_loss
from .utils import compute_ks, compute_gini


def evaluate_binary(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
	auc = roc_auc_score(y_true, y_score)
	ks = compute_ks(y_true, y_score)
	gini = compute_gini(y_true, y_score)
	brier = brier_score_loss(y_true, y_score)
	return {"roc_auc": float(auc), "ks": float(ks), "gini": float(gini), "brier": float(brier)}


def tune_threshold(y_true: np.ndarray, y_score: np.ndarray, fp_cost: float = 1.0, fn_cost: float = 5.0) -> Tuple[float, Dict[str, float]]:
	thresholds = np.linspace(0.01, 0.99, 99)
	best_th = 0.5
	best_cost = float("inf")
	best_stats: Dict[str, float] = {}
	for th in thresholds:
		y_pred = (y_score >= th).astype(int)
		fp = int(((y_pred == 1) & (y_true == 0)).sum())
		fn = int(((y_pred == 0) & (y_true == 1)).sum())
		cost = fp_cost * fp + fn_cost * fn
		if cost < best_cost:
			best_cost = cost
			best_th = float(th)
			best_stats = {"fp": fp, "fn": fn, "cost": float(cost)}
	return best_th, best_stats 