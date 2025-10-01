import numpy as np
import pandas as pd
from typing import Tuple


def compute_ks(y_true: np.ndarray, y_score: np.ndarray) -> float:
	"""Kolmogorov–Smirnov statistic for binary classification."""
	order = np.argsort(-y_score)
	y_true_sorted = y_true[order]
	cum_pos = np.cumsum(y_true_sorted) / np.sum(y_true_sorted)
	cum_neg = np.cumsum(1 - y_true_sorted) / np.sum(1 - y_true_sorted)
	ks = np.max(np.abs(cum_pos - cum_neg))
	return float(ks)


def compute_gini(y_true: np.ndarray, y_score: np.ndarray) -> float:
	"""Gini = 2 * AUC - 1; robust to NaNs."""
	from sklearn.metrics import roc_auc_score
	mask = ~np.isnan(y_score)
	auc = roc_auc_score(y_true[mask], y_score[mask])
	return float(2 * auc - 1)


def population_stability_index(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
	"""Compute PSI between two distributions using quantile bins of expected."""
	expected = pd.Series(expected).replace([np.inf, -np.inf], np.nan).dropna()
	actual = pd.Series(actual).replace([np.inf, -np.inf], np.nan).dropna()
	quantiles = np.quantile(expected, np.linspace(0, 1, bins + 1))
	quantiles[0] = -np.inf
	quantiles[-1] = np.inf
	exp_counts, _ = np.histogram(expected, bins=quantiles)
	act_counts, _ = np.histogram(actual, bins=quantiles)
	exp_prop = np.clip(exp_counts / max(exp_counts.sum(), 1), 1e-6, 1)
	act_prop = np.clip(act_counts / max(act_counts.sum(), 1), 1e-6, 1)
	psi = np.sum((act_prop - exp_prop) * np.log(act_prop / exp_prop))
	return float(psi)


def ks_gini_auc(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float, float]:
	from sklearn.metrics import roc_auc_score
	auc = roc_auc_score(y_true, y_score)
	return compute_ks(y_true, y_score), compute_gini(y_true, y_score), float(auc) 