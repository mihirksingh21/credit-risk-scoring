from typing import Dict
import numpy as np
import pandas as pd


def demographic_parity(y_pred: np.ndarray, group: pd.Series) -> Dict[str, float]:
	"""P(Ŷ=1|A=a) across groups and max absolute diff."""
	res: Dict[str, float] = {}
	groups = pd.Series(group).astype(str)
	p_all = {}
	for g in groups.unique():
		mask = groups == g
		p_all[g] = float(np.mean(y_pred[mask] == 1))
	res.update({f"rate_{g}": v for g, v in p_all.items()})
	res["dp_gap_max"] = float(max(p_all.values()) - min(p_all.values())) if len(p_all) > 1 else 0.0
	return res


def equal_opportunity(y_true: np.ndarray, y_pred: np.ndarray, group: pd.Series) -> Dict[str, float]:
	"""TPR across groups and max absolute diff."""
	res: Dict[str, float] = {}
	groups = pd.Series(group).astype(str)
	tpr_all = {}
	for g in groups.unique():
		mask = (groups == g) & (y_true == 1)
		if mask.sum() == 0:
			tpr = 0.0
		else:
			tpr = float(np.mean(y_pred[mask] == 1))
		tpr_all[g] = tpr
	res.update({f"tpr_{g}": v for g, v in tpr_all.items()})
	res["eo_gap_max"] = float(max(tpr_all.values()) - min(tpr_all.values())) if len(tpr_all) > 1 else 0.0
	return res 