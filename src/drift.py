from typing import Dict
import pandas as pd
from .utils import population_stability_index


def psi_report(expected: pd.Series, actual: pd.Series, name: str, bins: int = 10) -> Dict[str, float]:
	psi = population_stability_index(expected, actual, bins=bins)
	severity = (
		"none" if psi < 0.1 else
		"medium" if psi < 0.25 else
		"high"
	)
	return {"feature": name, "psi": float(psi), "severity": severity} 