import pandas as pd
import numpy as np
from pathlib import Path


def load_csv(path: str) -> pd.DataFrame:
	path_obj = Path(path)
	if not path_obj.exists():
		raise FileNotFoundError(f"File not found: {path}")
	return pd.read_csv(path_obj)


def create_sample_german_credit(path: str) -> None:
	"""Create a small synthetic dataset similar to UCI German Credit with a binary target."""
	rng = np.random.default_rng(42)
	n = 1000
	data = pd.DataFrame({
		"age": rng.integers(18, 75, size=n),
		"duration_months": rng.integers(4, 60, size=n),
		"amount": rng.integers(250, 20000, size=n),
		"housing": rng.choice(["own", "rent", "free"], size=n, p=[0.5, 0.45, 0.05]),
		"job": rng.choice(["unskilled", "skilled", "management"], size=n, p=[0.3, 0.6, 0.1]),
		"sex": rng.choice(["male", "female"], size=n, p=[0.6, 0.4]),
		"foreign_worker": rng.choice([0, 1], size=n, p=[0.85, 0.15]),
	})
	# Latent risk relation
	logit = (
		-2.0
		+ 0.00008 * data["amount"].values
		+ 0.02 * (data["duration_months"].values / 12)
		- 0.015 * (data["age"].values - 35)
		+ 0.3 * (data["housing"].eq("rent")).astype(float)
		+ 0.4 * (data["job"].eq("unskilled")).astype(float)
		+ 0.2 * (data["sex"].eq("male")).astype(float)
		+ 0.25 * data["foreign_worker"].values
	)
	p = 1 / (1 + np.exp(-logit))
	data["target"] = (rng.random(n) < p).astype(int)
	Path(path).parent.mkdir(parents=True, exist_ok=True)
	data.to_csv(path, index=False) 