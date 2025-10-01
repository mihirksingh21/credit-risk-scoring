from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


@dataclass
class WoETransformer:
	bin_edges_: Dict[str, np.ndarray] = None
	woe_map_: Dict[str, Dict[int, float]] = None
	categorical_levels_: Dict[str, List[str]] = None
	min_bins: int = 5
	max_bins: int = 10
	min_bin_size: float = 0.05
	
	def fit(self, X: pd.DataFrame, y: pd.Series, categorical: List[str]) -> "WoETransformer":
		self.bin_edges_ = {}
		self.woe_map_ = {}
		self.categorical_levels_ = {}
		numerical_cols = [c for c in X.columns if c not in categorical]
		for col in numerical_cols:
			self._fit_numeric(col, X[col], y)
		for col in categorical:
			self._fit_categorical(col, X[col], y)
		return self
	
	def transform(self, X: pd.DataFrame) -> pd.DataFrame:
		X_out = X.copy()
		for col, edges in self.bin_edges_.items():
			bins = np.digitize(X_out[col].values, edges, right=False) - 1
			woe_col_map = self.woe_map_[col]
			X_out[col] = np.vectorize(lambda b: woe_col_map.get(b, 0.0))(bins)
		for col, levels in self.categorical_levels_.items():
			cats = pd.Categorical(X_out[col], categories=levels)
			bin_idx = cats.codes
			woe_col_map = self.woe_map_[col]
			X_out[col] = np.vectorize(lambda b: woe_col_map.get(b, 0.0))(bin_idx)
		return X_out
	
	def _fit_numeric(self, col: str, s: pd.Series, y: pd.Series) -> None:
		values = s.replace([np.inf, -np.inf], np.nan).fillna(s.median()).values
		quantiles = np.linspace(0, 1, self.max_bins + 1)
		edges = np.unique(np.quantile(values, quantiles))
		if len(edges) - 1 < self.min_bins:
			edges = np.unique(np.quantile(values, np.linspace(0, 1, self.min_bins + 1)))
		# ensure closed intervals
		edges[0] = -np.inf
		edges[-1] = np.inf
		bins = np.digitize(values, edges, right=False) - 1
		woe_map = self._compute_woe_map(bins, y)
		self.bin_edges_[col] = edges
		self.woe_map_[col] = woe_map
	
	def _fit_categorical(self, col: str, s: pd.Series, y: pd.Series) -> None:
		levels = list(pd.Series(s).astype(str).fillna("missing").unique())
		cats = pd.Categorical(s.astype(str).fillna("missing"), categories=levels)
		bins = cats.codes
		woe_map = self._compute_woe_map(bins, y)
		self.categorical_levels_[col] = levels
		self.woe_map_[col] = woe_map
	
	def _compute_woe_map(self, bins: np.ndarray, y: pd.Series) -> Dict[int, float]:
		"""Compute WoE with Laplace smoothing to avoid -inf/inf."""
		y = y.values
		bin_ids = np.unique(bins)
		woe_map: Dict[int, float] = {}
		pos_total = max(y.sum(), 1)
		neg_total = max((1 - y).sum(), 1)
		for b in bin_ids:
			pos = np.sum((bins == b) & (y == 1)) + 0.5
			neg = np.sum((bins == b) & (y == 0)) + 0.5
			woe = np.log((pos / pos_total) / (neg / neg_total))
			woe_map[int(b)] = float(woe)
		return woe_map 