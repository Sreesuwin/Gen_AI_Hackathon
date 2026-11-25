"""Pure-Python + NumPy Isolation Forest fallback implementation.

Implements the Liu, Ting, and Zhou (2008) Isolation Forest algorithm using
random isolation trees, random feature selection, random split thresholds,
and harmonic path-length correction without compiled scikit-learn extensions.

Note: This is a fallback implementation (``PurePythonIsolationForest``), not
scikit-learn's ``IsolationForest``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

EULER_MASCHERONI = 0.5772156649015329


def _c_factor(n: int) -> float:
    """Average path length of unsuccessful search in a BST (harmonic correction)."""
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (math.log(n - 1) + EULER_MASCHERONI) - (2.0 * (n - 1) / n)


@dataclass
class _IsolationNode:
    """Node in a binary isolation tree."""

    size: int
    feature_idx: int | None = None
    split_value: float | None = None
    left: _IsolationNode | None = None
    right: _IsolationNode | None = None

    @property
    def is_leaf(self) -> bool:
        return self.feature_idx is None


class PurePythonIsolationForest:
    """Pure-Python / NumPy Isolation Forest for environments where sklearn cannot load."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: int = 256,
        contamination: float = 0.03,
        random_state: int | None = 42,
    ) -> None:
        if n_estimators <= 0:
            raise ValueError("n_estimators must be positive.")
        if max_samples <= 0:
            raise ValueError("max_samples must be positive.")
        if not (0.0 < contamination < 0.5):
            raise ValueError("contamination must be in (0.0, 0.5).")

        self.n_estimators = int(n_estimators)
        self.max_samples = int(max_samples)
        self.contamination = float(contamination)
        self.random_state = random_state

        self.trees_: list[_IsolationNode] = []
        self.sample_size_: int = 0
        self.c_psi_: float = 0.0
        self.threshold_: float = 1.0

    def _build_tree(
        self,
        X_sub: np.ndarray,
        depth: int,
        height_limit: int,
        rng: np.random.Generator,
    ) -> _IsolationNode:
        n_rows = X_sub.shape[0]
        if depth >= height_limit or n_rows <= 1:
            return _IsolationNode(size=n_rows)

        col_mins = np.min(X_sub, axis=0)
        col_maxs = np.max(X_sub, axis=0)
        valid_features = np.flatnonzero((col_maxs - col_mins) > 1e-12)
        if valid_features.size == 0:
            return _IsolationNode(size=n_rows)

        feat = int(rng.choice(valid_features))
        fmin = float(col_mins[feat])
        fmax = float(col_maxs[feat])
        split = float(rng.uniform(fmin, fmax))

        left_mask = X_sub[:, feat] < split
        if not np.any(left_mask) or np.all(left_mask):
            # Fallback to midpoint when uniform draw hits exact boundary
            split = 0.5 * (fmin + fmax)
            left_mask = X_sub[:, feat] < split
            if not np.any(left_mask) or np.all(left_mask):
                return _IsolationNode(size=n_rows)

        right_mask = ~left_mask
        return _IsolationNode(
            size=n_rows,
            feature_idx=feat,
            split_value=split,
            left=self._build_tree(X_sub[left_mask], depth + 1, height_limit, rng),
            right=self._build_tree(X_sub[right_mask], depth + 1, height_limit, rng),
        )

    def _path_length(self, row: np.ndarray, node: _IsolationNode, depth: int = 0) -> float:
        cur = node
        cur_depth = depth
        while not cur.is_leaf:
            assert cur.feature_idx is not None and cur.split_value is not None
            if row[cur.feature_idx] < cur.split_value:
                assert cur.left is not None
                cur = cur.left
            else:
                assert cur.right is not None
                cur = cur.right
            cur_depth += 1
        return float(cur_depth) + _c_factor(cur.size)

    def fit(self, X: np.ndarray) -> PurePythonIsolationForest:
        """Fit isolation trees and compute the contamination threshold."""
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array.")

        n_samples = X_arr.shape[0]
        self.trees_ = []
        if n_samples == 0:
            self.sample_size_ = 0
            self.c_psi_ = 0.0
            self.threshold_ = 1.0
            return self

        self.sample_size_ = min(self.max_samples, n_samples)
        self.c_psi_ = _c_factor(self.sample_size_)
        height_limit = max(1, math.ceil(math.log2(max(self.sample_size_, 2))))

        rng = np.random.default_rng(self.random_state)
        for _ in range(self.n_estimators):
            if self.sample_size_ < n_samples:
                idx = rng.choice(n_samples, size=self.sample_size_, replace=False)
                X_sub = X_arr[idx]
            else:
                X_sub = X_arr
            self.trees_.append(self._build_tree(X_sub, 0, height_limit, rng))

        train_scores = self.anomaly_scores(X_arr)
        self.threshold_ = float(np.quantile(train_scores, 1.0 - self.contamination))
        return self

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute Isolation Forest anomaly scores in (0, 1]; higher = more anomalous."""
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if X_arr.shape[0] == 0:
            return np.empty(0, dtype=np.float64)
        if not self.trees_ or self.c_psi_ <= 0.0:
            return np.full(X_arr.shape[0], 0.5, dtype=np.float64)

        scores = np.empty(X_arr.shape[0], dtype=np.float64)
        n_trees = float(len(self.trees_))
        for i, row in enumerate(X_arr):
            mean_path = sum(self._path_length(row, tree) for tree in self.trees_) / n_trees
            scores[i] = 2.0 ** (-mean_path / self.c_psi_)
        return scores

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return negated anomaly scores in [-1, 0) (lower = more anomalous)."""
        return -self.anomaly_scores(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return signed inlier margin (negative = anomaly, >= 0 = normal)."""
        return self.threshold_ - self.anomaly_scores(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return -1 for anomalies and 1 for normal observations."""
        scores = self.anomaly_scores(X)
        if scores.size == 0:
            return np.empty(0, dtype=np.int64)
        return np.where(scores > self.threshold_, -1, 1).astype(np.int64)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit on X and return -1 (anomaly) or 1 (normal) predictions."""
        return self.fit(X).predict(X)
