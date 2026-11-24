"""
eval/models/linear_probe.py

Frozen-encoder linear probing classifier: trains a LogisticRegression on
pre-extracted patient-level embeddings.

class_weight="balanced" handles class imbalance automatically (inverse
frequency weighting) without manual resampling -- consistent across binary
(HER2) and multiclass (molecular subtype) tasks via sklearn's built-in
multinomial handling.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def stack_embeddings(df, embedding_col: str = "embedding") -> np.ndarray:
    """Stack a column of (D,) arrays into a single (N, D) matrix."""
    return np.stack(df[embedding_col].values)


def fit_linear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = 42,
    C: float = 1.0,
    max_iter: int = 2000,
) -> LogisticRegression:
    clf = LogisticRegression(
        class_weight="balanced",
        C=C,
        max_iter=max_iter,
        random_state=seed,
    )
    clf.fit(X_train, y_train)
    return clf
