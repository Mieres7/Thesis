"""
eval/models/mlp_probe.py

Frozen-encoder MLP probe on patient-level embeddings. Minority classes can be
over-sampled because MLPClassifier does not support sample_weight in older
scikit-learn versions.
"""
from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample


def _balance_classes(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Oversample every observed class to the size of the largest class."""
    classes, counts = np.unique(y, return_counts=True)
    max_count = int(counts.max())
    balanced_X, balanced_y = [], []
    rng = np.random.RandomState(seed)

    for cls in classes:
        X_cls = X[y == cls]
        if len(X_cls) < max_count:
            X_cls = resample(
                X_cls,
                n_samples=max_count,
                random_state=rng,
                replace=True,
            )
        balanced_X.append(X_cls)
        balanced_y.append(np.full(max_count, cls, dtype=y.dtype))

    return np.vstack(balanced_X), np.concatenate(balanced_y)


def fit_mlp_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = 42,
    hidden_layer_sizes: tuple[int, ...] = (256,),
    alpha: float = 1e-4,
    max_iter: int = 500,
    early_stopping: bool = False,
    validation_fraction: float = 0.1,
    balanced: bool = True,
) -> MLPClassifier:
    """Fit an MLP probe.

    `early_stopping` is deliberately opt-in. The evaluation pipeline enables
    it only for `eval_method_reason == "small_n"`; large datasets preserve
    the former full-training behavior.
    """
    if balanced:
        X_train, y_train = _balance_classes(X_train, y_train, seed)

    clf = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        alpha=alpha,
        max_iter=max_iter,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        random_state=seed,
    )
    clf.fit(X_train, y_train)
    return clf