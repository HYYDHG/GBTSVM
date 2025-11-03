"""Utility functions for computing simple feature attention weights.

The original GBTSVM implementation operates on granular ball features that are
derived from the raw tabular data.  To inject an attention mechanism without
re-architecting the optimisation procedure, we compute per-feature weights that
highlight informative dimensions before the granular-ball construction step.

The attention mechanism implemented here is intentionally lightweight:

* For binary problems we compare the class-wise feature means and assign higher
  weights to dimensions whose means differ the most.
* For multi-class problems we measure how far each class mean deviates from the
  global mean.

The resulting scores are normalised with a temperature-controlled softmax and
rescaled so that the average weight equals 1.  This keeps the feature scale
compatible with the optimisation routine while still emphasising informative
dimensions.
"""

from __future__ import annotations

import numpy as np


def _safe_reshape(features: np.ndarray) -> np.ndarray:
    """Ensure that the feature array is two-dimensional.

    The downstream logic expects a 2-D array with shape ``(n_samples,
    n_features)``.  When the upstream pipeline provides a single feature the
    slicing operation can result in a 1-D array; reshaping keeps the code
    consistent without altering the data.
    """

    features = np.asarray(features, dtype=float)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    return features


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    """Numerically stable softmax with temperature control."""

    values = np.asarray(values, dtype=float)
    # Prevent division-by-zero and negative temperature values.
    temperature = max(float(temperature), 1e-8)
    shifted = values - np.max(values)
    exps = np.exp(shifted / temperature)
    return exps / np.sum(exps)


def compute_feature_attention(
    data: np.ndarray,
    *,
    label_column: int = -1,
    temperature: float = 1.0,
) -> np.ndarray:
    """Compute per-feature attention weights for tabular data.

    Parameters
    ----------
    data:
        Array where the last column corresponds to labels and the remaining
        columns are features.
    label_column:
        Index of the label column.  By default the last column is assumed to be
        the label.
    temperature:
        Temperature factor for the softmax normalisation.  Smaller values make
        the distribution peakier; higher values smooth it out.

    Returns
    -------
    np.ndarray
        One-dimensional array containing a positive attention weight for each
        feature.  The weights have unit mean so that the overall feature scale
        stays comparable to the original data.
    """

    features = _safe_reshape(data[:, :label_column])
    labels = np.asarray(data[:, label_column])

    unique_labels = np.unique(labels)
    if unique_labels.size <= 1:
        # Degenerate case: fall back to uniform attention.
        return np.ones(features.shape[1], dtype=float)

    class_means = []
    for label in unique_labels:
        class_features = features[labels == label]
        if class_features.size == 0:
            continue
        class_means.append(class_features.mean(axis=0))

    if not class_means:
        return np.ones(features.shape[1], dtype=float)

    class_means = np.vstack(class_means)

    if class_means.shape[0] == 2:
        # Binary case – emphasise features with large inter-class separation.
        raw_scores = np.abs(class_means[0] - class_means[1])
    else:
        # Multi-class case – emphasise deviation from the global mean.
        global_mean = features.mean(axis=0)
        raw_scores = np.sqrt(((class_means - global_mean) ** 2).mean(axis=0))

    attention = _softmax(raw_scores, temperature)

    # Rescale to keep the average weight at 1.0 – this avoids shrinking or
    # exploding the feature magnitudes while still highlighting important
    # dimensions.
    mean_value = attention.mean()
    if mean_value == 0:
        return np.ones_like(attention)
    return attention / mean_value


__all__ = ["compute_feature_attention"]

