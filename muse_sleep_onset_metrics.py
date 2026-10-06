"""Regression metrics in seconds; no training or weighted competition score."""
import numpy as np

BINS = ((0, 40), (40, 90), (90, 300), (300, 600))


def _validated(y_true, y_pred):
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.ndim != 1 or true.shape != pred.shape or not true.size:
        raise ValueError("Expected nonempty, aligned one-dimensional arrays")
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("Metrics require finite values")
    return true, pred


def mae_seconds(y_true, y_pred):
    true, pred = _validated(y_true, y_pred)
    return float(np.mean(np.abs(true - pred)))


def binned_mae_seconds(y_true, y_pred):
    """Bins use true targets: [low, high), with 600 included in the last bin.

    Empty bins return count=0 and mae_seconds=None, never a misleading zero.
    Predictions are not clipped by this evaluation utility.
    """
    true, pred = _validated(y_true, y_pred)
    if np.any((true < 0) | (true > 600)):
        raise ValueError("True targets must be in [0, 600]")
    result = {}
    for low, high in BINS:
        mask = (true >= low) & ((true <= high) if high == 600 else (true < high))
        result[f"{low}-{high}"] = {
            "count": int(mask.sum()),
            "mae_seconds": float(np.mean(np.abs(true[mask] - pred[mask]))) if mask.any() else None,
        }
    return result
