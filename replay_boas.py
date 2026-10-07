"""Offline replay of prepared BOAS epochs; no hardware, audio or training."""
import argparse
from pathlib import Path

import numpy as np

from src.data.boas_split import validate_participant_split
from src.models.hybrid_sleep_model import HybridSleepModel, LABELS

ROOT = Path(__file__).resolve().parent


def load_recording(path):
    with np.load(path, allow_pickle=False) as data:
        x, y, onsets = data["X"], data["y"], data["onsets"]
        fs = float(data["sampling_frequency"].item())
        recording = str(data["subject_id"].item())
    if fs != 100 or recording != path.stem:
        raise ValueError("Recording identity or sampling frequency does not match BOAS input")
    if x.ndim != 3 or x.shape[1:] != (2, 3000) or len(x) == 0:
        raise ValueError("Expected nonempty recording (N, 2, 3000)")
    if y.shape != (len(x),) or onsets.shape != (len(x),):
        raise ValueError("Epoch/label/onset counts differ")
    if not np.issubdtype(y.dtype, np.integer) or not ((y >= 0) & (y < 5)).all():
        raise ValueError("Invalid sleep labels")
    if not np.isfinite(onsets).all() or np.any(onsets < 0) or np.any(np.diff(onsets) < 30 - 1e-6):
        raise ValueError("Invalid or overlapping epoch onsets")
    return x, y, onsets


def metrics(y, predicted):
    confusion = np.zeros((5, 5), dtype=np.int64)
    np.add.at(confusion, (y, predicted), 1)
    denominator = confusion.sum(0) + confusion.sum(1)
    f1 = np.divide(2 * np.diag(confusion), denominator,
                   out=np.zeros(5, dtype=float), where=denominator > 0)
    return float(np.trace(confusion) / len(y)), f1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/boas_training_pid/hybrid_boas_best.pt")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed/boas")
    parser.add_argument("--recording", help="Recording ID, e.g. sub-101; default: first validation recording")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--show", type=int, default=20, help="Number of rows to display; all epochs are evaluated")
    args = parser.parse_args()
    if args.batch_size < 1 or args.show < 0:
        parser.error("batch-size must be positive; show must be nonnegative")
    model = HybridSleepModel(args.checkpoint, args.device)
    split = model.metadata.get("participant_split", {})
    validate_participant_split(split)
    recording = args.recording or split["recordings"]["val"][0]
    groups = [g for g, entries in split["recordings"].items() if recording in entries]
    if len(groups) != 1:
        parser.error("Recording must belong to this checkpoint's saved participant split")
    # Recording IDs are selected from the saved manifest, never used as arbitrary paths.
    if Path(recording).name != recording or not recording.startswith("sub-"):
        parser.error("Invalid recording ID")
    x, y, onsets = load_recording(args.data_dir / f"{recording}.npz")
    print(f"Checkpoint epoch: {model.metadata.get('epoch')} | Device: {model.device}")
    print(f"Recording: {recording} | Group: {groups[0]} | Epochs: {len(y)}")
    print("Offline BOAS replay: retained 30-second epochs; gaps are preserved in onset times.")
    print("Onset(s) | True | Predicted | Max model probability")
    predictions = []
    for start in range(0, len(x), args.batch_size):
        probabilities = model.predict_batch(x[start:start + args.batch_size])
        predicted = probabilities.argmax(1)
        predictions.extend(predicted.tolist())
        for j, label in enumerate(predicted):
            i = start + j
            if i < args.show:
                print(f"{onsets[i]:8.1f} | {LABELS[y[i]]:4} | {LABELS[label]:9} | {probabilities[j,label]:.3f}")
    accuracy, f1 = metrics(y, np.asarray(predictions))
    print(f"Recording accuracy: {accuracy:.4f} | Macro-F1 (5 classes): {f1.mean():.4f}")
    print("Stage | True epoch count | F1")
    for k, label in enumerate(LABELS):
        print(f"{label:5} | {int((y == k).sum()):16} | {f1[k]:.4f}")
    print("Single-recording result; not a replacement for the full held-out test score.")


if __name__ == "__main__":
    main()
