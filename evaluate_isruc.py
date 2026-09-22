
from pathlib import Path
import argparse
import csv

import numpy as np
import torch

from src.models.hybrid_crg_sleep import HybridCRGSleep


DATA_DIR = Path("data/processed/isruc_v2")
MODEL_PATH = Path("models/hybrid_crg_sleep_best.pt")
RESULTS_DIR = Path("exports/isruc_evaluation")

CLASSES = ["Wake", "N1", "N2", "N3", "REM"]


def load_model(device):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    # Support both a plain state_dict and common
    # training-checkpoint formats.
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        raise ValueError("Unexpected checkpoint format")

    model = HybridCRGSleep(num_classes=5)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    print(f"Model loaded: {MODEL_PATH}")
    print(f"Device: {device}")

    return model


def evaluate_recording(file, model, device, batch_size):
    with np.load(file, allow_pickle=False) as data:
        X = data["X"]
        y = data["y"]
        subject = str(data["subject_id"].item())
        scheme = str(data["channel_scheme"].item())

    if X.ndim != 3 or X.shape[1:] != (2, 3000):
        raise ValueError(f"Unexpected EEG shape: {X.shape}")

    if len(X) != len(y):
        raise ValueError("EEG and labels have different lengths")

    if not np.isfinite(X).all():
        raise ValueError("EEG contains NaN or Infinity")

    if not np.isin(y, np.arange(5)).all():
        raise ValueError("Unexpected sleep-stage labels")

    confusion = np.zeros((5, 5), dtype=np.int64)

    with torch.inference_mode():
        for start in range(0, len(y), batch_size):
            stop = start + batch_size

            batch = torch.from_numpy(
                X[start:stop].astype(np.float32, copy=False)
            ).to(device)

            logits = model(batch)
            predictions = logits.argmax(dim=1).cpu().numpy()

            np.add.at(
                confusion,
                (y[start:stop], predictions),
                1,
            )

    return subject, scheme, confusion


def calculate_metrics(confusion):
    tp = np.diag(confusion).astype(float)
    support = confusion.sum(axis=1).astype(float)
    predicted = confusion.sum(axis=0).astype(float)

    precision = np.divide(
        tp, predicted,
        out=np.zeros(5),
        where=predicted > 0,
    )

    recall = np.divide(
        tp, support,
        out=np.zeros(5),
        where=support > 0,
    )

    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(5),
        where=(precision + recall) > 0,
    )

    accuracy = (
        tp.sum() / confusion.sum()
        if confusion.sum() else 0.0
    )

    # Report five-class Macro F1 only when
    # all five classes are present.
    macro_f1 = (
        float(f1.mean())
        if np.all(support > 0)
        else float("nan")
    )

    return accuracy, macro_f1, f1, support


def save_confusion(path, confusion):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Actual / Predicted", *CLASSES])

        for name, row in zip(CLASSES, confusion):
            writer.writerow([name, *row.tolist()])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    files = sorted(DATA_DIR.glob("*.npz"))

    if args.limit is not None:
        files = files[:args.limit]

    if not files:
        raise FileNotFoundError("No prepared ISRUC recordings")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = load_model(device)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    total = np.zeros((5, 5), dtype=np.int64)
    by_scheme = {}
    failures = []

    for index, file in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {file.name}", flush=True)

        try:
            subject, scheme, confusion = evaluate_recording(
                file, model, device, args.batch_size
            )

            total += confusion

            if scheme not in by_scheme:
                by_scheme[scheme] = np.zeros(
                    (5, 5), dtype=np.int64
                )

            by_scheme[scheme] += confusion

        except Exception as exc:
            failures.append((file.name, str(exc)))
            print(f"FAILED: {exc}", flush=True)

    print("\n=== OVERALL RESULTS ===")

    accuracy, macro_f1, f1, support = calculate_metrics(total)

    print(f"Evaluated epochs: {total.sum()}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    for name, score, count in zip(CLASSES, f1, support):
        print(f"{name}: F1={score:.4f}, support={int(count)}")

    save_confusion(
        RESULTS_DIR / "overall_confusion.csv",
        total,
    )

    print("\n=== BY ELECTRODE SCHEME ===")

    for scheme, confusion in sorted(by_scheme.items()):
        acc, macro, _, _ = calculate_metrics(confusion)

        print(
            f"{scheme}: "
            f"epochs={confusion.sum()} "
            f"accuracy={acc:.4f} "
            f"macro_f1={macro:.4f}"
        )

        save_confusion(
            RESULTS_DIR / f"confusion_{scheme}.csv",
            confusion,
        )

    print(f"\nFailed recordings: {len(failures)}")
    for file, error in failures:
        print(f"{file}: {error}")

    if failures:
        with (RESULTS_DIR / "failures.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["file", "error"])
            writer.writerows(failures)


if __name__ == "__main__":
    main()