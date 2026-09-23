
from pathlib import Path
import json
import csv

import numpy as np
import torch

from src.models.hybrid_crg_sleep import HybridCRGSleep


DATA_DIR = Path("data/processed/isruc_v2")
SPLIT_FILE = Path("models/isruc_training/subject_split.json")

MODELS = {
    "SleepEDF_original": Path(
        "models/hybrid_crg_sleep_best.pt"
    ),
    "ISRUC_finetuned": Path(
        "models/isruc_training/hybrid_isruc_best.pt"
    ),
}

OUTPUT = Path("exports/model_comparison")
CLASSES = ["Wake", "N1", "N2", "N3", "REM"]
BATCH_SIZE = 32


def load_model(path, device):
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if "model_state_dict" in checkpoint:
        weights = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        weights = checkpoint["state_dict"]
    elif "model" in checkpoint:
        weights = checkpoint["model"]
    else:
        weights = checkpoint

    model = HybridCRGSleep(num_classes=5)
    model.load_state_dict(weights, strict=True)
    model.to(device)
    model.eval()
    return model


def get_test_files():
    split = json.loads(
        SPLIT_FILE.read_text(encoding="utf-8")
    )

    train = set(split["train"])
    val = set(split["val"])
    test = set(split["test"])

    if train & val or train & test or val & test:
        raise RuntimeError("Subject leakage detected")

    files = []
    for path in sorted(DATA_DIR.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            subject = str(data["subject_id"].item())

        if subject in test:
            files.append(path)

    if not files:
        raise RuntimeError("No test recordings found")

    print(f"Test subjects: {len(test)}")
    print(f"Test recordings: {len(files)}")
    return files


def metrics(cm):
    true_positive = np.diag(cm).astype(float)
    actual = cm.sum(axis=1)
    predicted = cm.sum(axis=0)

    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros(5),
        where=predicted > 0,
    )

    recall = np.divide(
        true_positive,
        actual,
        out=np.zeros(5),
        where=actual > 0,
    )

    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(5),
        where=precision + recall > 0,
    )

    accuracy = (
        true_positive.sum() / cm.sum()
        if cm.sum() else 0.0
    )

    return accuracy, f1.mean(), f1


def evaluate(model, files, device):
    cm = np.zeros((5, 5), dtype=np.int64)
    scheme_matrices = {}

    with torch.inference_mode():
        for index, path in enumerate(files, 1):
            with np.load(path, allow_pickle=False) as data:
                X = data["X"]
                y = data["y"]
                scheme = str(
                    data["channel_scheme"].item()
                )

            if X.shape[1:] != (2, 3000):
                raise ValueError(
                    f"{path.name}: invalid EEG shape"
                )

            if len(X) != len(y):
                raise ValueError(
                    f"{path.name}: X/y mismatch"
                )

            if not np.isfinite(X).all():
                raise ValueError(
                    f"{path.name}: invalid EEG"
                )

            if not np.isin(y, np.arange(5)).all():
                raise ValueError(
                    f"{path.name}: invalid labels"
                )

            if scheme not in scheme_matrices:
                scheme_matrices[scheme] = np.zeros(
                    (5, 5), dtype=np.int64
                )

            for start in range(0, len(y), BATCH_SIZE):
                stop = start + BATCH_SIZE

                batch = torch.from_numpy(
                    X[start:stop].astype(
                        np.float32,
                        copy=False,
                    )
                ).to(device)

                prediction = (
                    model(batch)
                    .argmax(dim=1)
                    .cpu()
                    .numpy()
                )

                actual = y[start:stop]

                np.add.at(
                    cm,
                    (actual, prediction),
                    1,
                )

                np.add.at(
                    scheme_matrices[scheme],
                    (actual, prediction),
                    1,
                )

            print(
                f"[{index}/{len(files)}] {path.name}",
                flush=True,
            )

    return cm, scheme_matrices


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    files = get_test_files()
    summary = []

    for model_name, model_path in MODELS.items():
        print(f"\n=== {model_name} ===", flush=True)

        model = load_model(model_path, device)
        cm, schemes = evaluate(model, files, device)

        accuracy, macro_f1, per_class = metrics(cm)

        print(f"Test epochs: {cm.sum()}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")

        for name, score in zip(CLASSES, per_class):
            print(f"{name}: F1={score:.4f}")

        np.savetxt(
            OUTPUT / f"{model_name}_confusion.csv",
            cm,
            fmt="%d",
            delimiter=",",
        )

        row = {
            "model": model_name,
            "scheme": "ALL",
            "epochs": int(cm.sum()),
            "accuracy": accuracy,
            "macro_f1": macro_f1,
        }

        summary.append(row)

        for scheme, scheme_cm in sorted(
            schemes.items()
        ):
            acc, macro, _ = metrics(scheme_cm)

            print(
                f"{scheme}: "
                f"epochs={scheme_cm.sum()}, "
                f"accuracy={acc:.4f}, "
                f"macro_f1={macro:.4f}"
            )

            summary.append({
                "model": model_name,
                "scheme": scheme,
                "epochs": int(scheme_cm.sum()),
                "accuracy": acc,
                "macro_f1": macro,
            })

            np.savetxt(
                OUTPUT / (
                    f"{model_name}_{scheme}_confusion.csv"
                ),
                scheme_cm,
                fmt="%d",
                delimiter=",",
            )

    with (
        OUTPUT / "comparison.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "scheme",
                "epochs",
                "accuracy",
                "macro_f1",
            ],
        )
        writer.writeheader()
        writer.writerows(summary)

    print("\nComparison saved:", OUTPUT)


if __name__ == "__main__":
    main()
