
from pathlib import Path
import csv
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn

from src.models.hybrid_crg_sleep import HybridCRGSleep


DATA_DIR = Path("data/processed/isruc_v2")
PRETRAINED = Path("models/hybrid_crg_sleep_best.pt")
OUTPUT_DIR = Path("models/isruc_training")

EPOCHS = 12
BATCH_SIZE = 16
LEARNING_RATE = 2e-4
PATIENCE = 4
SEED = 42

CLASSES = ["Wake", "N1", "N2", "N3", "REM"]


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def prepare_split():
    """Split by person, never by recording."""

    files = sorted(DATA_DIR.glob("*.npz"))

    if len(files) != 126:
        raise RuntimeError(
            f"Expected 126 recordings, found {len(files)}"
        )

    subjects = {}

    for file in files:
        with np.load(file, allow_pickle=False) as data:
            subject = str(data["subject_id"].item())

        subjects.setdefault(subject, []).append(file)

    if len(subjects) != 118:
        raise RuntimeError(
            f"Expected 118 subjects, found {len(subjects)}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    split_file = OUTPUT_DIR / "subject_split.json"

    if split_file.exists():
        split = json.loads(
            split_file.read_text(encoding="utf-8")
        )
        print("Using previously saved subject split.")
    else:
        ids = sorted(subjects)
        rng = np.random.default_rng(SEED)
        rng.shuffle(ids)

        n = len(ids)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        split = {
            "train": ids[:train_end],
            "val": ids[train_end:val_end],
            "test": ids[val_end:],
            "seed": SEED,
        }

        split_file.write_text(
            json.dumps(split, indent=2),
            encoding="utf-8",
        )

    train_ids = set(split["train"])
    val_ids = set(split["val"])
    test_ids = set(split["test"])

    assert not (train_ids & val_ids)
    assert not (train_ids & test_ids)
    assert not (val_ids & test_ids)

    assert train_ids | val_ids | test_ids == set(subjects)

    def select(ids):
        return [
            file
            for subject in sorted(ids)
            for file in subjects[subject]
        ]

    print(
        f"Subjects: train={len(train_ids)}, "
        f"val={len(val_ids)}, test={len(test_ids)}"
    )

    return (
        select(train_ids),
        select(val_ids),
        select(test_ids),
    )


def load_recording(file):
    with np.load(file, allow_pickle=False) as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int64)

    if X.ndim != 3 or X.shape[1:] != (2, 3000):
        raise ValueError(
            f"{file.name}: unexpected shape {X.shape}"
        )

    if len(X) != len(y):
        raise ValueError(
            f"{file.name}: X/y length mismatch"
        )

    if not np.isfinite(X).all():
        raise ValueError(
            f"{file.name}: invalid EEG samples"
        )

    if not np.isin(y, range(5)).all():
        raise ValueError(
            f"{file.name}: unexpected labels"
        )

    return X, y


def get_class_weights(train_files, device):
    counts = np.zeros(5, dtype=np.int64)

    for file in train_files:
        with np.load(file, allow_pickle=False) as data:
            counts += np.bincount(
                data["y"],
                minlength=5,
            )

    if np.any(counts == 0):
        raise RuntimeError(
            f"Missing training classes: {counts}"
        )

    # Moderate weighting to avoid overly aggressive
    # compensation for uncommon stages.
    weights = 1.0 / np.sqrt(counts)
    weights /= weights.mean()

    print("Training class counts:", counts.tolist())
    print("Class weights:", weights.round(3).tolist())

    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=device,
    )


def load_pretrained(device):
    checkpoint = torch.load(
        PRETRAINED,
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

    print("Pretrained Sleep-EDF weights loaded.")

    return model


def calculate_metrics(confusion):
    tp = np.diag(confusion)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)

    precision = np.divide(
        tp,
        predicted,
        out=np.zeros(5, dtype=float),
        where=predicted > 0,
    )

    recall = np.divide(
        tp,
        support,
        out=np.zeros(5, dtype=float),
        where=support > 0,
    )

    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(5, dtype=float),
        where=precision + recall > 0,
    )

    accuracy = tp.sum() / max(confusion.sum(), 1)

    return float(accuracy), float(f1.mean()), f1


def evaluate(model, files, device):
    model.eval()
    confusion = np.zeros((5, 5), dtype=np.int64)

    with torch.inference_mode():
        for file in files:
            X, y = load_recording(file)

            for start in range(0, len(y), BATCH_SIZE):
                stop = start + BATCH_SIZE

                batch = torch.from_numpy(
                    X[start:stop]
                ).to(device)

                logits = model(batch)

                predictions = (
                    logits.argmax(dim=1)
                    .cpu()
                    .numpy()
                )

                np.add.at(
                    confusion,
                    (y[start:stop], predictions),
                    1,
                )

    return calculate_metrics(confusion), confusion


def train_epoch(model, files, optimizer, criterion, device):
    model.train()
    shuffled_files = list(files)
    random.shuffle(shuffled_files)

    total_loss = 0.0
    total_samples = 0

    for file in shuffled_files:
        X, y = load_recording(file)

        indices = np.random.permutation(len(y))

        for start in range(0, len(indices), BATCH_SIZE):
            idx = indices[start:start + BATCH_SIZE]

            batch_x = torch.from_numpy(
                X[idx].copy()
            ).to(device)

            batch_y = torch.from_numpy(
                y[idx].copy()
            ).to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(batch_x)

            loss = criterion(logits, batch_y)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss in {file.name}"
                )

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            total_loss += loss.item() * len(idx)
            total_samples += len(idx)

    return total_loss / max(total_samples, 1)


def main():
    set_seed()

    if not PRETRAINED.exists():
        raise FileNotFoundError(PRETRAINED)

    train_files, val_files, test_files = prepare_split()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    print(
        "Recordings:",
        len(train_files),
        len(val_files),
        len(test_files),
    )

    model = load_pretrained(device)

    criterion = nn.CrossEntropyLoss(
        weight=get_class_weights(train_files, device)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-3,
    )

    best_f1 = -1.0
    bad_epochs = 0
    best_path = OUTPUT_DIR / "hybrid_isruc_best.pt"

    history_file = OUTPUT_DIR / "history.csv"

    with history_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "epoch",
            "train_loss",
            "val_accuracy",
            "val_macro_f1",
            "seconds",
        ])

        for epoch in range(1, EPOCHS + 1):
            started = time.monotonic()

            print(
                f"\n=== EPOCH {epoch}/{EPOCHS} ===",
                flush=True,
            )

            loss = train_epoch(
                model,
                train_files,
                optimizer,
                criterion,
                device,
            )

            metrics, _ = evaluate(
                model,
                val_files,
                device,
            )

            accuracy, macro_f1, _ = metrics
            elapsed = time.monotonic() - started

            print(
                f"Loss={loss:.4f} | "
                f"Val Accuracy={accuracy:.4f} | "
                f"Val Macro F1={macro_f1:.4f} | "
                f"Time={elapsed / 60:.1f} min",
                flush=True,
            )

            writer.writerow([
                epoch,
                loss,
                accuracy,
                macro_f1,
                round(elapsed, 1),
            ])
            handle.flush()

            if macro_f1 > best_f1:
                best_f1 = macro_f1
                bad_epochs = 0

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "val_macro_f1": best_f1,
                        "seed": SEED,
                    },
                    best_path,
                )

                print(
                    "BEST MODEL SAVED",
                    flush=True,
                )
            else:
                bad_epochs += 1

                if bad_epochs >= PATIENCE:
                    print(
                        "Early stopping.",
                        flush=True,
                    )
                    break

    print("\n=== FINAL HELD-OUT TEST ===", flush=True)

    checkpoint = torch.load(
        best_path,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    metrics, confusion = evaluate(
        model,
        test_files,
        device,
    )

    accuracy, macro_f1, per_class_f1 = metrics

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Macro F1: {macro_f1:.4f}")

    for name, score in zip(CLASSES, per_class_f1):
        print(f"{name}: F1={score:.4f}")

    np.savetxt(
        OUTPUT_DIR / "test_confusion.csv",
        confusion,
        fmt="%d",
        delimiter=",",
    )

    print(
        f"\nFINISHED. Model: {best_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()