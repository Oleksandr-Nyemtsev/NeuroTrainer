from pathlib import Path
import csv
import json
import random
import time

import numpy as np
import torch
from src.models.hybrid_crg_sleep import HybridCRGSleep


DATA = Path("data/processed/boas")
OUT = Path("models/boas_training")
PRETRAINED = Path("models/isruc_training/hybrid_isruc_best.pt")

SEED = 42
EPOCHS = 12
BATCH_SIZE = 4
LR = 0.0001
PATIENCE = 4
N_CLASSES = 5

OUT.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable. GPU training required.")

device = torch.device("cuda")
print("GPU:", torch.cuda.get_device_name(0), flush=True)


# Independent participant split
files = sorted(DATA.glob("sub-*.npz"))
subjects = {}

for path in files:
    with np.load(path, allow_pickle=False) as d:
        sid = str(d["subject_id"].item())
    subjects.setdefault(sid, []).append(path)

if len(subjects) < 20:
    raise RuntimeError(
        f"Too few participants: {len(subjects)}"
    )

split_path = OUT / "subject_split.json"

if split_path.exists():
    split = json.loads(split_path.read_text())
else:
    ids = sorted(subjects)
    random.shuffle(ids)

    n1 = int(len(ids) * 0.70)
    n2 = int(len(ids) * 0.85)

    split = {
        "train": ids[:n1],
        "val": ids[n1:n2],
        "test": ids[n2:],
        "seed": SEED,
    }

    split_path.write_text(json.dumps(split, indent=2))

sets = [set(split[k]) for k in ("train", "val", "test")]

if any(sets[i] & sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
    raise RuntimeError("Participant leakage!")

if set.union(*sets) != set(subjects):
    raise RuntimeError("Subject split does not match dataset")

def select(name):
    return [
        f
        for sid in split[name]
        for f in subjects[sid]
    ]

train_files = select("train")
val_files = select("val")
test_files = select("test")

print(
    "Participants:",
    {k: len(split[k]) for k in ("train", "val", "test")},
    flush=True,
)


# Load previous trained weights
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
model = model.to(device)

print("ISRUC pretrained weights loaded", flush=True)


# Class weights based ONLY on training participants
counts = np.zeros(N_CLASSES, dtype=np.int64)

for path in train_files:
    with np.load(path, allow_pickle=False) as d:
        counts += np.bincount(
            d["y"], minlength=N_CLASSES
        )[:N_CLASSES]

if np.any(counts == 0):
    raise RuntimeError(f"Missing training classes: {counts}")

class_weights = 1 / np.sqrt(counts)
class_weights /= class_weights.mean()

print("Training counts:", counts.tolist(), flush=True)
print("Class weights:", class_weights.round(3), flush=True)

criterion = torch.nn.CrossEntropyLoss(
    weight=torch.tensor(
        class_weights, dtype=torch.float32, device=device
    )
)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=LR, weight_decay=0.01
)


def load_file(path):
    with np.load(path, allow_pickle=False) as d:
        X = d["X"].astype(np.float32)
        y = d["y"].astype(np.int64)

    if X.ndim != 3 or X.shape[1:] != (2, 3000):
        raise ValueError(f"Invalid shape: {path}: {X.shape}")

    if len(X) != len(y) or not np.isfinite(X).all():
        raise ValueError(f"Invalid recording: {path}")

    return X, y


def run_files(paths, training):
    model.train(training)

    order = list(paths)
    if training:
        random.shuffle(order)

    confusion = np.zeros(
        (N_CLASSES, N_CLASSES), dtype=np.int64
    )

    total_loss = 0.0
    total_samples = 0

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for file_no, path in enumerate(order, 1):
            X, y = load_file(path)

            indices = np.arange(len(y))
            if training:
                np.random.shuffle(indices)

            for start in range(0, len(indices), BATCH_SIZE):
                idx = indices[start:start + BATCH_SIZE]

                xb = torch.from_numpy(
                    X[idx].copy()
                ).to(device)

                yb = torch.from_numpy(
                    y[idx].copy()
                ).to(device)

                if training:
                    optimizer.zero_grad(set_to_none=True)

                logits = model(xb)
                loss = criterion(logits, yb)

                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 1.0
                    )
                    optimizer.step()

                pred = logits.argmax(dim=1)

                true_np = yb.cpu().numpy()
                pred_np = pred.cpu().numpy()

                np.add.at(
                    confusion,
                    (true_np, pred_np),
                    1,
                )

                total_loss += loss.item() * len(idx)
                total_samples += len(idx)

            del X, y

            if file_no % 10 == 0:
                print(
                    f"  {'TRAIN' if training else 'EVAL'}: "
                    f"{file_no}/{len(order)} files",
                    flush=True,
                )

    accuracy = (
        np.trace(confusion) / max(confusion.sum(), 1)
    )

    tp = np.diag(confusion).astype(float)
    actual = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)

    precision = np.divide(
        tp, predicted,
        out=np.zeros(N_CLASSES),
        where=predicted > 0
    )
    recall = np.divide(
        tp, actual,
        out=np.zeros(N_CLASSES),
        where=actual > 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(N_CLASSES),
        where=(precision + recall) > 0
    )

    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": float(accuracy),
        "macro_f1": float(f1.mean()),
        "class_f1": f1.tolist(),
        "confusion": confusion.tolist(),
        "samples": int(total_samples),
    }


best_f1 = -1
bad_epochs = 0
history_path = OUT / "history.csv"
best_path = OUT / "hybrid_boas_best.pt"

with history_path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "epoch", "train_loss", "train_accuracy",
        "train_macro_f1", "val_loss",
        "val_accuracy", "val_macro_f1"
    ])

start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    print(f"\n========== EPOCH {epoch}/{EPOCHS} ==========",
          flush=True)

    train = run_files(train_files, training=True)
    val = run_files(val_files, training=False)

    print(
        f"Train: acc={train['accuracy']:.4f}, "
        f"F1={train['macro_f1']:.4f}",
        flush=True,
    )
    print(
        f"Val: acc={val['accuracy']:.4f}, "
        f"F1={val['macro_f1']:.4f}",
        flush=True,
    )

    with history_path.open("a", newline="") as f:
        csv.writer(f).writerow([
            epoch,
            train["loss"],
            train["accuracy"],
            train["macro_f1"],
            val["loss"],
            val["accuracy"],
            val["macro_f1"],
        ])

    if val["macro_f1"] > best_f1:
        best_f1 = val["macro_f1"]
        bad_epochs = 0

        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "val_macro_f1": best_f1,
            "class_names": ["Wake", "N1", "N2", "N3", "REM"],
        }, best_path)

        (OUT / "best_validation.json").write_text(
            json.dumps(val, indent=2)
        )

        print("NEW BEST MODEL SAVED", flush=True)
    else:
        bad_epochs += 1

    if bad_epochs >= PATIENCE:
        print("Early stopping", flush=True)
        break


# Evaluate held-out test participants only once,
# using the best validation checkpoint.
best = torch.load(
    best_path,
    map_location="cpu",
    weights_only=True,
)

model.load_state_dict(best["model_state_dict"])

print("\n========== FINAL TEST ==========", flush=True)
test = run_files(test_files, training=False)

(OUT / "test_results.json").write_text(
    json.dumps(test, indent=2)
)

print(
    f"TEST accuracy={test['accuracy']:.4f} "
    f"macro_F1={test['macro_f1']:.4f}",
    flush=True,
)

print(
    "Training completed in minutes:",
    round((time.time() - start_time) / 60, 1),
    flush=True,
)
print("Best checkpoint:", best_path, flush=True)
