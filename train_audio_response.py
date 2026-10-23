"""Train on explicitly rated audio trials. No hardware, playback, or inferred rewards."""
import argparse
import copy
import json
from pathlib import Path
import torch
from src.core.neural_audio import ACTIONS, FEATURES, ResponseNet, encode, candidate_parameters


def read_trials(path):
    rows = []
    seen = set()
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            key = (row["session_id"], row["trial_id"])
            if not all(isinstance(k, str) and k.strip() for k in key) or key in seen:
                raise ValueError("Missing or repeated trial identity")
            if row["rating_source"] != "self_report" or row["action_id"] not in ACTIONS:
                raise ValueError("Explicit self-report and known action required")
            if row["delivered"] is not True:
                raise ValueError("Shadow proposals are not delivered trials")
            if row["delivered_parameters"] != candidate_parameters(row["action_id"]):
                raise ValueError("Delivered settings must match the versioned candidate")
            rating = float(row["rating"])
            if not -1 <= rating <= 1:
                raise ValueError("Rating must be between -1 and 1")
            before = float(row["before"]["window_end_timestamp"])
            start, end, rated = [float(row[k]) for k in ("action_started_at", "action_ended_at", "rated_at")]
            if not before <= start < end <= rated or end - start < 30:
                raise ValueError("Require pre-action context and at least 30 seconds exposure")
            # Validate numeric timestamps including infinity.
            import math
            if not all(math.isfinite(v) for v in (before, start, end, rated)):
                raise ValueError("Invalid trial timestamps")
            context = encode(row["before"], row["previous_action"])
            rows.append((key[0], context, ACTIONS.index(row["action_id"]), rating))
            seen.add(key)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Trial line {number}: {error}") from error
    if not rows:
        raise ValueError("No rated trials; session journals alone are not training labels")
    return rows


def train(rows, validation_sessions, epochs=100, seed=17):
    if not 1 <= epochs <= 2000:
        raise ValueError("epochs must be between 1 and 2000")
    sessions = {r[0] for r in rows}
    validation_sessions = set(validation_sessions)
    if not validation_sessions or not validation_sessions < sessions:
        raise ValueError("Choose existing held-out sessions and keep training sessions")
    training = [r for r in rows if r[0] not in validation_sessions]
    validation = [r for r in rows if r[0] in validation_sessions]
    if {r[2] for r in training} != set(range(len(ACTIONS))):
        raise ValueError("Training requires rated examples of every candidate, including silence")
    def tensors(group):
        return (torch.tensor([r[1] for r in group], dtype=torch.float32),
                torch.tensor([r[2] for r in group], dtype=torch.long),
                torch.tensor([r[3] for r in group], dtype=torch.float32))
    x, a, y = tensors(training)
    vx, va, vy = tensors(validation)
    torch.manual_seed(seed)
    model = ResponseNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=.003, weight_decay=.001)
    best = float("inf")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = (model(x, a) - y).square().mean()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            error = (model(vx, va) - vy).abs().mean().item()
        if error < best:
            best, state, best_epoch = error, copy.deepcopy(model.state_dict()), epoch + 1
    baseline = (vy - y.median()).abs().mean().item()
    return {"version": 1, "mode": "shadow_only", "objective": "reported_comfort",
            "features": list(FEATURES), "actions": list(ACTIONS), "state_dict": state,
            "validation_mae": best, "constant_baseline_mae": baseline,
            "best_epoch": best_epoch, "seed": seed,
            "training_sessions": sorted(sessions - validation_sessions),
            "validation_sessions": sorted(validation_sessions),
            "training_trials": len(training), "validation_trials": len(validation)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trials", type=Path)
    parser.add_argument("--validation-session", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    checkpoint = train(read_trials(args.trials), args.validation_session, args.epochs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        torch.save(checkpoint, handle)
    print(json.dumps({k: v for k, v in checkpoint.items() if k != "state_dict"}, indent=2))
    print("Shadow-only checkpoint. Rating prediction is not evidence of improved sleep.")


if __name__ == "__main__":
    main()
