"""Offline-trained response predictor. Shadow recommendations never drive audio."""
import math
import torch
from torch import nn

BANDS = ("Delta", "Theta", "Alpha", "Beta", "Gamma")
STAGES = ("Wake", "N1", "N2", "N3", "REM")
ACTIONS = ("silence", "beat_6", "beat_8", "beat_10")
FEATURES = BANDS + STAGES + ("current_beat", "current_volume")


def candidate_parameters(action_id):
    """Versioned candidate definitions, proposals only; not a playback interface."""
    if action_id not in ACTIONS:
        raise ValueError("Unknown audio candidate")
    return {"carrier_frequency": 200.,
            "beat_frequency": 8. if action_id == "silence" else float(action_id.split("_")[1]),
            "master_volume": 0. if action_id == "silence" else .03,
            "binaural_volume": .01, "wave_volume": .05,
            "rain_volume": 0., "bird_volume": 0.}


def encode(observation, current_action):
    """Only pre-action state; no future EEG or ratings in network inputs."""
    if not observation.get("ready") or not observation.get("sleep", {}).get("ready"):
        raise ValueError("Reliable EEG and full sleep context required")
    bands = [float(observation["state"][k]) / 100 for k in BANDS]
    probs = [float(observation["sleep"]["probabilities"][k]) for k in STAGES]
    beat = float(current_action.get("beat_frequency", 8)) / 10
    volume = float(current_action.get("master_volume", 0)) / .1
    values = bands + probs + [beat, volume]
    if (not all(math.isfinite(v) and 0 <= v <= 1 for v in values)
            or not math.isclose(sum(probs), 1, abs_tol=1e-4)
            or not math.isclose(sum(bands), 1, abs_tol=.02)):
        raise ValueError("Invalid controller features")
    return values


class ResponseNet(nn.Module):
    """Predict an explicitly supplied comfort rating, not REM or treatment benefit."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(len(FEATURES) + len(ACTIONS), 32),
                                    nn.Tanh(), nn.Linear(32, 16), nn.Tanh(),
                                    nn.Linear(16, 1), nn.Tanh())

    def forward(self, context, action_ids):
        actions = nn.functional.one_hot(action_ids, len(ACTIONS)).to(context)
        return self.layers(torch.cat((context, actions), dim=-1)).squeeze(-1)


class ShadowController:
    """Load a trained model and log proposals at most once per 30 s."""
    def __init__(self, checkpoint):
        data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (data.get("version") != 1 or data.get("features") != list(FEATURES)
                or data.get("actions") != list(ACTIONS)
                or data.get("objective") != "reported_comfort"
                or data.get("mode") != "shadow_only"):
            raise ValueError("Incompatible response checkpoint")
        self.model = ResponseNet()
        self.model.load_state_dict(data["state_dict"], strict=True)
        if not all(torch.isfinite(p).all() for p in self.model.parameters()):
            raise ValueError("Non-finite response weights")
        self.model.eval()
        self.reset()

    def reset(self):
        self.last_time = None

    def suggest(self, observation, current_action):
        try:
            context = encode(observation, current_action)
            now = float(observation["window_end_timestamp"])
            if not math.isfinite(now):
                raise ValueError("Invalid timestamp")
        except (ValueError, KeyError, TypeError):
            self.reset()
            return {"ready": False, "mode": "shadow_only", "reason": "Invalid or incomplete context"}
        if self.last_time is not None and now < self.last_time:
            self.reset()
            return {"ready": False, "mode": "shadow_only", "reason": "Time moved backwards"}
        if self.last_time is not None and now - self.last_time < 30:
            return {"ready": False, "mode": "shadow_only", "reason": "Waiting for decision interval"}
        self.last_time = now
        with torch.inference_mode():
            scores = self.model(torch.tensor([context] * len(ACTIONS), dtype=torch.float32),
                                torch.arange(len(ACTIONS)))
        if not torch.isfinite(scores).all():
            self.reset()
            return {"ready": False, "mode": "shadow_only", "reason": "Invalid model output"}
        return {"ready": True, "mode": "shadow_only", "applied": False,
                "objective": "reported_comfort", "features": context,
                "action_id": ACTIONS[int(scores.argmax())],
                "proposed_parameters": candidate_parameters(ACTIONS[int(scores.argmax())]),
                "predicted_ratings": dict(zip(ACTIONS, scores.tolist()))}
