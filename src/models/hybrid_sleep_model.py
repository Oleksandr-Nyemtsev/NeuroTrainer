"""Inference for BOAS HybridCRGSleep checkpoints and prepared 30-second epochs."""
from pathlib import Path

import numpy as np
import torch

from src.models.hybrid_crg_sleep import HybridCRGSleep

LABELS = ("Wake", "N1", "N2", "N3", "REM")


class HybridSleepModel:
    def __init__(self, model_path, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(Path(model_path), map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError("Expected a BOAS checkpoint with model_state_dict and class_names")
        if tuple(checkpoint.get("class_names", ())) != LABELS:
            raise ValueError("Checkpoint class_names must be Wake/N1/N2/N3/REM in that order")
        self.metadata = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
        self.model = HybridCRGSleep(num_classes=len(LABELS))
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if any(not torch.isfinite(t).all() for t in self.model.state_dict().values()):
            raise ValueError("Checkpoint contains nonfinite weights")
        self.model.to(self.device).eval()

    def predict_batch(self, eeg):
        """Input is already filtered/resampled/z-scored, NOT raw Muse EEG."""
        x = np.asarray(eeg, dtype=np.float32)
        if x.ndim != 3 or x.shape[1:] != (2, 3000) or len(x) == 0:
            raise ValueError("Expected nonempty EEG batch (N, 2, 3000)")
        if not np.isfinite(x).all():
            raise ValueError("EEG must contain only finite values")
        if not (np.allclose(x.mean(-1), 0, atol=0.01)
                and np.allclose(x.std(-1), 1, atol=0.01)):
            raise ValueError("Expected per-channel z-scored BOAS epochs; raw/flat EEG is not accepted")
        with torch.inference_mode():
            logits = self.model(torch.from_numpy(np.ascontiguousarray(x)).to(self.device))
            probabilities = logits.softmax(dim=1).cpu().numpy()
        if not np.isfinite(probabilities).all():
            raise RuntimeError("Model produced nonfinite probabilities")
        return probabilities

    def predict(self, eeg):
        x = np.asarray(eeg, dtype=np.float32)
        if x.shape != (2, 3000):
            raise ValueError("Expected EEG shape (2, 3000)")
        return dict(zip(LABELS, map(float, self.predict_batch(x[None])[0])))
