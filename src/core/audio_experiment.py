"""Automatic data collection, not a learned EEG response policy."""
import math
import random


class AudioExperiment:
    requires_sleep_observer = False

    def __init__(self, seed=42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.remaining = []
        self.block_id = 0
        self.reset()

    def reset(self):
        # Any interruption requires a new silent reference before sound.
        self.started = None
        self.last_time = None
        self.candidate = "silence"

    def choose(self, score, observation, now):
        if not math.isfinite(now) or not observation.get("ready"):
            self.reset()
            return {"master_volume": 0.0}
        if self.last_time is not None and (now <= self.last_time or now-self.last_time > 4):
            self.reset()
        self.last_time = now
        if self.started is None:
            self.started = now
            self.block_id += 1
        elif now-self.started >= 60:
            if self.candidate != "silence":
                self.candidate = "silence"
            else:
                if not self.remaining:
                    self.remaining = ["beat_6", "beat_8", "beat_10"]
                    self.rng.shuffle(self.remaining)
                self.candidate = self.remaining.pop()
            self.started = now
            self.block_id += 1
        return {"carrier_frequency": 200.,
                "beat_frequency": 8. if self.candidate == "silence" else float(self.candidate.split("_")[1]),
                "master_volume": 0. if self.candidate == "silence" else .03,
                "binaural_volume": .01, "wave_volume": 0., "rain_volume": 0., "bird_volume": 0.}

    def describe(self):
        return {"mode": "automatic_collection", "candidate": self.candidate,
                "block_id": self.block_id, "block_start": self.started,
                "block_seconds": 60, "seed": self.seed,
                "sleep_stage": "unknown", "learned_selection": False}
