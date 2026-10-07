"""Conservative engineering limits, not a learned sleep-induction policy."""
import math
from src.core.controller import choose_audio_action


class BoundedAudioPolicy:
    def __init__(self):
        self.reset()

    def reset(self):
        self.last_update = None
        self.beat = 8.0
        self.volume = 0.0
        self.sleep_windows = 0

    def choose(self, score, observation, now):
        if not math.isfinite(score) or not math.isfinite(now) or not observation.get("ready"):
            self.reset()
            return {"master_volume": 0.0}
        # Count sleep evidence at >=30 s intervals, not each overlapping prediction.
        if self.last_update is None or now - self.last_update >= 30:
            self.last_update = now
            p = observation["probabilities"]
            sleep = sum(p[k] for k in ("N2", "N3", "REM")) >= .8
            self.sleep_windows = self.sleep_windows + 1 if sleep else 0
            target_volume = .03 if self.sleep_windows >= 2 else .10
            target_beat = choose_audio_action(score)["beat_frequency"]
            # Do not chase stages with increasingly strong stimulation.
            if self.sleep_windows < 2:
                self.beat += max(-.5, min(.5, target_beat - self.beat))
            self.volume += max(-.01, min(.01, target_volume - self.volume))
        return {"carrier_frequency": 200.0, "beat_frequency": min(10., max(6., self.beat)),
                "master_volume": min(.10, max(0., self.volume)),
                "binaural_volume": .01, "wave_volume": .05,
                "rain_volume": 0.0, "bird_volume": 0.0}
