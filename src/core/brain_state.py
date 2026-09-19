from dataclasses import dataclass, field
from typing import Dict


@dataclass
class BrainState:

    # Sleep-stage probabilities
    sleep_probs: Dict[str, float] = field(
        default_factory=lambda: {
            "Wake": 1.0,
            "N1": 0.0,
            "N2": 0.0,
            "N3": 0.0,
            "REM": 0.0,
        }
    )

    # Alpha information
    iaf: float | None = None
    alpha_amplitude: float = 0.0
    alpha_phase: float = 0.0
    alpha_stability: float = 0.0

    # Slow-wave information
    slow_wave_amplitude: float = 0.0
    slow_wave_phase: float = 0.0
    slow_wave_stability: float = 0.0

    # Signal quality
    signal_quality: float = 1.0

    # Session timing
    minutes_elapsed: float = 0.0


    def dominant_stage(self):

        return max(
            self.sleep_probs,
            key=self.sleep_probs.get,
        )


    def sleep_probability(self):

        return (
            self.sleep_probs.get("N1", 0.0)
            + self.sleep_probs.get("N2", 0.0)
            + self.sleep_probs.get("N3", 0.0)
            + self.sleep_probs.get("REM", 0.0)
        )


    def deep_probability(self):

        return (
            self.sleep_probs.get("N3", 0.0)
            + self.sleep_probs.get("REM", 0.0)
        )


    def alpha_reliable(
        self,
        threshold=0.70,
    ):

        return (
            self.iaf is not None
            and self.alpha_stability >= threshold
            and self.signal_quality >= 0.70
        )


    def slow_wave_reliable(
        self,
        threshold=0.70,
    ):

        return (
            self.slow_wave_stability >= threshold
            and self.signal_quality >= 0.70
        )


    def to_dict(self):

        return {
            "sleep_probs": self.sleep_probs,

            "iaf": self.iaf,
            "alpha_amplitude":
                self.alpha_amplitude,
            "alpha_phase":
                self.alpha_phase,
            "alpha_stability":
                self.alpha_stability,

            "slow_wave_amplitude":
                self.slow_wave_amplitude,
            "slow_wave_phase":
                self.slow_wave_phase,
            "slow_wave_stability":
                self.slow_wave_stability,

            "signal_quality":
                self.signal_quality,

            "minutes_elapsed":
                self.minutes_elapsed,
        }