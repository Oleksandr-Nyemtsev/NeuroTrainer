import numpy as np
from src.core.brain_state import BrainState
from src.signal.alpha_tracker import AlphaTracker
from src.signal.slow_wave_detector import SlowWaveDetector


class BrainStateEstimator:

    def __init__(
        self,
        sleep_model,
        fs=256,
    ):

        self.sleep_model = sleep_model
        self.fs = fs

        self.alpha_tracker = AlphaTracker(
            fs=fs
        )

        self.slow_wave_detector = SlowWaveDetector(
            fs=fs
        )


    def estimate(
        self,
        eeg_for_model,
        eeg_raw,
        signal_quality=1.0,
        minutes_elapsed=0.0,
    ):

        if (not np.isfinite(signal_quality) or not np.isfinite(minutes_elapsed)
                or np.size(eeg_raw) == 0 or np.size(eeg_for_model) == 0
                or not np.isfinite(eeg_raw).all() or not np.isfinite(eeg_for_model).all()):
            return BrainState(signal_quality=0.0)

        # Numerical flatline gate in raw input units, before model inference.
        if np.any(np.std(eeg_raw, axis=-1) <= self.alpha_tracker.min_amplitude):
            return BrainState(signal_quality=0.0)

        # Реальная SleepCNN
        sleep_probs = self.sleep_model.predict(
            eeg_for_model
        )


        if not sleep_probs or not all(np.isfinite(v) for v in sleep_probs.values()):
            return BrainState(signal_quality=0.0)

        # Alpha
        alpha_result = (
            self.alpha_tracker.analyze(
                eeg_raw
            )
        )


        # Slow wave
        slow_wave_result = (
            self.slow_wave_detector.analyze(
                eeg_raw
            )
        )


        state = BrainState(

            sleep_probs=sleep_probs,

            iaf=alpha_result[
                "iaf"
            ],

            alpha_amplitude=alpha_result[
                "alpha_amplitude"
            ],

            alpha_phase=alpha_result[
                "alpha_phase"
            ],

            alpha_stability=alpha_result[
                "alpha_stability"
            ],


            slow_wave_amplitude=(
                slow_wave_result[
                    "slow_wave_amplitude"
                ]
            ),

            slow_wave_phase=(
                slow_wave_result[
                    "slow_wave_phase"
                ]
            ),

            slow_wave_stability=(
                slow_wave_result[
                    "slow_wave_stability"
                ]
            ),


            signal_quality=float(
                signal_quality
            ),

            minutes_elapsed=float(
                minutes_elapsed
            ),
        )

        return state