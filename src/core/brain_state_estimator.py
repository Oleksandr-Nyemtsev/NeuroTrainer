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

        # Реальная SleepCNN
        sleep_probs = self.sleep_model.predict(
            eeg_for_model
        )


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