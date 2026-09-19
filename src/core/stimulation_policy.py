from dataclasses import dataclass
from enum import Enum

from src.core.brain_state import BrainState


class StimulationMode(Enum):

    NONE = "none"

    COMFORT_ONLY = "comfort_only"

    ALPHA_TRACKING = "alpha_tracking"

    SLOW_WAVE_TRACKING = "slow_wave_tracking"

    WAKE_UP = "wake_up"


@dataclass
class StimulationDecision:

    mode: StimulationMode

    enabled: bool = False

    target_frequency: float | None = None

    target_phase: float | None = None

    pulse_gain: float = 0.0

    reason: str = ""


class StimulationPolicy:

    def __init__(
        self,
        min_signal_quality=0.70,
        min_stage_confidence=0.60,
    ):

        self.min_signal_quality = (
            min_signal_quality
        )

        self.min_stage_confidence = (
            min_stage_confidence
        )


    def decide(
        self,
        state: BrainState,
    ) -> StimulationDecision:

        # --------------------------------
        # 1. Поганий EEG
        # --------------------------------

        if (
            state.signal_quality
            < self.min_signal_quality
        ):

            return StimulationDecision(
                mode=StimulationMode.NONE,
                enabled=False,
                reason="Low EEG signal quality",
            )


        dominant_stage = (
            state.dominant_stage()
        )


        stage_confidence = (
            state.sleep_probs.get(
                dominant_stage,
                0.0,
            )
        )


        # --------------------------------
        # 2. Модель сама не впевнена
        # --------------------------------

        if (
            stage_confidence
            < self.min_stage_confidence
        ):

            return StimulationDecision(
                mode=StimulationMode.COMFORT_ONLY,
                enabled=False,
                reason="Sleep-stage confidence too low",
            )


        # --------------------------------
        # 3. REM-like
        #
        # Не намагаємося активно
        # стимулювати REM.
        # --------------------------------

        if dominant_stage == "REM":

            return StimulationDecision(
                mode=StimulationMode.COMFORT_ONLY,
                enabled=False,
                reason="REM-like state detected",
            )


        # --------------------------------
        # 4. N3 / stable slow wave
        # --------------------------------

        if dominant_stage == "N3":

            if state.slow_wave_reliable():

                return StimulationDecision(
                    mode=(
                        StimulationMode
                        .SLOW_WAVE_TRACKING
                    ),
                    enabled=True,

                    target_frequency=None,

                    target_phase=(
                        state.slow_wave_phase
                    ),

                    pulse_gain=0.05,

                    reason=(
                        "Stable N3-like state "
                        "with reliable slow-wave activity"
                    ),
                )


            return StimulationDecision(
                mode=StimulationMode.COMFORT_ONLY,
                enabled=False,
                reason=(
                    "N3-like state but "
                    "slow wave is not reliable"
                ),
            )


        # --------------------------------
        # 5. Wake / N1
        #
        # Alpha tracking дозволяємо
        # лише якщо alpha надійна.
        # --------------------------------

        if dominant_stage in (
            "Wake",
            "N1",
        ):

            if state.alpha_reliable():

                return StimulationDecision(
                    mode=(
                        StimulationMode
                        .ALPHA_TRACKING
                    ),
                    enabled=True,

                    target_frequency=(
                        state.iaf
                    ),

                    target_phase=(
                        state.alpha_phase
                    ),

                    pulse_gain=0.04,

                    reason=(
                        "Reliable individual "
                        "alpha rhythm detected"
                    ),
                )


            return StimulationDecision(
                mode=StimulationMode.COMFORT_ONLY,
                enabled=False,
                reason=(
                    "Alpha rhythm "
                    "is not reliable"
                ),
            )


        # --------------------------------
        # 6. N2
        #
        # Поки не робимо агресивної
        # stimulation.
        # --------------------------------

        if dominant_stage == "N2":

            return StimulationDecision(
                mode=StimulationMode.COMFORT_ONLY,
                enabled=False,
                reason=(
                    "Stable N2-like state; "
                    "avoid unnecessary stimulation"
                ),
            )


        # --------------------------------
        # Fallback
        # --------------------------------

        return StimulationDecision(
            mode=StimulationMode.NONE,
            enabled=False,
            reason="No suitable stimulation mode",
        )