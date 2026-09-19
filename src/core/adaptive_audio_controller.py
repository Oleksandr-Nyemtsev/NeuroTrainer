from collections import deque
from enum import Enum

from src.audio.actions import AudioAction


class SessionPhase(Enum):
    CALMING = "calming"
    SLEEP_ONSET = "sleep_onset"
    RESTORATIVE = "restorative"
    WAKE_UP = "wake_up"


class AdaptiveAudioController:

    def __init__(
        self,
        session_minutes=45,
        history_size=30,
    ):
        self.session_minutes = session_minutes

        self.history = deque(
            maxlen=history_size
        )

        self.phase = SessionPhase.CALMING

        self.last_action = AudioAction()


    def update_phase(
        self,
        minutes_elapsed,
    ):

        progress = (
            minutes_elapsed
            / self.session_minutes
        )

        if progress < 0.20:
            self.phase = SessionPhase.CALMING

        elif progress < 0.45:
            self.phase = SessionPhase.SLEEP_ONSET

        elif progress < 0.85:
            self.phase = SessionPhase.RESTORATIVE

        else:
            self.phase = SessionPhase.WAKE_UP


    def _get_sleep_probability(
        self,
        sleep_probs,
    ):

        n1 = sleep_probs.get("N1", 0.0)
        n2 = sleep_probs.get("N2", 0.0)
        n3 = sleep_probs.get("N3", 0.0)
        rem = sleep_probs.get("REM", 0.0)

        return (
            n1
            + n2
            + n3
            + rem
        )


    def _get_deep_probability(
        self,
        sleep_probs,
    ):

        n3 = sleep_probs.get("N3", 0.0)
        rem = sleep_probs.get("REM", 0.0)

        return n3 + rem


    def _build_calming_action(
        self,
        state,
    ):

        return AudioAction(
            carrier_frequency=180.0,
            beat_frequency=8.0,
            binaural_volume=0.020,

            wave_volume=0.24,
            rain_volume=0.06,
            bird_volume=0.015,

            wave_density=0.60,
            rain_density=0.15,
            bird_density=0.04,

            master_volume=0.65,
            silence_probability=0.05,
        )


    def _build_sleep_onset_action(
        self,
        state,
    ):

        sleep_probs = state["sleep_probs"]

        sleep_probability = (
            self._get_sleep_probability(
                sleep_probs
            )
        )

        # Якщо людина ще явно не засинає,
        # залишаємо трохи більше звукових подій.
        if sleep_probability < 0.50:

            return AudioAction(
                carrier_frequency=175.0,
                beat_frequency=6.5,
                binaural_volume=0.018,

                wave_volume=0.22,
                rain_volume=0.045,
                bird_volume=0.006,

                wave_density=0.52,
                rain_density=0.10,
                bird_density=0.015,

                master_volume=0.58,
                silence_probability=0.10,
            )

        # Якщо перехід у сон вже почався —
        # менше стимуляції.
        return AudioAction(
            carrier_frequency=165.0,
            beat_frequency=5.0,
            binaural_volume=0.014,

            wave_volume=0.18,
            rain_volume=0.030,
            bird_volume=0.0,

            wave_density=0.40,
            rain_density=0.06,
            bird_density=0.0,

            master_volume=0.50,
            silence_probability=0.18,
        )


    def _build_restorative_action(
        self,
        state,
    ):

        sleep_probs = state["sleep_probs"]

        deep_probability = (
            self._get_deep_probability(
                sleep_probs
            )
        )

        # Якщо глибший стан ще нестабільний —
        # залишаємо дуже м'яку підтримку.
        if deep_probability < 0.45:

            return AudioAction(
                carrier_frequency=155.0,
                beat_frequency=4.0,
                binaural_volume=0.010,

                wave_volume=0.15,
                rain_volume=0.020,
                bird_volume=0.0,

                wave_density=0.30,
                rain_density=0.04,
                bird_density=0.0,

                master_volume=0.42,
                silence_probability=0.25,
            )

        # Якщо стан уже стабільний —
        # головна стратегія: НЕ заважати.
        return AudioAction(
            carrier_frequency=150.0,
            beat_frequency=3.0,
            binaural_volume=0.007,

            wave_volume=0.10,
            rain_volume=0.010,
            bird_volume=0.0,

            wave_density=0.18,
            rain_density=0.02,
            bird_density=0.0,

            master_volume=0.34,
            silence_probability=0.40,
        )


    def _build_wake_up_action(
        self,
        state,
    ):

        return AudioAction(
            carrier_frequency=190.0,
            beat_frequency=9.0,
            binaural_volume=0.015,

            wave_volume=0.18,
            rain_volume=0.025,
            bird_volume=0.012,

            wave_density=0.42,
            rain_density=0.06,
            bird_density=0.025,

            master_volume=0.48,
            silence_probability=0.10,
        )


    def choose_action(
        self,
        state,
    ):

        minutes_elapsed = state.get(
            "minutes_elapsed",
            0.0,
        )

        signal_quality = state.get(
            "signal_quality",
            1.0,
        )


        self.update_phase(
            minutes_elapsed
        )


        self.history.append(
            state
        )


        # Якщо EEG поганий —
        # НЕ робимо агресивних змін.
        if signal_quality < 0.50:

            return self.last_action


        if self.phase == SessionPhase.CALMING:

            action = (
                self._build_calming_action(
                    state
                )
            )


        elif self.phase == SessionPhase.SLEEP_ONSET:

            action = (
                self._build_sleep_onset_action(
                    state
                )
            )


        elif self.phase == SessionPhase.RESTORATIVE:

            action = (
                self._build_restorative_action(
                    state
                )
            )


        else:

            action = (
                self._build_wake_up_action(
                    state
                )
            )


        self.last_action = action

        return action