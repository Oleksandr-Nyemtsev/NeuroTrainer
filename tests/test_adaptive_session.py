import time

from src.audio.engine import AudioEngine
from src.core.adaptive_audio_controller import (
    AdaptiveAudioController,
)


def main():

    audio = AudioEngine(
        device=None,
    )

    controller = AdaptiveAudioController(
        session_minutes=45,
    )

    test_states = [

        {
            "minutes_elapsed": 2,
            "signal_quality": 0.95,
            "sleep_probs": {
                "Wake": 0.85,
                "N1": 0.10,
                "N2": 0.03,
                "N3": 0.01,
                "REM": 0.01,
            },
        },

        {
            "minutes_elapsed": 12,
            "signal_quality": 0.95,
            "sleep_probs": {
                "Wake": 0.45,
                "N1": 0.35,
                "N2": 0.15,
                "N3": 0.03,
                "REM": 0.02,
            },
        },

        {
            "minutes_elapsed": 22,
            "signal_quality": 0.95,
            "sleep_probs": {
                "Wake": 0.15,
                "N1": 0.20,
                "N2": 0.40,
                "N3": 0.20,
                "REM": 0.05,
            },
        },

        {
            "minutes_elapsed": 32,
            "signal_quality": 0.95,
            "sleep_probs": {
                "Wake": 0.05,
                "N1": 0.10,
                "N2": 0.30,
                "N3": 0.45,
                "REM": 0.10,
            },
        },

        {
            "minutes_elapsed": 42,
            "signal_quality": 0.95,
            "sleep_probs": {
                "Wake": 0.20,
                "N1": 0.25,
                "N2": 0.30,
                "N3": 0.15,
                "REM": 0.10,
            },
        },
    ]

    try:

        audio.start()

        for state in test_states:

            action = controller.choose_action(
                state
            )

            print()
            print(
                "Minutes:",
                state["minutes_elapsed"]
            )

            print(
                "Phase:",
                controller.phase.value
            )

            print(
                "Action:",
                action
            )

            audio.apply_action(
    action
)

            time.sleep(10)

    finally:

        audio.stop()


if __name__ == "__main__":
    main()
