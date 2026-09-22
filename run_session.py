
from src.hardware.muse import MuseDevice
from src.audio.engine import AudioEngine
from src.core.session import NeuroSession


def main():
    muse = MuseDevice()

    audio = AudioEngine(
        carrier_frequency=200,
        beat_frequency=8,
        binaural_volume=0.01,
        wave_volume=0.05,
        rain_volume=0.0,
        bird_volume=0.0,
        master_volume=0.10,
        device=3,
    )

    session = NeuroSession(
        muse=muse,
        audio=audio,
        fs=256,
    )

    print("NeuroTrainer — Muse 2")
    print("Test duration: 30 seconds")

    try:
        session.run(
            duration_seconds=30,
            step_seconds=2,
        )

    except KeyboardInterrupt:
        print("\nSession interrupted.")

    except Exception as error:
        print("Session error:", error)


if __name__ == "__main__":
    main()