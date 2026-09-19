from src.hardware.muse import MuseDevice
from src.audio.engine import AudioEngine
from src.core.session import NeuroSession


def main():

    muse = MuseDevice()

    audio = AudioEngine(
        carrier_frequency=200,
        beat_frequency=8,
        volume=0.10,
        device=3
    )

    session = NeuroSession(
        muse=muse,
        audio=audio
    )

    try:
        print("Connecting Muse 2...")

        muse.connect()
        muse.start()

        print("Muse connected.")
        print("Starting NeuroTrainer session...")

        session.run(
            duration_seconds=60,
            step_seconds=2
        )

    except KeyboardInterrupt:
        print("Session stopped by user.")

    except Exception as error:
        print("ERROR:", error)

    finally:
        audio.stop()
        muse.stop()

        print("NeuroTrainer stopped.")


if __name__ == "__main__":
    main()