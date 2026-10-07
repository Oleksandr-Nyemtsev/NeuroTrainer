"""Experimental dual-timescale Muse session; silent by default."""
import argparse
import logging
import hashlib
from pathlib import Path

from src.hardware.muse import MuseDevice
from src.audio.engine import AudioEngine
from src.core.session import NeuroSession
from src.models.hybrid_sleep_model import HybridSleepModel
from src.core.sleep_observer import SleepObserver
from src.core.bounded_audio import BoundedAudioPolicy
from src.data.session_journal import SessionJournal


class SilentAudio:
    target_master_volume = 0.0
    def start(self): pass
    def stop(self): pass
    def apply_action(self, action): pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", action="store_true", help="Enable bounded experimental audio")
    parser.add_argument("--minutes", type=float, default=40)
    parser.add_argument("--audio-device", type=int, default=None)
    args = parser.parse_args()
    if not 0 < args.minutes <= 50:
        parser.error("minutes must be >0 and <=50")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    path = Path(__file__).resolve().parent / "models/boas_training_pid/hybrid_boas_best.pt"
    model = HybridSleepModel(path)
    # Current output starts at zero, so connection/warm-up cannot play audio.
    audio = AudioEngine(carrier_frequency=200, beat_frequency=8,
                        binaural_volume=.01, wave_volume=.05, rain_volume=0,
                        bird_volume=0, master_volume=0, device=args.audio_device) if args.audio else SilentAudio()
    journal = SessionJournal(Path(__file__).resolve().parent / "data/sessions",
                             audio_enabled=args.audio, metadata={
                                 "checkpoint": str(path),
                                 "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                 "duration_minutes": args.minutes, "audio_device": args.audio_device,
                                 "channels": ["TP9", "AF7", "AF8", "TP10"], "sampling_rate": 256,
                                 "policy": "BoundedAudioPolicy", "policy_version": 1,
                                 "fast_window_seconds": 2, "sleep_window_seconds": 30,
                                 "muse_transfer": "experimental", "sound_source": "procedural"})
    print("Session journal:", journal.path)
    status = "returned"

    print("Experimental Muse transfer; EEG rhythms every 2 s, rolling sleep context 30 s.")
    print("Audio:", "enabled, bounded" if args.audio else "OFF", "| Ctrl+C stops the session.")
    try:
        session = NeuroSession(MuseDevice(), audio, sleep_observer=SleepObserver(model),
                               audio_policy=BoundedAudioPolicy(), journal=journal)
        status = session.run(duration_seconds=args.minutes * 60, step_seconds=2)
    except KeyboardInterrupt:
        status = "interrupted"
        print("Session stopped.")
    except Exception:
        status = "error"
        raise
    finally:
        journal.close(status)


if __name__ == "__main__":
    main()
