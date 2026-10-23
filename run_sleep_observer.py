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
from src.data.raw_session import RawSessionRecorder
from brainflow.board_shim import BoardShim, BoardIds
from importlib.metadata import version


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
    parser.add_argument("--controller-shadow", type=Path, help="Trained response checkpoint; logs proposals only")
    parser.add_argument("--record-raw", action="store_true", help="Save unmodified BrainFlow reads locally for diagnosis")
    parser.add_argument("--condition", choices=["unspecified", "awake-eyes-open", "awake-eyes-closed"],
                        default="unspecified", help="Self-reported condition for the whole session")
    parser.add_argument("--personal-baseline", type=Path, help="Local personal EEG reference JSON; diagnostic only")
    parser.add_argument("--baseline-reference", choices=["awake-eyes-open", "awake-eyes-closed"],
                        default="awake-eyes-open", help="Which saved awake reference to compare against")
    parser.add_argument("--audio-experiment", action="store_true", help="Automatic sound/silence collection; sleep model disabled")
    parser.add_argument("--experiment-seed", type=int, default=42)
    args = parser.parse_args()
    if args.audio_experiment and args.controller_shadow:
        parser.error("audio-experiment cannot use the sleep-based shadow controller")
    if args.audio_experiment:
        args.record_raw = True
    if not 0 < args.minutes <= 50:
        parser.error("minutes must be >0 and <=50")
    if args.personal_baseline is None and args.baseline_reference != "awake-eyes-open":
        parser.error("baseline-reference requires personal-baseline")
    baseline = None
    if args.personal_baseline is not None:
        from src.signal.personal_baseline import PersonalBaseline
        baseline = PersonalBaseline.load(args.personal_baseline, args.baseline_reference)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    path = Path(__file__).resolve().parent / "models/boas_training_pid/hybrid_boas_best.pt"
    shadow = None
    if args.controller_shadow is not None:
        from src.core.neural_audio import ShadowController
        shadow = ShadowController(args.controller_shadow)
    model = None if args.audio_experiment else HybridSleepModel(path)
    checkpoint_hash = hashlib.sha256(path.read_bytes()).hexdigest() if model else None
    if args.audio_experiment:
        from src.core.audio_experiment import AudioExperiment
        policy = AudioExperiment(args.experiment_seed)
    else:
        policy = BoundedAudioPolicy()
    # Current output starts at zero, so connection/warm-up cannot play audio.
    audio = AudioEngine(carrier_frequency=200, beat_frequency=8,
                        binaural_volume=.01, wave_volume=.05, rain_volume=0,
                        bird_volume=0, master_volume=0, device=args.audio_device) if args.audio else SilentAudio()
    journal = SessionJournal(Path(__file__).resolve().parent / "data/sessions",
                             audio_enabled=args.audio, metadata={
                                 "condition": args.condition, "condition_source": "self_report",
                                 "personal_baseline_sha256": baseline.profile_sha256 if baseline else None,
                                 "personal_baseline_reference": baseline.reference if baseline else None,
                                 "personal_baseline_role": "diagnostic_only" if baseline else None,
                                 "raw_recording_enabled": args.record_raw,
                                 "checkpoint": str(path) if model else None,
                                 "checkpoint_sha256": checkpoint_hash,
                                 "duration_minutes": args.minutes, "audio_device": args.audio_device,
                                 "channels": ["TP9", "AF7", "AF8", "TP10"], "sampling_rate": 256,
                                 "controller_shadow_sha256": hashlib.sha256(args.controller_shadow.read_bytes()).hexdigest() if shadow else None,
                                 "policy": type(policy).__name__, "policy_version": 1,
                                 "experiment_seed": args.experiment_seed if args.audio_experiment else None,
                                 "fast_window_seconds": 2, "sleep_window_seconds": 30,
                                 "muse_transfer": "experimental", "sound_source": "procedural"})
    print("Session journal:", journal.path)
    if baseline is not None:
        print("Personal EEG baseline:", args.personal_baseline, "| Reference:", baseline.reference, "| Diagnostic only")
    status = "returned"
    raw_recorder = None

    print("Automatic audio collection; sleep stage UNKNOWN, model disabled." if args.audio_experiment
          else "Experimental Muse transfer: stage predictions are UNVALIDATED, not confirmed sleep.")
    print("Audio:", "enabled, bounded" if args.audio else "OFF", "| Ctrl+C stops the session.")
    try:
        if args.record_raw:
            raw_recorder = RawSessionRecorder(journal.path.with_suffix('.raw'), metadata={
                "journal": journal.path.name, "condition": args.condition,
                "condition_source": "self_report", "sampling_rate": 256,
                "brainflow_version": version("brainflow"),
                "board_descriptor": BoardShim.get_board_descr(BoardIds.MUSE_2_BOARD.value),
                "eeg_units": "microvolts", "audio_enabled": args.audio,
                "checkpoint_sha256": checkpoint_hash})
            print("Raw EEG recording:", raw_recorder.directory)
        print("Session condition / Умови сеансу:", args.condition)
        session = NeuroSession(MuseDevice(), audio, sleep_observer=SleepObserver(model) if model else None,
                               audio_policy=policy, journal=journal, shadow_controller=shadow, raw_recorder=raw_recorder, personal_baseline=baseline)
        status = session.run(duration_seconds=args.minutes * 60, step_seconds=2)
    except KeyboardInterrupt:
        status = "interrupted"
        print("Session stopped.")
    except Exception:
        status = "error"
        raise
    finally:
        try:
            if raw_recorder is not None:
                raw_recorder.close(status)
        finally:
            journal.close(status)


if __name__ == "__main__":
    main()
