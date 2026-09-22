
import time

import numpy as np

from brainflow.board_shim import BoardShim, BoardIds

from src.signal.multichannel import process_eeg
from src.signal.state import build_state
from src.signal.metrics import relaxation_score
from src.signal.smoothing import ScoreSmoother
from src.signal.muse_quality import MuseSignalQuality
from src.core.controller import choose_audio_action


class NeuroSession:

    def __init__(self, muse, audio, fs=256):
        self.muse = muse
        self.audio = audio
        self.fs = fs

        self.eeg_channels = BoardShim.get_eeg_channels(
            BoardIds.MUSE_2_BOARD.value
        )

        self.channel_names = ["TP9", "AF7", "AF8", "TP10"]
        self.window_samples = int(2 * fs)

        self.eeg_buffer = np.empty((4, 0))

        self.quality_checker = MuseSignalQuality(fs=fs)
        self.smoother = ScoreSmoother(window_size=5)

    def run_step(self):

        # Read only new samples from BrainFlow.
        data = self.muse.get_data()

        if data is None or data.size == 0:
            return {
                "ready": False,
                "reason": "Waiting for EEG",
            }

        eeg_new = data[self.eeg_channels]

        if eeg_new.shape[0] != 4:
            raise ValueError("Expected four Muse EEG channels")

        # Add new samples to the buffer.
        self.eeg_buffer = np.concatenate(
            [self.eeg_buffer, eeg_new],
            axis=1,
        )

        if self.eeg_buffer.shape[1] < self.window_samples:
            return {
                "ready": False,
                "samples": self.eeg_buffer.shape[1],
            }

        # Extract one complete window.
        eeg = self.eeg_buffer[:, :self.window_samples].copy()

        # Keep unused samples for the next step.
        self.eeg_buffer = self.eeg_buffer[:, self.window_samples:]

        # ----------------------------------------
        # SIGNAL QUALITY
        # ----------------------------------------

        quality = self.quality_checker.analyze(eeg)

        for name, info in zip(
            self.channel_names,
            quality["channels"],
        ):
            print(
                f"{name}: "
                f"Q={info['quality']:.2f} "
                f"Amplitude={info['amplitude_uv']:.1f} uV "
                f"Flatline={info['flatline']}"
            )

        scores = [
            channel["quality"]
            for channel in quality["channels"]
        ]

        if min(scores) < 0.75:
            return {
                "ready": False,
                "reason": "Unreliable EEG",
                "channel_quality": scores,
                "audio": "Previous action unchanged",
            }

        # ----------------------------------------
        # CURRENT EEG PIPELINE
        # ----------------------------------------

        results = process_eeg(eeg, fs=self.fs)

        state = build_state(results)

        raw_score = relaxation_score(state)

        smoothed_score = self.smoother.update(raw_score)

        action = choose_audio_action(smoothed_score)

        self.audio.apply_action(action)

        return {
            "ready": True,
            "channel_quality": scores,
            "state": state,
            "raw_score": raw_score,
            "smoothed_score": smoothed_score,
            "action": action,
        }

    def run(self, duration_seconds=30, step_seconds=2):

        if not self.muse.connect():
            print("Unable to connect to Muse.")
            return

        if not self.muse.start():
            self.muse.stop()
            return

        start_time = time.monotonic()

        try:
            self.audio.start()

            while time.monotonic() - start_time < duration_seconds:
                result = self.run_step()
                print(result)
                time.sleep(step_seconds)

        finally:
            self.audio.stop()
            self.muse.stop()