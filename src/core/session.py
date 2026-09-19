import time

from brainflow.board_shim import BoardShim, BoardIds

from src.signal.multichannel import process_eeg
from src.signal.state import build_state
from src.signal.metrics import relaxation_score
from src.signal.smoothing import ScoreSmoother
from src.core.controller import choose_audio_action


class NeuroSession:

    def __init__(self, muse, audio, fs=256):
        self.muse = muse
        self.audio = audio
        self.fs = fs

        self.eeg_channels = BoardShim.get_eeg_channels(
            BoardIds.MUSE_2_BOARD
        )

        self.smoother = ScoreSmoother(
            window_size=5
        )


    def run_step(self, num_samples=512):

        data = self.muse.get_data(num_samples)

        if data.shape[1] < num_samples:
            return {
                "ready": False,
                "samples": data.shape[1]
            }

        eeg = data[self.eeg_channels]

        results = process_eeg(
            eeg,
            fs=self.fs
        )

        state = build_state(results)

        raw_score = relaxation_score(state)

        smoothed_score = self.smoother.update(
            raw_score
        )

        action = choose_audio_action(
            smoothed_score
        )

        self.audio.apply_action(action)

        return {
            "ready": True,
            "state": state,
            "raw_score": raw_score,
            "smoothed_score": smoothed_score,
            "action": action
        }


    def run(self, duration_seconds=60, step_seconds=2):

        start_time = time.time()

        try:
            self.audio.start()

            while time.time() - start_time < duration_seconds:

                result = self.run_step()

                print(result)

                time.sleep(step_seconds)

        finally:
            self.audio.stop()
            self.muse.stop()