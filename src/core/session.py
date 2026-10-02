import logging
import time

import numpy as np
from brainflow.board_shim import BoardShim, BoardIds

from src.signal.multichannel import process_eeg
from src.signal.state import build_state
from src.signal.metrics import relaxation_score
from src.signal.smoothing import ScoreSmoother
from src.signal.muse_quality import MuseSignalQuality
from src.core.controller import choose_audio_action, neutral_audio_action

logger = logging.getLogger(__name__)


class NeuroSession:
    def __init__(self, muse, audio, fs=256, max_latency_seconds=4.0):
        if not np.isfinite(fs) or fs <= 0:
            raise ValueError("fs must be positive and finite")
        if not np.isfinite(max_latency_seconds) or max_latency_seconds <= 0:
            raise ValueError("max_latency_seconds must be positive and finite")
        self.muse = muse
        self.audio = audio
        self.fs = fs
        board = BoardIds.MUSE_2_BOARD.value
        self.eeg_channels = BoardShim.get_eeg_channels(board)
        self.timestamp_channel = BoardShim.get_timestamp_channel(board)
        self.channel_names = ["TP9", "AF7", "AF8", "TP10"]
        self.window_samples = int(2 * fs)
        self.max_latency_seconds = max_latency_seconds
        self.eeg_buffer = np.empty((4, 0))
        self.timestamp_buffer = np.empty(0)
        self.last_received_timestamp = None
        self.dropped_samples = 0
        self.quality_checker = MuseSignalQuality(fs=fs)
        self.smoother = ScoreSmoother(window_size=5)
        self.normal_master_volume = getattr(audio, "target_master_volume", 0.10)
        self.in_safe_state = False
        self.diagnostics = self._diagnostics()

    def _diagnostics(self, timestamps=None):
        has_time = timestamps is not None and len(timestamps) > 0
        return {
            "window_start_timestamp": float(timestamps[0]) if has_time else None,
            "window_end_timestamp": float(timestamps[-1]) if has_time else None,
            "latency_seconds": max(0.0, time.time() - float(timestamps[-1])) if has_time else None,
            # Complete-window equivalents; dropped_samples includes partial drops.
            "dropped_windows": self.dropped_samples // self.window_samples,
            "dropped_samples": self.dropped_samples,
            "buffer_samples": self.eeg_buffer.shape[1],
        }

    def _clear_buffer(self):
        self.eeg_buffer = np.empty((4, 0))
        self.timestamp_buffer = np.empty(0)

    def _safe_result(self, reason, clear_buffer=True, discarded_samples=0, **details):
        self.dropped_samples += discarded_samples
        if clear_buffer:
            self.dropped_samples += self.eeg_buffer.shape[1]
            self._clear_buffer()
        self.diagnostics["dropped_samples"] = self.dropped_samples
        self.diagnostics["dropped_windows"] = self.dropped_samples // self.window_samples
        self.smoother.values.clear()
        self.in_safe_state = True
        action = neutral_audio_action()
        self.audio.apply_action(action)
        self.diagnostics["buffer_samples"] = self.eeg_buffer.shape[1]
        result = {"ready": False, "reason": reason, "action": action,
                  **self.diagnostics, **details}
        logger.info("EEG safe state: %s", result)
        return result

    def run_step(self):
        self.diagnostics = self._diagnostics()
        data = self.muse.get_data()
        if data is None or data.size == 0:
            self.last_received_timestamp = None
            return self._safe_result("Waiting for EEG")
        if data.ndim != 2 or data.shape[0] <= max(*self.eeg_channels, self.timestamp_channel):
            return self._safe_result("Invalid BrainFlow data shape")
        eeg_new = data[self.eeg_channels]
        timestamps = data[self.timestamp_channel]
        if not np.isfinite(timestamps).all():
            self.last_received_timestamp = None
            return self._safe_result("Invalid EEG timestamps", discarded_samples=data.shape[1])

        # BrainFlow timestamps are Unix seconds. Include the boundary between reads.
        check_times = timestamps
        if self.last_received_timestamp is not None:
            check_times = np.r_[self.last_received_timestamp, timestamps]
        intervals = np.diff(check_times)
        self.last_received_timestamp = float(timestamps[-1])
        self.diagnostics = self._diagnostics(timestamps)
        if np.any(intervals <= 0) or np.any(intervals > 1.5 / self.fs):
            return self._safe_result("EEG timestamp gap or duplicate", discarded_samples=data.shape[1])
        if (self.diagnostics["latency_seconds"] > self.max_latency_seconds
                or timestamps[-1] > time.time() + self.max_latency_seconds):
            return self._safe_result("Stale or future EEG timestamps", discarded_samples=data.shape[1])

        # Keep at most the newest complete window. Never accumulate an old backlog.
        previous = self.eeg_buffer.shape[1]
        incoming = eeg_new.shape[1]
        excess = max(0, previous + incoming - self.window_samples)
        self.dropped_samples += excess
        if incoming >= self.window_samples:
            self.eeg_buffer = eeg_new[:, -self.window_samples:].copy()
            self.timestamp_buffer = timestamps[-self.window_samples:].copy()
        else:
            self.eeg_buffer = np.concatenate([self.eeg_buffer[:, excess:], eeg_new], axis=1)
            self.timestamp_buffer = np.concatenate([self.timestamp_buffer[excess:], timestamps])
        self.diagnostics = self._diagnostics(self.timestamp_buffer)
        if self.eeg_buffer.shape[1] < self.window_samples:
            return self._safe_result("Waiting for full window", clear_buffer=False)

        eeg = self.eeg_buffer.copy()
        window_timestamps = self.timestamp_buffer.copy()
        self._clear_buffer()
        self.diagnostics["buffer_samples"] = 0
        quality = self.quality_checker.analyze(eeg, timestamps=window_timestamps)
        scores = [channel["quality"] for channel in quality["channels"]]
        for name, info in zip(self.channel_names, quality["channels"]):
            logger.info("%s: Q=%.2f Amplitude=%.1f uV Flatline=%s", name,
                        info["quality"], info["amplitude_uv"], info["flatline"])
        if min(scores) < 0.75:
            return self._safe_result("Unreliable EEG", discarded_samples=self.window_samples, channel_quality=scores)
        try:
            results = process_eeg(eeg, fs=self.fs)
            state = build_state(results)
            raw_score = relaxation_score(state)
            smoothed_score = self.smoother.update(raw_score)
        except ValueError as error:
            return self._safe_result(str(error), discarded_samples=self.window_samples)

        # Include processing time in age; never act on a window that became stale.
        self.diagnostics = self._diagnostics(window_timestamps)
        if self.diagnostics["latency_seconds"] > self.max_latency_seconds:
            return self._safe_result("EEG became stale during processing", discarded_samples=self.window_samples)
        action = choose_audio_action(smoothed_score)
        if self.in_safe_state:
            action["master_volume"] = self.normal_master_volume
            self.in_safe_state = False
        self.audio.apply_action(action)
        result = {"ready": True, "channel_quality": scores, "state": state,
                  "raw_score": raw_score, "smoothed_score": smoothed_score,
                  "action": action, **self.diagnostics}
        logger.info("EEG decision: %s", result)
        return result

    def run(self, duration_seconds=30, step_seconds=2):
        if not np.isfinite(step_seconds) or step_seconds <= 0:
            raise ValueError("step_seconds must be positive and finite")
        if not self.muse.connect():
            print("Unable to connect to Muse.")
            return
        if not self.muse.start():
            self.muse.stop()
            return
        start_time = time.monotonic()
        next_step = start_time
        try:
            self.audio.start()
            while time.monotonic() - start_time < duration_seconds:
                result = self.run_step()
                print(result)
                next_step += step_seconds
                now = time.monotonic()
                next_step = max(next_step, now)
                time.sleep(max(0.0, min(next_step - now, start_time + duration_seconds - now)))
        finally:
            try:
                self.audio.stop()
            finally:
                self.muse.stop()
