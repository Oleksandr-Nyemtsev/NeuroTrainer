import logging
import time

import numpy as np
from brainflow.board_shim import BoardShim, BoardIds

from src.core.packet_diagnostics import MusePacketDiagnostics, packet_sequence_usable, timestamp_interval_limit
from src.signal.multichannel import process_eeg
from src.signal.state import build_state
from src.signal.metrics import relaxation_score
from src.signal.smoothing import ScoreSmoother
from src.signal.muse_quality import MuseSignalQuality
from src.core.controller import choose_audio_action, neutral_audio_action

logger = logging.getLogger(__name__)


class NeuroSession:
    def __init__(self, muse, audio, fs=256, max_latency_seconds=4.0,
                 sleep_observer=None, audio_policy=None, journal=None, shadow_controller=None, raw_recorder=None, personal_baseline=None):
        if not np.isfinite(fs) or fs <= 0:
            raise ValueError("fs must be positive and finite")
        if not np.isfinite(max_latency_seconds) or max_latency_seconds <= 0:
            raise ValueError("max_latency_seconds must be positive and finite")
        if (audio_policy is not None and sleep_observer is None
                and getattr(audio_policy, "requires_sleep_observer", True)):
            raise ValueError("Audio policy requires a sleep observer")
        if personal_baseline is not None and fs != 256:
            raise ValueError("Personal baseline requires Muse sampling rate 256 Hz")
        self.personal_baseline = personal_baseline
        self.shadow_controller = shadow_controller
        self.raw_recorder = raw_recorder
        self.journal = journal
        self.sleep_observer = sleep_observer
        self.audio_policy = audio_policy
        self.rolling_fast_window = (audio_policy is not None and
                                    not getattr(audio_policy, "requires_sleep_observer", True))
        self.packet_tracker = MusePacketDiagnostics()
        self.packet_diagnostics = {}
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
        self.timestamp_diagnostics = {}
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
            **self.timestamp_diagnostics,
            **self.packet_diagnostics,
        }

    def _apply_audio(self, action):
        started = time.time()
        self.audio.apply_action(action)
        completed = time.time()
        parameters = ("carrier_frequency", "beat_frequency", "master_volume",
                      "binaural_volume", "wave_volume", "rain_volume", "bird_volume",
                      "wave_density", "rain_density", "bird_density", "silence_probability")
        return {"started_at": started, "completed_at": completed,
                "software_current_parameters": {
                    k: float(getattr(self.audio, k)) for k in parameters if hasattr(self.audio, k)}}

    def _clear_buffer(self):
        self.eeg_buffer = np.empty((4, 0))
        self.timestamp_buffer = np.empty(0)

    def _safe_result(self, reason, clear_buffer=True, discarded_samples=0, **details):
        if self.shadow_controller is not None:
            self.shadow_controller.reset()
        self.dropped_samples += discarded_samples
        if clear_buffer:
            if self.sleep_observer is not None:
                self.sleep_observer.reset()
            if self.audio_policy is not None:
                self.audio_policy.reset()
            self.dropped_samples += self.eeg_buffer.shape[1]
            self._clear_buffer()
        self.diagnostics["dropped_samples"] = self.dropped_samples
        self.diagnostics["dropped_windows"] = self.dropped_samples // self.window_samples
        if self.audio_policy is not None and not getattr(self.audio_policy, "requires_sleep_observer", True):
            self.audio_policy.reset()
        self.smoother.values.clear()
        self.in_safe_state = True
        action = neutral_audio_action()
        command = self._apply_audio(action)
        self.diagnostics["buffer_samples"] = self.eeg_buffer.shape[1]
        result = {"ready": False, "reason": reason, "action": action,
                  **self.diagnostics, **details}
        if self.personal_baseline is not None:
            result["personal_baseline"] = {"ready": False, "role": "diagnostic_only",
                "reference": self.personal_baseline.reference, "reason": "EEG unavailable or rejected"}
        if self.journal is not None:
            self.journal.record(result, command)
        logger.info("EEG safe state: %s", result)
        return result

    def run_step(self):
        self.packet_diagnostics = {}
        self.timestamp_diagnostics = {}
        self.diagnostics = self._diagnostics()
        data = self.muse.get_data()
        if self.raw_recorder is not None:
            self.raw_recorder.record(data)
        if data is None or data.size == 0:
            self.last_received_timestamp = None
            return self._safe_result("Waiting for EEG")
        if data.ndim != 2 or data.shape[0] <= max(*self.eeg_channels, self.timestamp_channel):
            return self._safe_result("Invalid BrainFlow data shape")
        packet_channel = getattr(self.muse, "package_num_channel", None)
        if packet_channel is not None and 0 <= packet_channel < data.shape[0]:
            self.packet_diagnostics = {"packet_diagnostics":
                self.packet_tracker.update(data[packet_channel])}
        else:
            self.packet_diagnostics = {"packet_diagnostics":
                {"available": False, "reason": "Packet channel unavailable"}}
        self.diagnostics = self._diagnostics()
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
        packet_evidence = packet_sequence_usable(self.packet_diagnostics.get("packet_diagnostics", {}))
        interval_limit = timestamp_interval_limit(self.fs, packet_evidence)
        # Keep the old threshold count for comparison; record the applied limit too.
        self.timestamp_diagnostics = {"timestamp_diagnostics": {
            "received_samples": int(timestamps.size),
            "expected_interval_ms": 1000.0 / self.fs,
            "gap_threshold_ms": 1500.0 / self.fs,
            "applied_gap_threshold_ms": interval_limit * 1000,
            "packet_evidence": packet_evidence,
            "applied_gap_count": int(np.count_nonzero(intervals > interval_limit)),
            "min_interval_ms": float(intervals.min() * 1000) if intervals.size else None,
            "max_interval_ms": float(intervals.max() * 1000) if intervals.size else None,
            "duplicate_count": int(np.count_nonzero(intervals == 0)),
            "backward_count": int(np.count_nonzero(intervals < 0)),
            "gap_count": int(np.count_nonzero(intervals > 1.5 / self.fs)),
            "boundary_interval_ms": float((timestamps[0] - self.last_received_timestamp) * 1000)
                if self.last_received_timestamp is not None else None,
        }}
        self.last_received_timestamp = float(timestamps[-1])
        self.diagnostics = self._diagnostics(timestamps)
        if packet_channel is not None and not packet_evidence:
            return self._safe_result("EEG packet sequence anomaly", discarded_samples=data.shape[1])
        if np.any(intervals <= 0) or np.any(intervals > interval_limit):
            return self._safe_result("EEG timestamp gap or duplicate", discarded_samples=data.shape[1])
        if (self.diagnostics["latency_seconds"] > self.max_latency_seconds
                or timestamps[-1] > time.time() + self.max_latency_seconds):
            return self._safe_result("Stale or future EEG timestamps", discarded_samples=data.shape[1])

        sleep_result = None
        if self.sleep_observer is not None:
            try:
                if packet_evidence:
                    sleep_result = self.sleep_observer.update(eeg_new, timestamps, packet_evidence=True)
                else:
                    sleep_result = self.sleep_observer.update(eeg_new, timestamps)
            except (ValueError, RuntimeError) as error:
                return self._safe_result(str(error), discarded_samples=data.shape[1])

        # Keep at most the newest complete window. Never accumulate an old backlog.
        previous = self.eeg_buffer.shape[1]
        incoming = eeg_new.shape[1]
        excess = max(0, previous + incoming - self.window_samples)
        self.dropped_samples += max(0, incoming - self.window_samples) if self.rolling_fast_window else excess
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
        # Keep recent samples for a rolling window: BLE reads need not contain
        # exactly 512 samples. Quality/timing failures still clear the buffer.
        if not self.rolling_fast_window:
            self._clear_buffer()
        self.diagnostics["buffer_samples"] = self.eeg_buffer.shape[1]
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

        baseline_result = None
        if self.personal_baseline is not None:
            try:
                baseline_result = self.personal_baseline.compare(eeg)
            except ValueError as error:
                # A diagnostic failure must not replace the existing state/audio path.
                baseline_result = {"ready": False, "role": "diagnostic_only", "reason": str(error)}

        # Include processing time in age; never act on a window that became stale.
        self.diagnostics = self._diagnostics(window_timestamps)
        if self.diagnostics["latency_seconds"] > self.max_latency_seconds:
            return self._safe_result("EEG became stale during processing", discarded_samples=self.window_samples)
        if self.audio_policy is not None:
            policy_context = sleep_result if self.sleep_observer is not None else {"ready": True, "state": state}
            action = self.audio_policy.choose(smoothed_score, policy_context, float(timestamps[-1]))
        else:
            action = choose_audio_action(smoothed_score)
            if self.in_safe_state:
                action["master_volume"] = self.normal_master_volume
        self.in_safe_state = False
        previous_audio = {"beat_frequency": getattr(self.audio, "target_beat_frequency", 8.),
                          "master_volume": getattr(self.audio, "target_master_volume", 0.)}
        command = self._apply_audio(action)
        result = {"ready": True, "channel_quality": scores, "state": state,
                  "raw_score": raw_score, "smoothed_score": smoothed_score,
                  "action": action, **self.diagnostics}
        if self.audio_policy is not None and hasattr(self.audio_policy, "describe"):
            result["audio_experiment"] = self.audio_policy.describe()
        if baseline_result is not None:
            result["personal_baseline"] = baseline_result
        if sleep_result is not None:
            result["sleep"] = sleep_result
        if self.shadow_controller is not None:
            result["controller_shadow"] = self.shadow_controller.suggest(result, previous_audio)
        if self.journal is not None:
            self.journal.record(result, command)
        logger.info("EEG decision: %s", result)
        return result

    def run(self, duration_seconds=30, step_seconds=2):
        if not np.isfinite(step_seconds) or step_seconds <= 0:
            raise ValueError("step_seconds must be positive and finite")
        if not self.muse.connect():
            print("Unable to connect to Muse.")
            return "connection_failed"
        print("Muse is about to start. Please get ready.", flush=True)
        print("Muse зараз почне запис. Будь ласка, підготуйтеся.", flush=True)
        try:
            for seconds in range(5, 0, -1):
                print(f"Starting in / Початок через: {seconds}", flush=True)
                time.sleep(1)
        except BaseException:
            self.muse.stop()
            raise
        if not self.muse.start():
            self.muse.stop()
            return "stream_start_failed"
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
        return "completed"
