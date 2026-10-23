"""Observe Muse EEG packet IDs without changing acceptance or reconstructing EEG.

BrainFlow 5.23.0 muse.cpp exposes a 16-bit ID repeated for 12 samples.
The exposed ID is not a separate continuity guarantee for every EEG channel.
"""
import numpy as np


class MusePacketDiagnostics:
    def __init__(self):
        self.last_id = None
        self.run_samples = 0
        self.run_has_start = False

    def update(self, values):
        values = np.asarray(values)
        if (values.ndim != 1 or not values.size or not np.isfinite(values).all()
                or np.any(values != np.floor(values)) or np.any(values < 0)
                or np.any(values > 65535)):
            self.__init__()
            return {"available": False, "reason": "Invalid packet ID values"}
        ids = values.astype(np.int64)
        previous = self.last_id
        result = {"available": True, "received_samples": int(ids.size),
                  "first_id": int(ids[0]), "last_id": int(ids[-1]),
                  "previous_id": previous, "expected_samples_per_packet": 12,
                  "same_id_steps": 0, "consecutive_packet_steps": 0,
                  "wrap_count": 0, "forward_jump_count": 0,
                  "suspected_missing_packets": 0, "backward_or_reset_count": 0,
                  "completed_packet_runs": 0, "unexpected_run_lengths": 0,
                  "jump_examples": []}
        for raw in ids:
            current = int(raw)
            if self.last_id is None:
                self.run_samples = 1
                self.run_has_start = False  # First read may start midway through a packet.
            elif current == self.last_id:
                result["same_id_steps"] += 1
                self.run_samples += 1
            else:
                if self.run_has_start:
                    result["completed_packet_runs"] += 1
                    result["unexpected_run_lengths"] += int(self.run_samples != 12)
                delta = (current - self.last_id) % 65536
                if delta == 1:
                    result["consecutive_packet_steps"] += 1
                    result["wrap_count"] += int(current < self.last_id)
                elif delta < 32768:
                    result["forward_jump_count"] += 1
                    result["suspected_missing_packets"] += delta - 1
                else:
                    result["backward_or_reset_count"] += 1
                if delta != 1 and len(result["jump_examples"]) < 5:
                    result["jump_examples"].append({"from": self.last_id, "to": current,
                                                    "modulo_step": delta})
                self.run_samples = 1
                self.run_has_start = True
            self.last_id = current
        result["open_run_samples"] = self.run_samples
        result["open_run_exceeds_expected"] = self.run_samples > 12
        return result


def packet_sequence_usable(diagnostic):
    """Evidence of regular exposed IDs; not a guarantee for every EEG channel."""
    return (diagnostic.get("available") is True
            and diagnostic.get("received_samples", 0) > 0
            and not any(diagnostic.get(k, 0) for k in (
                "forward_jump_count", "backward_or_reset_count",
                "unexpected_run_lengths", "open_run_exceeds_expected")))


def timestamp_interval_limit(fs, packet_evidence=False):
    # Bounded allowance for host-interpolated Muse times. Never rewrite timestamps.
    return (2.0 if packet_evidence else 1.5) / fs
