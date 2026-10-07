"""Local observation/action timeline. Associations are not causal rewards."""
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class SessionJournal:
    def __init__(self, directory, *, audio_enabled, metadata):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:8]
        self.path = directory / (name + '.jsonl')
        self.handle = self.path.open('x', encoding='utf-8')
        self.audio_enabled = audio_enabled
        self.sequence = 0
        self.active = None
        self.targets = {"master_volume": 0.0}
        self.quality_segment = 0
        self.previous_valid = False
        self._write({"type": "session_start", "schema_version": 1,
                     "unix_time": time.time(), "audio_enabled": audio_enabled,
                     "metadata": metadata,
                     "limitations": "Command times and software levels, not measured acoustic onset. No causal reward or sleep ground truth."})

    def _write(self, record):
        self.handle.write(json.dumps(clean(record), ensure_ascii=False, allow_nan=False) + '\n')
        self.handle.flush()

    def record(self, observation, command):
        self.sequence += 1
        valid = bool(observation.get('ready'))
        if not valid or not self.previous_valid:
            self.quality_segment += 1
        previous = self.active
        proposed = observation['action']
        effective = {**self.targets, **proposed} if self.audio_enabled else {"master_volume": 0.0}
        changed = self.active is None or effective != self.targets
        new_segment = self.active is not None and self.active['quality_segment'] != self.quality_segment
        if changed or new_segment:
            self.active = {"id": self.sequence, "completed_at": command['completed_at'],
                           "targets": effective.copy(), "quality_segment": self.quality_segment,
                           "before_observation_id": self.sequence if valid else None}
        self.targets = effective
        # A window may straddle a change. Preserve both timestamps and reject
        # association unless the whole window follows the previous command.
        def association(start, end):
            if (not valid or previous is None or start is None or end is None
                    or previous['quality_segment'] != self.quality_segment
                    or not previous['completed_at'] <= start <= end <= command['started_at']):
                return None
            return previous['id']
        fast_id = association(observation.get('window_start_timestamp'), observation.get('window_end_timestamp'))
        sleep = observation.get('sleep', {})
        sleep_id = association(sleep.get('window_start'), sleep.get('window_end')) if sleep.get('ready') else None
        self._write({"type": "observation", "id": self.sequence, "unix_time": time.time(),
                     "quality_segment": self.quality_segment, "observation": observation,
                     "command": command, "audio_enabled": self.audio_enabled,
                     "requested_action": proposed, "effective_targets": effective,
                     "action_changed": changed, "active_action_id": self.active['id'],
                     "before_observation_id": self.active['before_observation_id'],
                     "fast_window_after_action_id": fast_id,
                     "sleep_window_after_action_id": sleep_id})
        self.previous_valid = valid

    def close(self, status):
        if not self.handle.closed:
            try:
                self._write({"type": "session_end", "unix_time": time.time(),
                             "status": status, "observations": self.sequence})
            finally:
                self.handle.close()
