"""Opt-in local BrainFlow reads, before filtering/rejection, for offline replay."""
import json
import time
from pathlib import Path

import numpy as np


class RawSessionRecorder:
    def __init__(self, directory, *, metadata):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.handle = (self.directory / 'manifest.jsonl').open('x', encoding='utf-8')
        self.sequence = 0
        self._write({'type': 'start', 'schema_version': 1, 'metadata': metadata,
                     'note': 'Unmodified BrainFlow reads, including rejected data. Condition is self-reported, not sleep ground truth.'})

    def _write(self, item):
        self.handle.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + '\n')
        self.handle.flush()

    def record(self, data):
        self.sequence += 1
        filename = None
        if data is not None:
            data = np.asarray(data)
            if data.dtype.hasobject:
                raise ValueError('Raw capture requires numeric BrainFlow data')
            filename = f'chunk_{self.sequence:06d}.npy'
            # Exclusive write and no pickle. Keep precision, shape and even NaNs.
            with (self.directory / filename).open('xb') as handle:
                np.save(handle, data, allow_pickle=False)
        self._write({'type': 'read', 'observation_id': self.sequence,
                     'saved_at': time.time(), 'file': filename})

    def close(self, status):
        if not self.handle.closed:
            try:
                self._write({'type': 'end', 'status': status, 'reads': self.sequence})
            finally:
                self.handle.close()


def iter_raw_reads(directory):
    """Return (read metadata, original board matrix or None), never play/connect."""
    directory = Path(directory)
    with (directory / 'manifest.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            item = json.loads(line)
            if item['type'] != 'read':
                continue
            filename = item['file']
            if filename is not None:
                if filename != f"chunk_{item['observation_id']:06d}.npy":
                    raise ValueError('Invalid raw chunk filename')
                data = np.load(directory / filename, allow_pickle=False)
            else:
                data = None
            yield item, data
