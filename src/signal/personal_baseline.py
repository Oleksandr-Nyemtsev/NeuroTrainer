"""Personal EEG references, diagnostic only. No stage labels or audio decisions."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

from src.data.raw_session import iter_raw_reads
from src.signal.muse_quality import MuseSignalQuality

CHANNELS = ('TP9', 'AF7', 'AF8', 'TP10')
CONDITIONS = ('awake-eyes-open', 'awake-eyes-closed')
BANDS = {'Theta': (4, 8), 'Alpha': (8, 13), 'Beta': (13, 30)}
MIN_WINDOWS = 20  # >=40 seconds of accepted, non-overlapping input; not clinical validation.
METHOD = {'version': 1, 'fs': 256, 'samples': 512, 'units': 'microvolts',
          'filter': 'butter4_1_40Hz_sosfiltfilt_per_window',
          'psd': 'welch_hann_nperseg256_noverlap128_detrend_constant_density',
          'bands': {k: list(v) for k,v in BANDS.items()},
          'integration': 'sum_psd_times_df_lower_inclusive_upper_exclusive',
          'relative_denominator_hz': [1, 40], 'channel_order': list(CHANNELS),
          'summary': 'per_channel_per_condition_median', 'power_ratio_min_denominator_uv2': 1e-6}
_SOS = butter(4, [1,40], btype='bandpass', fs=256, output='sos')


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def channel_features(eeg_uv):
    """Absolute µV² and relative percent for raw (4,512), Muse 256 Hz."""
    x = np.asarray(eeg_uv, dtype=np.float64)
    if x.shape != (4,512) or not np.isfinite(x).all():
        raise ValueError('Baseline requires finite raw Muse (4,512) at 256 Hz in microvolts')
    quality = MuseSignalQuality(fs=256).analyze(x)
    if any(c['quality'] < .75 for c in quality['channels']):
        raise ValueError('Baseline input failed channel quality gate')
    signal = sosfiltfilt(_SOS, x, axis=-1)
    f,p = welch(signal, fs=256, window='hann', nperseg=256, noverlap=128,
                detrend='constant', scaling='density', axis=-1)
    df = f[1]-f[0]
    total = p[:,(f>=1)&(f<40)].sum(-1)*df
    if not np.isfinite(p).all() or np.any(total <= 1e-12):
        raise ValueError('No usable band power for baseline comparison')
    result = {}
    for index,name in enumerate(CHANNELS):
        result[name] = {}
        for band,(low,high) in BANDS.items():
            power = float(p[index,(f>=low)&(f<high)].sum()*df)
            result[name][band] = {'power_uv2': power, 'relative_percent': float(100*power/total[index])}
    return result


def condition_reference(journal_path, expected_condition):
    """Build from independently labelled, silent, completed raw/journal pairs."""
    journal_path = Path(journal_path)
    rows = [json.loads(line) for line in journal_path.read_text(encoding='utf-8').splitlines()]
    if (not rows or rows[0].get('type') != 'session_start'
            or rows[-1].get('type') != 'session_end' or rows[-1].get('status') != 'completed'):
        raise ValueError('Baseline source must be a completed session journal')
    metadata = rows[0]['metadata']
    if (metadata.get('condition') != expected_condition or metadata.get('condition_source') != 'self_report'
            or rows[0].get('audio_enabled') is not False or metadata.get('sampling_rate') != 256
            or metadata.get('channels') != list(CHANNELS)):
        raise ValueError('Wrong baseline condition, units/channel metadata or audio enabled')
    raw_dir = journal_path.with_suffix('.raw')
    manifest = raw_dir/'manifest.jsonl'
    raw_rows = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines()]
    if (not raw_rows or raw_rows[0].get('type') != 'start' or raw_rows[-1].get('type') != 'end'
            or raw_rows[-1].get('status') != 'completed'):
        raise ValueError('Baseline source must have completed raw capture')
    raw_meta = raw_rows[0]['metadata']
    desc = raw_meta['board_descriptor']
    if (raw_meta.get('journal') != journal_path.name or raw_meta.get('condition') != expected_condition
            or raw_meta.get('sampling_rate') != 256 or raw_meta.get('eeg_units') != 'microvolts'
            or raw_meta.get('audio_enabled') is not False or desc.get('eeg_names') != ','.join(CHANNELS)):
        raise ValueError('Raw source metadata differs from journal/expected Muse input')
    data = []
    chunk_digest = hashlib.sha256()
    for read,chunk in iter_raw_reads(raw_dir):
        if read['file'] is not None:
            chunk_digest.update(read['file'].encode())
            chunk_digest.update(bytes.fromhex(file_hash(raw_dir/read['file'])))
        if chunk is not None:
            if chunk.ndim != 2:
                raise ValueError('Invalid raw matrix')
            if chunk.shape[1]:
                data.append(chunk)
    if not data:
        raise ValueError('No raw data for baseline')
    board = np.concatenate(data,axis=1)
    ts = board[desc['timestamp_channel']]
    eeg = board[desc['eeg_channels']]
    if not np.isfinite(ts).all() or np.any(np.diff(ts)<=0) or eeg.shape[0] != 4:
        raise ValueError('Invalid raw timestamps/channels')
    features, rejected = [], []
    previous_end = -np.inf
    observed = [row for row in rows if row['type']=='observation']
    original_ready = 0
    for row in observed:
        observation = row['observation']
        if observation.get('ready') is not True:
            continue
        original_ready += 1
        start,end = observation['window_start_timestamp'],observation['window_end_timestamp']
        if not np.isfinite([start,end]).all() or start > end:
            raise ValueError('Invalid journal window times')
        if start <= previous_end:
            rejected.append({'id':row['id'],'reason':'Overlapping fast window'})
            continue
        begin,stop = int(np.searchsorted(ts,start)),int(np.searchsorted(ts,end))
        if begin>=len(ts) or stop>=len(ts) or ts[begin]!=start or ts[stop]!=end or stop-begin+1!=512:
            raise ValueError('Accepted journal window does not exactly match raw samples')
        evidence = observation.get('timestamp_diagnostics',{}).get('packet_evidence') is True
        if np.any(np.diff(ts[begin:stop+1]) > (2 if evidence else 1.5)/256):
            rejected.append({'id':row['id'],'reason':'Timestamp gap'})
            continue
        try:
            features.append(channel_features(eeg[:,begin:stop+1]))
        except ValueError as error:
            rejected.append({'id':row['id'],'reason':str(error)})
            continue
        previous_end = end
    if len(features) < MIN_WINDOWS:
        raise ValueError(f'Insufficient baseline: {len(features)} accepted non-overlapping windows; need {MIN_WINDOWS}')
    channels = {name:{band:{key:float(np.median([f[name][band][key] for f in features]))
                             for key in ('power_uv2','relative_percent')} for band in BANDS} for name in CHANNELS}
    return {'channels':channels,'accepted_windows':len(features),'accepted_seconds_nominal':2*len(features),
            'total_observations':len(observed),'original_ready_windows':original_ready,'extra_rejections':rejected,
            'source':{'journal':journal_path.name,'condition_source':'self_report',
                      'journal_sha256':file_hash(journal_path),'raw_manifest_sha256':file_hash(manifest),
                      'ordered_raw_chunk_hashes_sha256':chunk_digest.hexdigest()}}


def build_profile(eyes_open, eyes_closed):
    profile = {'schema_version':1,'role':'diagnostic_only','method':copy.deepcopy(METHOD),
               'conditions':{condition:condition_reference(path,condition) for condition,path in
                             zip(CONDITIONS,(eyes_open,eyes_closed))},
               'limitations':'Self-reported awake references; no sleep, meditation, ADHD or recovery validation. No stage thresholds or audio control.'}
    validate_profile(profile)
    return profile


def validate_profile(profile):
    if (profile.get('schema_version') != 1 or profile.get('role') != 'diagnostic_only'
            or profile.get('method') != METHOD or set(profile.get('conditions',{})) != set(CONDITIONS)):
        raise ValueError('Unsupported personal baseline profile/method')
    for reference in profile['conditions'].values():
        count = reference.get('accepted_windows')
        if type(count) is not int or count < MIN_WINDOWS or set(reference.get('channels',{})) != set(CHANNELS):
            raise ValueError('Incomplete baseline conditions/channels')
        for channel in reference['channels'].values():
            if set(channel) != set(BANDS):
                raise ValueError('Incomplete baseline bands')
            for values in channel.values():
                power,percent = values['power_uv2'],values['relative_percent']
                if (not np.isfinite([power,percent]).all() or power<0 or not 0<=percent<=100):
                    raise ValueError('Invalid baseline power/reference percentage')


class PersonalBaseline:
    def __init__(self, profile, reference='awake-eyes-open', profile_sha256=None):
        validate_profile(profile)
        if reference not in CONDITIONS:
            raise ValueError('Unsupported baseline reference condition')
        self.profile = copy.deepcopy(profile)
        self.reference = reference
        self.profile_sha256 = profile_sha256

    @classmethod
    def load(cls, path, reference='awake-eyes-open'):
        payload = Path(path).read_bytes()
        return cls(json.loads(payload.decode('utf-8')),reference,hashlib.sha256(payload).hexdigest())

    def compare(self, eeg_uv):
        current = channel_features(eeg_uv)
        reference = self.profile['conditions'][self.reference]['channels']
        changes = {}
        for name in CHANNELS:
            changes[name] = {}
            for band in BANDS:
                value,base = current[name][band],reference[name][band]
                changes[name][band] = {
                    'delta_pp':round(value['relative_percent']-base['relative_percent'],3),
                    'power_ratio':round(value['power_uv2']/base['power_uv2'],3) if base['power_uv2']>1e-6 else None}
        return {'ready':True,'role':'diagnostic_only','reference':self.reference,
                'profile_sha256':self.profile_sha256,'channels':changes}
