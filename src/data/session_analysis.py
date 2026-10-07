"""Descriptive analysis of local journals; never a causal reward function."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean

BANDS = ('Delta', 'Theta', 'Alpha', 'Beta', 'Gamma')
STAGES = ('Wake', 'N1', 'N2', 'N3', 'REM')


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def load_journal(path):
    raw = Path(path).read_bytes()
    lines = raw.splitlines()
    records, warnings = [], []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except (ValueError, UnicodeError) as error:
            if index == len(lines)-1 and not raw.endswith(b'\n'):
                warnings.append('Incomplete final line ignored; recording may have been interrupted.')
                break
            raise ValueError(f'Invalid JSON at line {index+1}') from error
        if not isinstance(record, dict):
            raise ValueError(f'Expected JSON object at line {index+1}')
        records.append(record)
    if not records or records[0].get('type') != 'session_start' or records[0].get('schema_version') != 1:
        raise ValueError('Expected session_start with schema_version 1')
    if type(records[0].get('audio_enabled')) is not bool:
        raise ValueError('Missing audio mode')
    previous_id = 0
    end = None
    for record in records[1:]:
        if end is not None:
            raise ValueError('Records after session_end')
        if record.get('type') == 'session_end':
            end = record
        elif record.get('type') == 'observation':
            identifier = record.get('id')
            if type(identifier) is not int or identifier != previous_id+1:
                raise ValueError('Observation IDs must be consecutive and unique')
            previous_id = identifier
            if not isinstance(record.get('observation'), dict) or not isinstance(record.get('command'), dict):
                raise ValueError('Observation or command missing')
            if type(record['observation'].get('ready')) is not bool:
                raise ValueError('Observation ready flag must be boolean')
            if record.get('audio_enabled') != records[0]['audio_enabled']:
                raise ValueError('Audio mode changed inside journal')
        else:
            raise ValueError('Unexpected journal record type')
    if end is None:
        warnings.append('No session_end record; report covers only saved observations.')
    elif end.get('observations') != previous_id:
        raise ValueError('session_end observation count differs from journal')
    return records, warnings, hashlib.sha256(raw).hexdigest()


def valid_window(start, end):
    return finite(start) and finite(end) and start <= end


def analyze_session(path):
    records, warnings, digest = load_journal(path)
    header = records[0]
    rows = [r for r in records if r['type'] == 'observation']
    by_id = {r['id']: r for r in rows}
    valid = [r for r in rows if r['observation']['ready']]
    end = records[-1] if records[-1]['type'] == 'session_end' else None
    reasons = Counter(r['observation'].get('reason', 'Unspecified') for r in rows if not r['observation']['ready'])
    stages = Counter(r['observation']['sleep'].get('stage', 'Unknown') for r in valid
                     if r['observation'].get('sleep', {}).get('ready'))
    groups = {}
    for row in rows:
        if row.get('active_action_id') != row['id']:
            continue
        targets = row.get('effective_targets', {})
        volume = targets.get('master_volume')
        mode = 'silent' if not header['audio_enabled'] else ('muted' if volume == 0 else 'audio_commanded')
        groups[row['id']] = {'action_id': row['id'], 'mode': mode, 'targets': targets,
                             'before_observation_id': row.get('before_observation_id'),
                             'command_completed_at': row['command'].get('completed_at'),
                             'quality_segment': row.get('quality_segment'),
                             'fast': [], 'sleep': []}
    rejected_links = 0
    for row in valid:
        obs, cmd = row['observation'], row['command']
        for kind, field in (('fast', 'fast_window_after_action_id'), ('sleep', 'sleep_window_after_action_id')):
            link = row.get(field)
            if link is None:
                continue
            group = groups.get(link)
            source = obs if kind == 'fast' else obs.get('sleep', {})
            start = source.get('window_start_timestamp' if kind == 'fast' else 'window_start')
            finish = source.get('window_end_timestamp' if kind == 'fast' else 'window_end')
            # Independently recheck links; never trust a journal's association alone.
            safe = (group is not None and link < row['id']
                    and group['quality_segment'] == row.get('quality_segment')
                    and valid_window(start, finish)
                    and finite(group['command_completed_at']) and finite(cmd.get('started_at')))
            if safe:
                safe = group['command_completed_at'] <= start <= finish <= cmd['started_at']
                # Check no intervening change or invalid observation was omitted from the link.
                safe = safe and all(r['observation']['ready'] and r.get('active_action_id') == link
                                    for r in rows[link:row['id']-1])
            if not safe:
                rejected_links += 1
                continue
            group[kind].append((start, finish, row['id'], source))
    comparisons = []
    for group in groups.values():
        before_row = by_id.get(group['before_observation_id'])
        before = before_row['observation'] if before_row else {}
        before_end = before.get('window_end_timestamp')
        before_ok = (before.get('ready') and finite(before_end)
                     and finite(group['command_completed_at'])
                     and before_end <= group['command_completed_at'])
        def independent(windows):
            chosen, last_end = [], -math.inf
            for start, finish, identifier, source in sorted(windows):
                if start > last_end:
                    chosen.append((identifier, source))
                    last_end = finish
            return chosen
        fast, sleep = independent(group.pop('fast')), independent(group.pop('sleep'))
        deltas = {}
        if before_ok:
            baseline = before.get('state', {})
            for band in BANDS:
                after = [source.get('state', {}).get(band) for _, source in fast]
                after = [v for v in after if finite(v)]
                if finite(baseline.get(band)) and len(after) >= 3:
                    deltas[band] = mean(after) - baseline[band]
        group.update({'usable_fast_windows': len(fast), 'usable_sleep_windows': len(sleep),
                      'after_observation_ids': [identifier for identifier, _ in fast],
                      'band_change_percentage_points': deltas,
                      'comparison_status': 'descriptive_only' if deltas else 'insufficient_data'})
        comparisons.append(group)
    if rejected_links:
        warnings.append(f'{rejected_links} invalid temporal associations excluded.')
    if not rows:
        warnings.append('No EEG observations recorded; no response analysis is possible.')
    if not header['audio_enabled']:
        warnings.append('Silent session: cannot estimate a response to played audio.')
    return {'schema_version': 1, 'source_name': Path(path).name, 'source_sha256': digest,
            'audio_enabled': header['audio_enabled'], 'status': end.get('status') if end else 'incomplete',
            'observations': len(rows), 'valid_observations': len(valid),
            'rejected_observations': len(rows)-len(valid),
            'valid_fraction': len(valid)/len(rows) if rows else None,
            'rejection_reasons': dict(reasons), 'stage_prediction_counts': dict(stages),
            'comparisons': comparisons, 'warnings': warnings,
            'limitations': ['Associations do not demonstrate an effect caused by sound.',
                            'Stage counts are overlapping model predictions, not verified sleep durations.',
                            'Band differences are percentage points of relative band power, not absolute EEG amplitude.',
                            'Software target changes do not measure acoustic onset or loudness.']}


def markdown_report(report):
    lines = ['# NeuroTrainer session report', '', f"Source: `{report['source_name']}`",
             f"Status: {report['status']}", f"Audio enabled: {report['audio_enabled']}", '',
             f"Observations: {report['observations']}; usable: {report['valid_observations']}; rejected: {report['rejected_observations']}.", '',
             '## Data checks', '']
    lines += ['- ' + item for item in report['warnings']] or ['No additional warnings.']
    if report['rejection_reasons']:
        lines += ['', 'Rejections:', '']
        lines += [f'- {count}: {reason.replace(chr(10), " ")}' for reason, count in report['rejection_reasons'].items()]
    lines += ['', '## Changes after target commands', '',
              'At least three non-overlapping fast windows are required for a band comparison.', '',
              '| Action | Mode | Windows | Alpha change (pp) | Beta change (pp) | Status |',
              '|---|---|---:|---:|---:|---|']
    for row in report['comparisons']:
        delta = row['band_change_percentage_points']
        a, b = (f'{delta[k]:+.2f}' if k in delta else '—' for k in ('Alpha', 'Beta'))
        lines.append(f"| {row['action_id']} | {row['mode']} | {row['usable_fast_windows']} | {a} | {b} | {row['comparison_status']} |")
    lines += ['', '## Interpretation limits', ''] + ['- '+item for item in report['limitations']]
    return '\n'.join(lines)+'\n'
