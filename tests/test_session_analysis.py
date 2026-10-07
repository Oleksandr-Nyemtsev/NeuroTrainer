import json
from types import SimpleNamespace

import pytest

from src.data.session_journal import SessionJournal
from src.data.session_analysis import analyze_session, markdown_report
import analyze_session as cli
from src.core.session import NeuroSession


def make_journal(directory, audio=True):
    log=SessionJournal(directory,audio_enabled=audio,metadata={'synthetic_test':True})
    for i in range(5):
        start=8+i*3
        obs={'ready':True,'window_start_timestamp':start,'window_end_timestamp':start+2,
             'state':{'Alpha':20 if i==0 else 22,'Beta':10 if i==0 else 9},
             'action':{'master_volume':.02},'sleep':{'ready':False}}
        log.record(obs,{'started_at':start+2.1,'completed_at':start+2.11})
    log.close('completed')
    return log.path


def test_known_response_summary(tmp_path):
    report=analyze_session(make_journal(tmp_path))
    assert report['observations']==5 and report['valid_fraction']==1
    group=report['comparisons'][0]
    assert group['usable_fast_windows']==4
    assert group['band_change_percentage_points']=={'Alpha':2,'Beta':-1}
    assert group['mode']=='audio_commanded'
    assert 'caused by sound' in markdown_report(report)


def test_silence_never_reported_as_played_sound(tmp_path):
    report=analyze_session(make_journal(tmp_path,audio=False))
    assert all(g['mode']=='silent' for g in report['comparisons'])
    assert any('Silent session' in w for w in report['warnings'])


def test_incomplete_tail_and_corrupt_interior(tmp_path):
    path=make_journal(tmp_path)
    lines=path.read_text().splitlines()
    path.write_text('\n'.join(lines[:-1])+'\n{"type":')
    report=analyze_session(path)
    assert report['status']=='incomplete' and len(report['warnings'])==2
    path.write_text(lines[0]+'\nBAD\n'+lines[1]+'\n')
    with pytest.raises(ValueError,match='Invalid JSON'): analyze_session(path)


def test_tampered_links_and_overlapping_windows_are_excluded(tmp_path):
    path=make_journal(tmp_path)
    records=[json.loads(line) for line in path.read_text().splitlines()]
    records[2]['observation']['window_start_timestamp']=9  # Before action completed.
    records[3]['observation']['window_start_timestamp']=17
    records[3]['observation']['window_end_timestamp']=18
    records[3]['command']['started_at']=19
    records[4]['observation']['window_start_timestamp']=17  # Duplicate window, not evidence twice.
    records[4]['observation']['window_end_timestamp']=18
    path.write_text(''.join(json.dumps(r)+'\n' for r in records))
    report=analyze_session(path)
    assert report['comparisons'][0]['usable_fast_windows']==2
    assert report['comparisons'][0]['comparison_status']=='insufficient_data'
    assert any('invalid temporal' in w for w in report['warnings'])


def test_report_cli_outputs_and_preserves_source(tmp_path,monkeypatch):
    path=make_journal(tmp_path/'data/sessions')
    original=path.read_bytes()
    monkeypatch.setattr(cli,'ROOT',tmp_path)
    monkeypatch.setattr('sys.argv',['analyze_session.py','--latest'])
    cli.main()
    cli.main()
    reports=list((tmp_path/'data/sessions/reports').glob('*/report.json'))
    assert len(reports)==2 and path.read_bytes()==original
    assert all(json.loads(p.read_text())['observations']==5 for p in reports)
    assert all(p.with_suffix('.md').exists() for p in reports)


def test_empty_connection_failure_is_not_a_completed_experiment(tmp_path):
    log=SessionJournal(tmp_path,audio_enabled=False,metadata={})
    log.close('connection_failed')
    report=analyze_session(log.path)
    assert report['status']=='connection_failed' and report['valid_fraction'] is None
    assert not report['comparisons']
    session=NeuroSession(SimpleNamespace(connect=lambda:False),SimpleNamespace())
    assert session.run()=='connection_failed'


def test_duplicate_ids_rejected(tmp_path):
    path=make_journal(tmp_path)
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    rows[2]['id']=1
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError,match='IDs'): analyze_session(path)


@pytest.mark.parametrize('location,key,bad', [
    ('observation','sleep',None), ('observation','state',[]),
    ('observation','reason',[]), ('record','effective_targets',None),
    ('record','fast_window_after_action_id',[]), ('record','before_observation_id',{}),
])
def test_damaged_nested_fields_raise_readable_error(tmp_path,location,key,bad):
    path=make_journal(tmp_path)
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    target=rows[1]['observation'] if location=='observation' else rows[1]
    target[key]=bad
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    with pytest.raises(ValueError,match=key): analyze_session(path)
