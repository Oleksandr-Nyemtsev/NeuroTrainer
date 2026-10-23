import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.signal.personal_baseline import (BANDS, CHANNELS, CONDITIONS, METHOD,
    PersonalBaseline, build_profile, channel_features, condition_reference)


def eeg(n=512):
    t = np.arange(n)/256
    return np.array([12*np.sin(2*np.pi*(9+i*.5)*t)+5*np.sin(2*np.pi*6*t)
                     +3*np.sin(2*np.pi*20*t) for i in range(4)])


def profile():
    features = channel_features(eeg())
    return {'schema_version':1,'role':'diagnostic_only','method':copy.deepcopy(METHOD),
            'conditions':{c:{'accepted_windows':20,'channels':copy.deepcopy(features)} for c in CONDITIONS}}


def source(tmp_path,condition, *, n=20, audio=False):
    journal = tmp_path/(condition+'.jsonl')
    raw_dir = journal.with_suffix('.raw'); raw_dir.mkdir()
    ts = 1000+np.arange(512*n)/256
    board = np.zeros((8,len(ts)))
    board[1:5] = np.tile(eeg(),(1,n))
    board[6] = ts
    np.save(raw_dir/'chunk_000001.npy',board,allow_pickle=False)
    metadata = {'journal':journal.name,'condition':condition,'sampling_rate':256,
                'eeg_units':'microvolts','audio_enabled':audio,
                'board_descriptor':{'eeg_names':','.join(CHANNELS),'eeg_channels':[1,2,3,4],'timestamp_channel':6}}
    manifest = [{'type':'start','metadata':metadata},
                {'type':'read','observation_id':1,'file':'chunk_000001.npy'},
                {'type':'end','status':'completed'}]
    (raw_dir/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in manifest))
    rows = [{'type':'session_start','audio_enabled':audio,'metadata':{
        'condition':condition,'condition_source':'self_report','sampling_rate':256,'channels':list(CHANNELS)}}]
    for i in range(n):
        rows.append({'type':'observation','id':i+1,'observation':{
            'ready':True,'window_start_timestamp':ts[i*512],'window_end_timestamp':ts[(i+1)*512-1],
            'sleep':{'ready':True,'stage':'REM'}}})  # Model output must never label the baseline.
    rows.append({'type':'session_end','status':'completed'})
    journal.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return journal


def test_scale_preserves_percent_but_quadruples_power():
    baseline = PersonalBaseline(profile())
    result = baseline.compare(2*eeg())
    for channel in result['channels'].values():
        for band in channel.values():
            assert band['delta_pp'] == pytest.approx(0,abs=.001)
            assert band['power_ratio'] == pytest.approx(4,abs=.001)
    assert result['role'] == 'diagnostic_only'
    assert not {'stage','score','action'}.intersection(result)


def test_zero_reference_power_is_not_divided_and_profile_is_immutable():
    p = profile()
    p['conditions'][CONDITIONS[0]]['channels']['TP9']['Theta']['power_uv2'] = 0
    baseline = PersonalBaseline(p)
    before = copy.deepcopy(baseline.profile)
    assert baseline.compare(eeg())['channels']['TP9']['Theta']['power_ratio'] is None
    p['conditions'][CONDITIONS[0]]['channels']['TP9']['Theta']['power_uv2'] = 999
    assert baseline.profile == before
    baseline.compare(eeg())
    assert baseline.profile == before


@pytest.mark.parametrize('kind',['shape','nan','flat','amplitude'])
def test_bad_live_input_cannot_produce_baseline_values(kind):
    x = eeg()
    if kind == 'shape': x = x[:,:-1]
    elif kind == 'nan': x[0,0] = np.nan
    elif kind == 'flat': x[0] = 0
    else: x[0] *= 30
    with pytest.raises(ValueError): PersonalBaseline(profile()).compare(x)


@pytest.mark.parametrize('kind',['method','channels','condition','count','nonfinite'])
def test_incompatible_profiles_are_rejected(kind):
    p = profile()
    if kind == 'method': p['method']['fs'] = 128
    elif kind == 'channels': del p['conditions'][CONDITIONS[0]]['channels']['TP9']
    elif kind == 'condition': del p['conditions'][CONDITIONS[0]]
    elif kind == 'count': p['conditions'][CONDITIONS[0]]['accepted_windows'] = 19
    else: p['conditions'][CONDITIONS[0]]['channels']['TP9']['Alpha']['power_uv2'] = float('nan')
    with pytest.raises(ValueError): PersonalBaseline(p)


def test_raw_build_uses_conditions_not_predictions_and_refuses_overwrite(tmp_path,monkeypatch):
    import build_personal_baseline as cli
    opened = source(tmp_path,CONDITIONS[0])
    closed = source(tmp_path,CONDITIONS[1])
    before = opened.read_bytes()
    p = build_profile(opened,closed)
    assert set(p['conditions']) == set(CONDITIONS)
    for condition in CONDITIONS:
        reference = p['conditions'][condition]
        assert reference['accepted_windows'] == 20
        assert len(reference['source']['ordered_raw_chunk_hashes_sha256']) == 64
        for name in CHANNELS:
            assert set(reference['channels'][name]) == set(BANDS)
    output = tmp_path/'baseline.json'
    monkeypatch.setattr('sys.argv',['build_personal_baseline.py','--eyes-open',str(opened),
        '--eyes-closed',str(closed),'--output',str(output)])
    cli.main()
    loaded = PersonalBaseline.load(output,reference=CONDITIONS[1])
    assert loaded.reference == CONDITIONS[1] and len(loaded.profile_sha256)==64
    saved = output.read_bytes()
    with pytest.raises(SystemExit): cli.main()
    assert output.read_bytes()==saved and opened.read_bytes()==before


@pytest.mark.parametrize('problem',['condition','audio','incomplete','few','window'])
def test_invalid_recordings_cannot_become_baselines(tmp_path,problem):
    p = source(tmp_path,CONDITIONS[0],n=19 if problem=='few' else 20,audio=problem=='audio')
    rows = [json.loads(s) for s in p.read_text().splitlines()]
    if problem == 'incomplete': rows[-1]['status'] = 'interrupted'
    if problem == 'window': rows[1]['observation']['window_end_timestamp'] += .001
    p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError): condition_reference(p,CONDITIONS[1] if problem=='condition' else CONDITIONS[0])


def test_live_baseline_does_not_change_audio_and_clears_on_fault(monkeypatch):
    import src.core.session as module
    monkeypatch.setattr(module.time,'time',lambda:1000.)
    monkeypatch.setattr(module,'BoardShim',SimpleNamespace(get_eeg_channels=lambda _: [0,1,2,3],get_timestamp_channel=lambda _:4))
    data = np.vstack([eeg(),1000-(511-np.arange(512))/256])
    journals,actions,sessions = [],[],[]
    for baseline in (None,PersonalBaseline(profile())):
        entries,commands = [],[]
        journals.append(entries); actions.append(commands)
        sessions.append(module.NeuroSession(SimpleNamespace(get_data=lambda:data.copy()),
            SimpleNamespace(apply_action=commands.append,target_master_volume=.1),
            journal=SimpleNamespace(record=lambda result,command,entries=entries:entries.append(copy.deepcopy(result))),
            personal_baseline=baseline))
    plain,compared = [s.run_step() for s in sessions]
    assert plain['ready'] and compared['ready']
    assert plain['action']==compared['action'] and plain['state']==compared['state']
    assert 'personal_baseline' not in plain
    assert compared['personal_baseline']['ready']
    assert journals[1][-1]['personal_baseline'] == compared['personal_baseline']
    sessions[1].muse.get_data = lambda:None
    fault = sessions[1].run_step()
    assert not fault['personal_baseline']['ready'] and 'channels' not in fault['personal_baseline']
    assert actions[1][-1] == {'master_volume':0.}


def test_diagnostic_error_does_not_change_existing_audio_decision(monkeypatch):
    import src.core.session as module
    monkeypatch.setattr(module.time,'time',lambda:1000.)
    monkeypatch.setattr(module,'BoardShim',SimpleNamespace(get_eeg_channels=lambda _: [0,1,2,3],get_timestamp_channel=lambda _:4))
    class Broken:
        reference = CONDITIONS[0]
        def compare(self,eeg): raise ValueError('diagnostic failure')
    data = np.vstack([eeg(),1000-(511-np.arange(512))/256])
    def session(baseline):
        return module.NeuroSession(SimpleNamespace(get_data=lambda:data),SimpleNamespace(apply_action=lambda _:None),personal_baseline=baseline)
    plain,broken = session(None).run_step(),session(Broken()).run_step()
    assert broken['ready'] and broken['action']==plain['action']
    assert not broken['personal_baseline']['ready']
    with pytest.raises(ValueError,match='256'): module.NeuroSession(None,None,fs=128,personal_baseline=Broken())
