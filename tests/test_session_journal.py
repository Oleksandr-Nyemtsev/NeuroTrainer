import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.data.session_journal import SessionJournal
import src.core.session as session_module


def read(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def observation(start, end, volume=.02, ready=True):
    return {'ready':ready, 'window_start_timestamp':start, 'window_end_timestamp':end,
            'action':{'master_volume':volume}, 'state':{'Alpha':20.,'Beta':10.},
            'sleep':{'ready':True,'window_start':end-30,'window_end':end}}


def command(t):
    return {'started_at':t, 'completed_at':t+.01, 'software_current_parameters':{}}


def test_before_after_link_rejects_straddling_and_invalid_windows(tmp_path):
    log=SessionJournal(tmp_path,audio_enabled=True,metadata={})
    log.record(observation(8,10),command(10.1))
    log.record(observation(10,12),command(12.1))  # Straddles the audio command.
    log.record(observation(12,14),command(14.1))
    log.record(observation(14,16,0,False),command(16.1))
    log.record(observation(16,18),command(18.1))
    log.close('returned')
    rows=read(log.path)
    assert rows[1]['before_observation_id']==1
    assert rows[2]['fast_window_after_action_id'] is None
    assert rows[3]['fast_window_after_action_id']==1
    assert rows[3]['sleep_window_after_action_id'] is None
    assert rows[4]['fast_window_after_action_id'] is None
    assert rows[5]['fast_window_after_action_id'] is None
    assert rows[-1]['observations']==5


def test_silent_run_records_proposal_as_not_played_and_preserves_files(tmp_path):
    log=SessionJournal(tmp_path,audio_enabled=False,metadata={})
    log.record(observation(0,2,.08),command(2.1))
    log.close('interrupted')
    first=log.path.read_bytes()
    other=SessionJournal(tmp_path,audio_enabled=False,metadata={})
    other.close('returned')
    assert other.path != log.path and log.path.read_bytes()==first
    row=read(log.path)[1]
    assert row['requested_action']['master_volume']==.08
    assert row['effective_targets']=={'master_volume':0.}
    assert row['audio_enabled'] is False


def test_stream_integration_records_valid_and_rejected_windows(tmp_path,monkeypatch):
    now=[1000.]
    monkeypatch.setattr(session_module.time,'time',lambda:now[0])
    monkeypatch.setattr(session_module,'BoardShim',SimpleNamespace(
        get_eeg_channels=lambda _: [0,1,2,3],get_timestamp_channel=lambda _:4))
    x=np.tile(20*np.sin(2*np.pi*10*np.arange(512)/256),(4,1))
    data=np.vstack([x,1000-(511-np.arange(512))/256])
    muse=SimpleNamespace(get_data=lambda:data)
    applied=[]
    log=SessionJournal(tmp_path,audio_enabled=True,metadata={})
    session=session_module.NeuroSession(muse,SimpleNamespace(apply_action=applied.append),journal=log)
    session.run_step()
    muse.get_data=lambda:None
    now[0]+=2
    session.run_step()
    log.close('returned')
    rows=read(log.path)
    assert rows[1]['observation']['ready']
    assert rows[1]['command']['started_at']==1000
    assert rows[2]['observation']['reason']=='Waiting for EEG'
    assert rows[2]['effective_targets']['master_volume']==0
    assert len(applied)==2


def test_journal_failure_stops_session_and_releases_resources(monkeypatch):
    stopped=[]
    class Broken:
        def record(self,*args): raise OSError('disk full')
    muse=SimpleNamespace(connect=lambda:True,start=lambda:True,get_data=lambda:None,
                         stop=lambda:stopped.append('muse'))
    audio=SimpleNamespace(start=lambda:None,stop=lambda:stopped.append('audio'),apply_action=lambda _:None)
    session=session_module.NeuroSession(muse,audio,journal=Broken())
    with pytest.raises(OSError,match='disk full'): session.run(duration_seconds=2)
    assert stopped==['audio','muse']
