import json
from types import SimpleNamespace
import numpy as np
import pytest
from src.data.raw_session import RawSessionRecorder, iter_raw_reads


def test_roundtrip_preserves_rejected_values_and_empty_reads(tmp_path):
    path = tmp_path / 'capture'
    recorder = RawSessionRecorder(path, metadata={'condition': 'awake-eyes-open'})
    data = np.arange(8*24,dtype=np.float64).reshape(8,24)
    data[1,0] = np.nan
    recorder.record(data)
    recorder.record(None)
    recorder.record(np.empty((8,0)))
    recorder.close('interrupted')
    recorder.close('completed')
    reads = list(iter_raw_reads(path))
    np.testing.assert_array_equal(reads[0][1],data)
    assert reads[0][1].dtype == data.dtype
    assert reads[1][1] is None and reads[2][1].shape == (8,0)
    rows = [json.loads(s) for s in (path/'manifest.jsonl').read_text().splitlines()]
    assert rows[-1]['status'] == 'interrupted'
    with pytest.raises(FileExistsError): RawSessionRecorder(path, metadata={})


def test_capture_precedes_quality_rejection_and_matches_journal(tmp_path, monkeypatch):
    import src.core.session as module
    from src.data.session_journal import SessionJournal
    monkeypatch.setattr(module.time,'time',lambda:1000.)
    monkeypatch.setattr(module,'BoardShim',SimpleNamespace(
        get_eeg_channels=lambda _: [0,1,2,3],get_timestamp_channel=lambda _:4))
    data=np.zeros((5,512))
    data[4]=1000-(511-np.arange(512))/256
    raw=RawSessionRecorder(tmp_path/'capture',metadata={})
    journal=SessionJournal(tmp_path/'logs',audio_enabled=False,metadata={})
    muse=SimpleNamespace(get_data=lambda:data)
    session=module.NeuroSession(muse,SimpleNamespace(apply_action=lambda _:None),
                                raw_recorder=raw,journal=journal)
    assert not session.run_step()['ready']  # Flat EEG is saved but remains rejected.
    muse.get_data=lambda:None
    session.run_step()
    raw.close('completed');journal.close('completed')
    reads=list(iter_raw_reads(raw.directory))
    rows=[json.loads(s) for s in journal.path.read_text().splitlines() if json.loads(s)['type']=='observation']
    assert [r[0]['observation_id'] for r in reads] == [r['id'] for r in rows]
    np.testing.assert_array_equal(reads[0][1],data)


def test_capture_write_failure_stops_audio_and_device(monkeypatch):
    import src.core.session as module
    monkeypatch.setattr(module.time,'sleep',lambda _:None)
    stopped=[]
    class Broken:
        def record(self,data): raise OSError('disk full')
    muse=SimpleNamespace(connect=lambda:True,start=lambda:True,get_data=lambda:None,
                         stop=lambda:stopped.append('muse'))
    audio=SimpleNamespace(start=lambda:None,stop=lambda:stopped.append('audio'))
    session=module.NeuroSession(muse,audio,raw_recorder=Broken())
    with pytest.raises(OSError,match='disk full'):session.run(duration_seconds=1)
    assert stopped==['audio','muse']
