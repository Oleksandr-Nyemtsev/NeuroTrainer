from types import SimpleNamespace
import numpy as np
import pytest

from src.core.sleep_observer import SleepObserver, prepare_sleep_input
from src.core.bounded_audio import BoundedAudioPolicy
import src.core.session as module


def eeg(n=512):
    t = np.arange(n)/256
    return np.array([20*np.sin(2*np.pi*f*t) for f in (8,9,10,11)])


def model():
    calls = []
    def predict(x):
        calls.append(x)
        assert x.shape == (2,3000) and x.dtype == np.float32
        return dict(Wake=.9,N1=.025,N2=.025,N3=.025,REM=.025)
    return SimpleNamespace(predict=predict), calls


def test_preprocessing_and_invalid_input():
    x = prepare_sleep_input(eeg(7680))
    assert x.shape == (2,3000)
    np.testing.assert_allclose(x.mean(-1), 0, atol=1e-6)
    np.testing.assert_allclose(x.std(-1), 1, atol=1e-6)
    for bad in (eeg(512), np.zeros((4,7680)), eeg(7680)*np.nan, eeg(7680)*100):
        with pytest.raises(ValueError): prepare_sleep_input(bad)


def test_rolling_context_and_two_second_updates():
    m,calls = model()
    observer = SleepObserver(m)
    for i in range(16):
        ts = (np.arange(512)+i*512)/256
        result = observer.update(eeg(),ts)
        assert result['ready'] == (i>=14)
    assert len(calls)==2
    assert observer.buffer.shape==(4,7680)
    assert result['window_start']==2
    with pytest.raises(ValueError,match='gap'):
        observer.update(eeg(),ts+3)
    assert observer.buffer.shape[1]==0 and not observer.result['ready']


def test_faults_reset_previous_prediction():
    for kind in ('nan','artifact','probability'):
        m,_ = model()
        obs=SleepObserver(m)
        obs.update(eeg(7680),np.arange(7680)/256)
        data=eeg()
        if kind=='nan': data[1,10]=np.nan
        elif kind=='artifact': data[1,10]=600
        else: m.predict=lambda x: dict(Wake=np.nan,N1=0,N2=0,N3=0,REM=0)
        with pytest.raises(ValueError): obs.update(data,30+np.arange(512)/256)
        assert not obs.result['ready']


def test_audio_bounds_cadence_and_sleep_attenuation():
    policy=BoundedAudioPolicy()
    awake={'ready':True,'probabilities':dict(Wake=1,N1=0,N2=0,N3=0,REM=0)}
    asleep={'ready':True,'probabilities':dict(Wake=0,N1=0,N2=1,N3=0,REM=0)}
    previous=None
    for t in range(0,1200,2):
        action=policy.choose(0 if t<600 else 10,awake,t)
        assert 0<=action['master_volume']<=.1
        assert 6<=action['beat_frequency']<=10
        assert action['binaural_volume']==.01 and action['wave_volume']==.05
        if previous is not None:
            assert abs(action['master_volume']-previous['master_volume'])<=.0100001
            assert abs(action['beat_frequency']-previous['beat_frequency'])<=.500001
            if t%30: assert action==previous
        previous=action
    for t in range(1200,1560,2): action=policy.choose(10,asleep,t)
    assert action['master_volume']==pytest.approx(.03)
    assert policy.choose(10,{'ready':False},1560)=={'master_volume':0.0}
    assert policy.sleep_windows==0


def test_live_integration_retains_fast_analysis_and_mutes_fault(monkeypatch):
    now=[2000.]
    monkeypatch.setattr(module.time,'time',lambda:now[0])
    monkeypatch.setattr(module,'BoardShim',SimpleNamespace(get_eeg_channels=lambda _: [0,1,2,3],get_timestamp_channel=lambda _:4))
    m,calls=model()
    actions=[]
    muse=SimpleNamespace()
    session=module.NeuroSession(muse,SimpleNamespace(apply_action=actions.append,target_master_volume=0),
                               sleep_observer=SleepObserver(m),audio_policy=BoundedAudioPolicy())
    for i in range(16):
        now[0]=2000.+2*i
        data=np.vstack([eeg(),now[0]-(511-np.arange(512))/256])
        muse.get_data=lambda:data
        result=session.run_step()
        assert result['ready'] and 'Alpha' in result['state'] and 'Beta' in result['state']
        assert result['sleep']['ready']==(i>=14)
        if i<14: assert actions[-1]['master_volume']==0
    assert len(calls)==2 and session.eeg_buffer.shape[1]==0
    now[0]+=2
    data=np.vstack([eeg(),now[0]-(511-np.arange(512))/256]); data[1,50]=np.nan
    assert not session.run_step()['ready']
    assert actions[-1]=={'master_volume':0.0}
    assert session.sleep_observer.buffer.shape[1]==0


def test_quality_details_distinguish_offset_from_flatline():
    m, calls = model()
    obs = SleepObserver(m)
    data = eeg()
    data[1] += 600  # DC offset, while variation remains normal.
    with pytest.raises(ValueError) as error:
        obs.update(data, np.arange(512)/256)
    message = str(error.value)
    assert "AF7: mean=600.00 uV" in message
    assert "absolute peak > 500 uV" in message
    assert "p99-p1=" in message and "std=" in message
    assert all(name + ":" in message for name in ("TP9", "AF7", "AF8", "TP10"))
    assert not calls and obs.buffer.shape[1] == 0
    data = eeg()
    data[3] = 0  # Non-model channel still triggers the unchanged four-channel gate.
    with pytest.raises(ValueError) as error:
        obs.update(data, np.arange(512)/256)
    assert "TP10: mean=0.00 uV" in str(error.value)
    assert "Flatline (quality < 0.75)" in str(error.value)
    assert "absolute peak > 500" not in str(error.value)
