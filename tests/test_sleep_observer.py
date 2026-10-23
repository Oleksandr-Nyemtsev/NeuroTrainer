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
        if result['ready']:
            assert result['stage_validated_for_input'] is False
            assert result['experimental_muse_transfer'] is True
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
    # Explicitly validated synthetic input exercises the retained policy branch.
    asleep={'ready':True,'stage_validated_for_input':True,
            'probabilities':dict(Wake=0,N1=0,N2=1,N3=0,REM=0)}
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


@pytest.mark.parametrize('stage', ['N2', 'REM'])
@pytest.mark.parametrize('validation', ['missing', False, None, 1, 'true'])
def test_unvalidated_stage_does_not_change_audio_action(stage, validation):
    policies = [BoundedAudioPolicy(), BoundedAudioPolicy()]
    observations = []
    for label in ('Wake', stage):
        observation = {'ready': True, 'probabilities': {
            name: float(name == label) for name in ('Wake', 'N1', 'N2', 'N3', 'REM')}}
        if validation != 'missing':
            observation['stage_validated_for_input'] = validation
        observations.append(observation)
    for t in range(0, 900, 2):
        score = 0 if t < 450 else 10
        awake, predicted_sleep = [policy.choose(score, obs, t)
                                  for policy, obs in zip(policies, observations)]
        assert predicted_sleep == awake
        assert policies[1].sleep_windows == 0
    assert predicted_sleep['master_volume'] == pytest.approx(.10)


@pytest.mark.parametrize('validation', ['missing', False])
def test_losing_validation_clears_sleep_evidence_before_next_update(validation):
    policy = BoundedAudioPolicy()
    observation = {'ready': True, 'stage_validated_for_input': True,
                   'probabilities': dict(Wake=0, N1=0, N2=0, N3=0, REM=1)}
    for t in range(0, 600, 30):
        previous = policy.choose(0, observation, t)
    assert policy.sleep_windows >= 2
    assert previous['master_volume'] == pytest.approx(.03)
    if validation == 'missing':
        observation.pop('stage_validated_for_input')
    else:
        observation['stage_validated_for_input'] = validation
    # Validation can disappear between scheduled 30-second policy updates.
    assert policy.choose(10, observation, 571) == previous
    assert policy.sleep_windows == 0
    next_action = policy.choose(10, observation, 600)
    assert next_action['master_volume'] == pytest.approx(.04)
    assert abs(next_action['beat_frequency'] - previous['beat_frequency']) <= .5
    assert policy.sleep_windows == 0


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


def test_packet_continuity_allows_bounded_jitter_and_accumulates_sleep(monkeypatch):
    now = [2000.]
    monkeypatch.setattr(module.time, 'time', lambda: now[0])
    monkeypatch.setattr(module, 'BoardShim', SimpleNamespace(
        get_eeg_channels=lambda _: [0,1,2,3], get_timestamp_channel=lambda _: 4))
    m, calls = model()
    muse = SimpleNamespace(package_num_channel=5)
    observer = SleepObserver(m)
    session = module.NeuroSession(muse, SimpleNamespace(apply_action=lambda _: None),
                                  sleep_observer=observer)
    last = 2000.
    for i in range(16):
        dt = np.full(516, 1/256)
        dt[:12] = .0065
        dt[12:24] = 2/256 - .0065
        ts = last + np.cumsum(dt)
        last = float(ts[-1]); now[0] = last
        ids = np.repeat(np.arange(i*43, (i+1)*43), 12)
        data = np.vstack([eeg(516), ts, ids])
        muse.get_data = lambda: data
        result = session.run_step()
        assert result['ready']
        assert result['timestamp_diagnostics']['gap_count'] > 0
        assert result['timestamp_diagnostics']['applied_gap_count'] == 0
    assert result['sleep']['ready'] and calls
    # Skip an ID while keeping timestamps normal: must reject and reset.
    ts = last + np.arange(1,517)/256; now[0] = float(ts[-1])
    ids = np.repeat(np.arange(16*43+1,17*43+1),12)
    data = np.vstack([eeg(516),ts,ids])
    result = session.run_step()
    assert not result['ready'] and result['reason'] == 'EEG packet sequence anomaly'
    assert observer.buffer.shape[1] == 0


@pytest.mark.parametrize('kind', ['large_gap', 'duplicate', 'backward'])
def test_observer_packet_evidence_does_not_override_bad_time(kind):
    m, _ = model()
    observer = SleepObserver(m)
    ts = np.arange(512)/256
    if kind == 'large_gap': ts[200:] += .1
    elif kind == 'duplicate': ts[200] = ts[199]
    else: ts[200] = ts[199] - .001
    with pytest.raises(ValueError, match='timestamp gap'):
        observer.update(eeg(), ts, packet_evidence=True)


def test_observer_without_packet_evidence_retains_strict_limit():
    m, _ = model()
    ts = np.arange(512)/256
    ts[200:] += .0025
    with pytest.raises(ValueError, match='timestamp gap'):
        SleepObserver(m).update(eeg(), ts)
    assert not SleepObserver(m).update(eeg(), ts, packet_evidence=True)['ready']
