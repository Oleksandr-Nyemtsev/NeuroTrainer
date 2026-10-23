# NeuroTrainer

## Поточний стан — 2026-10-23

Мета: **EEG → автоматичний вибір звуку → вимірювання реакції → персональне навчання**.
Зараз реалізовано автоматичний збір даних для цього навчання. Навчена модель
персонального підбору звуку ще не готова.

### Що вже працює

- Muse 2: читання EEG, перевірка якості й часової послідовності, аналіз ритмів кожні 2 секунди.
- Режим `--audio-experiment`: автоматичне чергування хвилини тиші та хвилини
  бінаурального звуку з різницею частот 6, 8 або 10 Гц. Порядок задається
  перемішуванням із фіксованим seed, а не навченою нейромережею.
- При непридатному сигналі звук вимикається; після відновлення починається новий тихий блок.
- Локальний запис сирого EEG, аудіокоманд і звітів.
- Експорт завершених блоків «EEG до → звук → EEG після» та змін під час тиші.
  Перервані блоки й повторні вікна не використовуються як незалежні приклади.
- Додаткове порівняння з особистим базовим EEG та заготовка прогнозу суб'єктивного
  комфорту (`--controller-shadow`). Ця заготовка не керує звуком.

### Межі поточної версії

У режимі аудіоексперименту **модель сну не завантажується**. Перенесені на Muse
класифікатори не підтверджені для визначення стадій сну; їхні ймовірності не є
підтвердженням REM. Зростання alpha/theta також не є підтвердженням REM або відновлення.

Експорт містить спостережувані зміни EEG, а не доведений ефект звуку.
Наступний крок — зібрати придатні аудіосесії та перевірити прогноз реакції на
окремих сесіях, порівнюючи його з простими прогнозами й змінами під час тиші.
`train_audio_response.py` має іншу ціль — ручні оцінки комфорту; нові EEG-експорти
не є готовим входом для цього скрипту.

### Запуск аудіоексперименту

З кореня репозиторію, в уже налаштованому середовищі `neuro_env`, з Muse 2:

```powershell
python run_sleep_observer.py --audio-experiment --audio --minutes 8 --condition awake-eyes-closed
python analyze_session.py --latest
python prepare_audio_trials.py --latest
```

`awake-eyes-closed` означає заявлене користувачем неспання із закритими очима.
Для відкритих очей використовуйте `awake-eyes-open`. Програма не визначає цю
умову автоматично. `Ctrl+C` зупиняє сесію. Без `--audio` звук не відтворюється;
такий запуск не дає прикладів реакції на звук.

Режим генерує звук локально, автоматично записує сирий EEG і не потребує
ваг моделі сну чи інтернету під час роботи. Залежності та драйвери повинні
бути встановлені заздалегідь. Починайте з комфортної гучності пристрою;
програмна гучність не є вимірюванням акустичного рівня.

Результати зберігаються в `data/sessions/`, приклади — в `data/sessions/trials/`.
Нуль прикладів можливий, якщо блоки переривалися. Саме `Audio enabled: True`
не підтверджує, що була команда ненульової гучності. Новий живий запуск після
виправлення накопичення EEG ще потребує перевірки користувачем.

### Документація та перевірки

- [Офлайн-робота та підготовка прикладів](docs/offline_week.md)
- [Автоматичний аудіоексперимент](docs/automatic_audio_experiment.md)
- [Особистий базовий EEG](docs/personal_baseline.md)
- [Журнал сесії](docs/session_journal.md)
- [Описовий аналіз реакцій](docs/session_analysis.md)
- [Експериментальний спостерігач сну](docs/live_sleep_observer.md)
- [Заготовка нейроконтролера комфорту](docs/neural_audio_controller.md)

```powershell
python -m pytest -q
```

Стандартні тести використовують штучні дані без підключення Muse чи відтворення
звуку; перевірки, що потребують великих датасетів, за замовчуванням пропускаються.
Особисті EEG, звіти та локальні ваги не входять до цього оновлення GitHub.
Їм потрібна окрема резервна копія.

BOAS Hybrid: test accuracy **0.8028**, macro-F1 **0.6693** на окремих учасниках
після виправлення split. Це результат на BOAS, а не оцінка точності на Muse.
[Розподіл учасників](docs/boas_participant_split.md), [офлайн-перевірка](docs/hybrid_replay.md).

## Початкова архітектура та дослідницький план

Нижче збережено початковий опис. Він містить також заплановані компоненти;
актуальний склад робочого режиму наведено вище.

---

## Версія / Version

**0.1 — Experimental Research Build**

---

## Про проєкт

**NeuroTrainer** — експериментальна EEG-guided closed-loop платформа для роботи з мозковими сигналами в реальному часі.

Мета проєкту — створити систему, яка не просто відтворює релакс-музику, а:

1. зчитує EEG;
2. оцінює поточний стан мозку;
3. аналізує alpha, slow-wave та sleep-like patterns;
4. приймає рішення про тип аудіостимуляції;
5. змінює аудіо відповідно до EEG;
6. повторно аналізує реакцію мозку;
7. адаптує наступну дію.

Основна ідея:

```text
EEG
↓
Brain State Estimation
↓
Adaptive Decision
↓
Audio / Stimulation
↓
EEG Response
↓
Next Decision
```

Це формує **closed-loop system** — замкнену систему з постійним зворотним зв’язком.

---

# Основна ціль

NeuroTrainer створюється як дослідницька платформа для:

- adaptive relaxation;
- sleep-onset support;
- brain-state estimation;
- EEG-guided audio;
- personalized stimulation;
- sleep-like state tracking;
- future neurofeedback;
- future reinforcement learning control.

Проєкт не повинен бути звичайною програмою типу:

```text
натиснув Play
↓
грає дощ / море / музика
```

Замість цього цільова архітектура:

```text
EEG
↓
BrainState
↓
Controller
↓
Stimulation Policy
↓
Audio Engine
↓
EEG
```

---

# Пристрої / Hardware

## Поточний пристрій

Перший підтримуваний пристрій:

```text
Muse 2
```

Канали Muse 2:

```text
TP9
AF7
AF8
TP10
```

Sampling rate:

```text
256 Hz
```

---

## Планована підтримка

У майбутньому:

- Muse S
- OpenBCI
- Unicorn
- LSL-compatible EEG devices
- інші багатоканальні EEG-пристрої

---

# Поточна архітектура

```text
Muse 2 EEG
    │
    ├── Signal Processing
    │   ├── filtering
    │   ├── artifact detection
    │   ├── PSD
    │   ├── band features
    │   ├── alpha tracking
    │   └── slow-wave tracking
    │
    ├── Sleep Model
    │   └── Wake / N1 / N2 / N3 / REM probabilities
    │
    ↓
BrainStateEstimator
    ↓
BrainState
    │
    ├───────────────┐
    │               │
    ↓               ↓
AdaptiveAudio     Stimulation
Controller        Policy
    │               │
    └───────┬───────┘
            ↓
        AudioEngine
            ↓
        Headphones
            ↓
        EEG Response
            ↺
```

---

# Signal Processing

Поточний EEG pipeline включає:

- band-pass filtering;
- Welch PSD;
- Delta / Theta / Alpha / Beta / Gamma bands;
- multi-channel EEG processing;
- basic artifact detection;
- smoothing;
- alpha analysis;
- slow-wave analysis.

Основні модулі:

```text
src/signal/
```

---

## Alpha Tracker

Файл:

```text
src/signal/alpha_tracker.py
```

Модуль оцінює:

- Individual Alpha Frequency — IAF;
- alpha amplitude;
- alpha phase;
- alpha stability.

Типовий pipeline:

```text
EEG
↓
PSD / Welch
↓
7.5–12.5 Hz
↓
alpha peak
↓
Individual Alpha Frequency
↓
band-pass around IAF
↓
Hilbert transform
↓
Amplitude + Phase
```

Приклад:

```text
IAF = 10.0 Hz
Alpha amplitude = ...
Alpha phase = ...
Alpha stability = ...
```

Synthetic test уже перевіряє, що сигнал з alpha 10 Hz правильно визначається як приблизно 10 Hz.

---

# Slow-Wave Detector

Файл:

```text
src/signal/slow_wave_detector.py
```

Модуль аналізує slow-wave activity приблизно в діапазоні:

```text
0.5–1.2 Hz
```

Визначає:

- slow-wave amplitude;
- slow-wave phase;
- slow-wave stability.

Використовується для майбутньої NREM / slow-wave closed-loop logic.

---

# Sleep CNN

Модель:

```text
src/models/sleep_cnn.py
```

Wrapper:

```text
src/models/sleep_model.py
```

Модель — 1D CNN для класифікації 30-секундних EEG-вікон.

Класи:

```text
Wake
N1
N2
N3
REM
```

---

## Dataset

Для pretraining використовується:

```text
Sleep-EDF
```

Поточний формат підготовлених даних:

```text
2 EEG channels
100 Hz
30 seconds
3000 samples
```

Тобто один приклад має форму:

```text
(2, 3000)
```

---

# Sleep-EDF preprocessing

Основний скрипт:

```text
prepare_sleep_edf.py
```

Виконує:

```text
download
↓
read EDF
↓
annotations
↓
30-second epochs
↓
EEG channels
↓
wake trimming
↓
label remapping
↓
normalization
↓
save .npz
```

Оброблені записи зберігаються в:

```text
data/processed/sleep_edf/
```

---

# Dataset split

Dataset ділиться не випадково по epochs, а по subjects.

Приклад:

```text
subjects A,B,C → train
subjects D,E → validation
subjects F,G → test
```

Це потрібно, щоб записи однієї людини не потрапляли одночасно в train і test.

---

# GPU Training

Тренувальний скрипт:

```text
train_sleep_model.py
```

Підтримується автоматичний вибір:

```text
CUDA
або
CPU
```

Поточний тестовий GPU:

```text
NVIDIA T500 4 GB
```

У тестовому запуску GPU давав приблизно десятикратне прискорення проти CPU.

---

# Поточний benchmark

На проміжному датасеті було отримано приблизно:

```text
Test Accuracy ≈ 79%
Macro F1 ≈ 0.74
```

Це дослідницький результат, не клінічна валідація.

---

# BrainState

Файл:

```text
src/core/brain_state.py
```

BrainState об'єднує:

```text
sleep_probs
IAF
alpha amplitude
alpha phase
alpha stability
slow-wave amplitude
slow-wave phase
slow-wave stability
signal quality
session time
```

Приклад:

```text
SleepCNN:
N3 = 0.79

Alpha:
reliable = False

Slow wave:
reliable = True

Signal quality:
0.95
```

---

# BrainStateEstimator

Файл:

```text
src/core/brain_state_estimator.py
```

Об'єднує:

```text
SleepModel
+
AlphaTracker
+
SlowWaveDetector
↓
BrainState
```

Це центральний модуль оцінки стану мозку.

---

# AdaptiveAudioController

Файл:

```text
src/core/adaptive_audio_controller.py
```

Відповідає за загальне аудіосередовище.

Session phases:

```text
CALMING
SLEEP_ONSET
RESTORATIVE
WAKE_UP
```

Controller може змінювати:

```text
carrier frequency
beat frequency
binaural volume
wave volume
rain volume
bird volume
event density
master volume
silence probability
```

Поточна логіка rule-based.

У майбутньому планується personalized / learned controller.

---

# StimulationPolicy

Файл:

```text
src/core/stimulation_policy.py
```

Це окремий блок, який вирішує:

```text
чи дозволена активна stimulation зараз?
```

Режими:

```text
NONE
COMFORT_ONLY
ALPHA_TRACKING
SLOW_WAVE_TRACKING
WAKE_UP
```

Приклад логіки:

```text
Poor EEG
→ NONE

Wake / N1 + reliable alpha
→ ALPHA_TRACKING

N3 + reliable slow wave
→ SLOW_WAVE_TRACKING

REM-like state
→ COMFORT_ONLY
```

---

# AudioAction

Файл:

```text
src/audio/actions.py
```

AudioAction описує не один файл звуку, а набір параметрів аудіосередовища.

Наприклад:

```text
carrier_frequency
beat_frequency
binaural_volume
wave_volume
rain_volume
bird_volume
wave_density
rain_density
bird_density
master_volume
silence_probability
```

---

# AudioEngine

Файл:

```text
src/audio/engine.py
```

Відповідає за:

- stereo output;
- left/right binaural frequencies;
- smooth parameter transitions;
- soundscape mixing;
- master volume;
- future stimulation pulses.

---

# Soundscape

Файл:

```text
src/audio/soundscape.py
```

Поточний procedural soundscape експериментальний.

Було протестовано:

- generated waves;
- rain-like events;
- bird-like events;
- binaural layer.

Результат показав, що чисто синтетичний soundscape звучить неприродно.

Тому майбутній план:

```text
real audio textures
+
adaptive mixing
+
neuromodulation layer
+
stimulation layer
```

---

# Closed-Loop Concept

Цільова логіка NeuroTrainer:

```text
EEG before action
↓
BrainState
↓
Audio / Stimulation Decision
↓
EEG after action
↓
State Change
↓
Reward / Penalty
↓
Next Action
```

---

# Future Personalization

У майбутньому система повинна вчитися:

- які звуки підходять конкретній людині;
- які частоти викликають arousal;
- які режими покращують relaxation;
- які параметри допомагають переходу між states;
- коли краще взагалі не стимулювати.

---

# Reinforcement Learning

Майбутня архітектура може містити:

```text
State
↓
Policy
↓
Action
↓
EEG Response
↓
Reward
```

Possible state:

```text
SleepCNN probabilities
Alpha features
Slow-wave features
artifact score
signal quality
session history
```

Possible actions:

```text
audio parameters
stimulation mode
pulse timing
silence
binaural parameters
ambient complexity
```

---

# Riemannian EEG / Alexandre Barachant

Планований наступний research-блок:

```text
pyRiemann
```

Ідеї Alexandre Barachant планується використати для:

- covariance matrices;
- Riemannian EEG features;
- artifact detection;
- Riemannian Potato;
- robust multi-channel state representation;
- future transfer learning.

План:

```text
CNN features
+
Spectral features
+
Riemannian features
↓
Improved BrainState
```

---

# Artifact Detection

Поточна базова версія:

```text
peak-to-peak threshold
```

У майбутньому:

```text
Riemannian artifact detection
+
signal quality score
+
movement rejection
+
blink handling
```

---

# Muse Domain Adaptation

Важливе обмеження:

SleepCNN навчена на Sleep-EDF:

```text
2 channels
100 Hz
```

Muse 2:

```text
4 channels
256 Hz
TP9
AF7
AF8
TP10
```

Тому пряме використання Sleep-EDF моделі на Muse є експериментальним.

Майбутня робота:

- resampling;
- channel mapping;
- Muse-specific preprocessing;
- Muse calibration;
- own-session dataset;
- fine-tuning;
- domain adaptation;
- personalized model.

---

# Signal Quality

Signal quality використовується як safety gate.

Наприклад:

```text
signal_quality < threshold
↓
do not change stimulation aggressively
```

Це потрібно, щоб controller не реагував на:

- movement;
- poor contact;
- artifacts;
- bad Bluetooth EEG frames.

---

# Tests

Усі тести знаходяться в:

```text
tests/
```

Основні:

```text
test_alpha_tracker.py
test_slow_wave_detector.py
test_sleep_model_real.py
test_real_brain_state.py
test_brain_to_audio.py
test_stimulation_policy.py
```

Запуск із кореня проєкту:

```bash
python -m tests.test_alpha_tracker
python -m tests.test_slow_wave_detector
python -m tests.test_sleep_model_real
python -m tests.test_real_brain_state
python -m tests.test_brain_to_audio
python -m tests.test_stimulation_policy
```

---

# Структура проєкту

```text
NeuroTrainer/
│
├── src/
│   │
│   ├── audio/
│   │   ├── actions.py
│   │   ├── engine.py
│   │   └── soundscape.py
│   │
│   ├── core/
│   │   ├── adaptive_audio_controller.py
│   │   ├── brain_state.py
│   │   ├── brain_state_estimator.py
│   │   ├── controller.py
│   │   ├── session.py
│   │   └── stimulation_policy.py
│   │
│   ├── data/
│   │   ├── logger.py
│   │   ├── sleep_dataset.py
│   │   └── sleep_edf.py
│   │
│   ├── hardware/
│   │   └── muse.py
│   │
│   ├── models/
│   │   ├── sleep_cnn.py
│   │   └── sleep_model.py
│   │
│   ├── signal/
│   │   ├── alpha_tracker.py
│   │   ├── artifacts.py
│   │   ├── features.py
│   │   ├── filters.py
│   │   ├── metrics.py
│   │   ├── multichannel.py
│   │   ├── processor.py
│   │   ├── slow_wave_detector.py
│   │   ├── smoothing.py
│   │   ├── state.py
│   │   └── welch.py
│   │
│   └── config.py
│
├── tests/
├── notebooks/
├── data/
├── models/
├── sounds/
│
├── prepare_sleep_edf.py
├── train_sleep_model.py
├── run_session.py
├── start_neurotrainer.bat
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Installation

Рекомендована версія:

```text
Python 3.11
```

Приклад створення Conda environment:

```bash
conda create -n neuro_env python=3.11
conda activate neuro_env
```

Установка залежностей:

```bash
pip install -r requirements.txt
```

---

# Requirements

Основні бібліотеки:

```text
numpy
scipy
pandas
matplotlib
mne
brainflow
sounddevice
torch
scikit-learn
jupyterlab
```

У майбутньому:

```text
pyriemann
```

---

# Sleep-EDF preprocessing

Запуск:

```bash
python prepare_sleep_edf.py
```

Processed data:

```text
data/processed/sleep_edf/
```

---

# Sleep CNN training

Запуск:

```bash
python train_sleep_model.py
```

Найкраща модель:

```text
models/sleep_cnn_best.pt
```

---

# Run NeuroTrainer

Поточний experimental entry point:

```bash
python run_session.py
```

Real-time Muse closed-loop integration ще знаходиться в розробці.

---

# Git / GitHub

Основний репозиторій:

```text
NeuroTrainer
```

Рекомендований робочий цикл:

```bash
git status
git add .
git commit -m "опис змін"
git push
```

---

# .gitignore

У GitHub не повинні потрапляти:

```text
raw EEG
Sleep-EDF EDF files
processed NPZ files
trained model weights
Jupyter cache
Python cache
temporary files
```

---

# Поточний статус

На даному етапі вже працює:

```text
Sleep-EDF preprocessing
Sleep CNN training
CUDA training
Sleep-stage inference
AlphaTracker
SlowWaveDetector
BrainState
BrainStateEstimator
AdaptiveAudioController
StimulationPolicy
AudioAction
AudioEngine
Synthetic tests
Real Sleep-EDF integration tests
Git / GitHub workflow
```

---

# Наступний великий блок

Наступний research milestone:

```text
Riemannian EEG processing
```

План:

```text
install pyRiemann
↓
covariance features
↓
Riemannian artifact detection
↓
compare CNN-only vs CNN + Riemannian
↓
integrate into BrainState
```

---

# Довгострокова ціль

NeuroTrainer має стати персоналізованою EEG closed-loop research platform:

```text
Brain
↓
EEG
↓
AI
↓
Decision
↓
Sound / Stimulation
↓
Brain Response
↓
Learning
↺
```

---

# Важливе обмеження

NeuroTrainer є **експериментальним дослідницьким проєктом**.

Він:

- не є медичним пристроєм;
- не призначений для діагностики;
- не замінює polysomnography;
- не замінює нормальний сон;
- не повинен використовуватися для клінічних рішень.

Sleep-stage outputs, alpha tracking, slow-wave tracking та stimulation decisions на даному етапі є research estimates.

---

# License

Ліцензія ще не вибрана.

Перед повноцінною open-source публікацією потрібно окремо додати:

```text
LICENSE
```

---

# Author

**Oleksandr Nyemtsev**

Project:

```text
NeuroTrainer
```

EEG / BCI / Neurotechnology / Machine Learning / Closed-Loop Audio Research