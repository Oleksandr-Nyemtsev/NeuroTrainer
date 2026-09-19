# NeuroTrainer

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