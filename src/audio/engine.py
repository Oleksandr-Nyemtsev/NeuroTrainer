import numpy as np
import sounddevice as sd

from src.audio.soundscape import NaturalSoundscape


class AudioEngine:

    def __init__(
        self,
        sample_rate=44100,
        carrier_frequency=180.0,
        beat_frequency=8.0,
        binaural_volume=0.02,
        wave_volume=0.20,
        rain_volume=0.03,
        bird_volume=0.01,
        wave_density=0.50,
        rain_density=0.10,
        bird_density=0.02,
        silence_probability=0.05,
        master_volume=0.70,
        device=None,
    ):

        self.sample_rate = sample_rate
        self.device = device

        # Поточні значення
        self.carrier_frequency = carrier_frequency
        self.beat_frequency = beat_frequency

        self.binaural_volume = binaural_volume
        self.wave_volume = wave_volume
        self.rain_volume = rain_volume
        self.bird_volume = bird_volume

        self.wave_density = wave_density
        self.rain_density = rain_density
        self.bird_density = bird_density

        self.silence_probability = silence_probability
        self.master_volume = master_volume

        # Цільові значення
        self.target_carrier_frequency = carrier_frequency
        self.target_beat_frequency = beat_frequency

        self.target_binaural_volume = binaural_volume
        self.target_wave_volume = wave_volume
        self.target_rain_volume = rain_volume
        self.target_bird_volume = bird_volume

        self.target_wave_density = wave_density
        self.target_rain_density = rain_density
        self.target_bird_density = bird_density

        self.target_silence_probability = silence_probability
        self.target_master_volume = master_volume

        # Фази binaural tone
        self.phase_left = 0.0
        self.phase_right = 0.0

        # Natural sound generator
        self.soundscape = NaturalSoundscape(
            sample_rate=sample_rate
        )

        self.stream = None


    def _smooth(
        self,
        current,
        target,
        speed=0.02,
    ):

        return (
            current
            + (target - current)
            * speed
        )


    def _update_parameters(self):

        self.carrier_frequency = self._smooth(
            self.carrier_frequency,
            self.target_carrier_frequency,
        )

        self.beat_frequency = self._smooth(
            self.beat_frequency,
            self.target_beat_frequency,
        )

        self.binaural_volume = self._smooth(
            self.binaural_volume,
            self.target_binaural_volume,
        )

        self.wave_volume = self._smooth(
            self.wave_volume,
            self.target_wave_volume,
        )

        self.rain_volume = self._smooth(
            self.rain_volume,
            self.target_rain_volume,
        )

        self.bird_volume = self._smooth(
            self.bird_volume,
            self.target_bird_volume,
        )

        self.wave_density = self._smooth(
            self.wave_density,
            self.target_wave_density,
        )

        self.rain_density = self._smooth(
            self.rain_density,
            self.target_rain_density,
        )

        self.bird_density = self._smooth(
            self.bird_density,
            self.target_bird_density,
        )

        self.silence_probability = self._smooth(
            self.silence_probability,
            self.target_silence_probability,
        )

        self.master_volume = self._smooth(
            self.master_volume,
            self.target_master_volume,
        )


    def _generate_binaural(
        self,
        frames,
    ):

        t = (
            np.arange(frames)
            / self.sample_rate
        )

        left_frequency = (
            self.carrier_frequency
        )

        right_frequency = (
            self.carrier_frequency
            + self.beat_frequency
        )

        left_phase = (
            2
            * np.pi
            * left_frequency
            * t
            + self.phase_left
        )

        right_phase = (
            2
            * np.pi
            * right_frequency
            * t
            + self.phase_right
        )

        left = np.sin(
            left_phase
        )

        right = np.sin(
            right_phase
        )

        self.phase_left = (
            left_phase[-1]
            + 2
            * np.pi
            * left_frequency
            / self.sample_rate
        ) % (
            2 * np.pi
        )

        self.phase_right = (
            right_phase[-1]
            + 2
            * np.pi
            * right_frequency
            / self.sample_rate
        ) % (
            2 * np.pi
        )

        left *= self.binaural_volume
        right *= self.binaural_volume

        return left, right


    def _callback(
        self,
        outdata,
        frames,
        time_info,
        status,
    ):

        if status:
            print(status)

        self._update_parameters()

        # Natural mono soundscape
        natural = self.soundscape.render(
            frames=frames,
            wave_volume=self.wave_volume,
            rain_volume=self.rain_volume,
            bird_volume=self.bird_volume,
            wave_density=self.wave_density,
            rain_density=self.rain_density,
            bird_density=self.bird_density,
            silence_probability=self.silence_probability,
        )

        # Stereo binaural layer
        binaural_left, binaural_right = (
            self._generate_binaural(
                frames
            )
        )

        # Невелика асиметрія natural sound
        natural_left = (
            natural * 0.96
        )

        natural_right = (
            natural * 1.04
        )

        left = (
            natural_left
            + binaural_left
        )

        right = (
            natural_right
            + binaural_right
        )

        left *= self.master_volume
        right *= self.master_volume

        left = np.clip(
            left,
            -1.0,
            1.0,
        )

        right = np.clip(
            right,
            -1.0,
            1.0,
        )

        outdata[:, 0] = left
        outdata[:, 1] = right


    def start(self):

        if self.stream is not None:
            return

        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=2,
            dtype="float32",
            callback=self._callback,
            device=self.device,
        )

        self.stream.start()

        print(
            "AudioEngine started."
        )


    def apply_action(
        self,
        action,
    ):

        # Працює і з AudioAction,
        # і зі словником

        if hasattr(
            action,
            "carrier_frequency",
        ):

            self.target_carrier_frequency = float(
                action.carrier_frequency
            )

            self.target_beat_frequency = float(
                action.beat_frequency
            )

            self.target_binaural_volume = float(
                action.binaural_volume
            )

            self.target_wave_volume = float(
                action.wave_volume
            )

            self.target_rain_volume = float(
                action.rain_volume
            )

            self.target_bird_volume = float(
                action.bird_volume
            )

            self.target_wave_density = float(
                action.wave_density
            )

            self.target_rain_density = float(
                action.rain_density
            )

            self.target_bird_density = float(
                action.bird_density
            )

            self.target_silence_probability = float(
                action.silence_probability
            )

            self.target_master_volume = float(
                action.master_volume
            )

            return


        if isinstance(
            action,
            dict,
        ):

            if "carrier_frequency" in action:
                self.target_carrier_frequency = float(
                    action["carrier_frequency"]
                )

            if "beat_frequency" in action:
                self.target_beat_frequency = float(
                    action["beat_frequency"]
                )

            if "binaural_volume" in action:
                self.target_binaural_volume = float(
                    action["binaural_volume"]
                )

            if "wave_volume" in action:
                self.target_wave_volume = float(
                    action["wave_volume"]
                )

            if "rain_volume" in action:
                self.target_rain_volume = float(
                    action["rain_volume"]
                )

            if "bird_volume" in action:
                self.target_bird_volume = float(
                    action["bird_volume"]
                )

            if "wave_density" in action:
                self.target_wave_density = float(
                    action["wave_density"]
                )

            if "rain_density" in action:
                self.target_rain_density = float(
                    action["rain_density"]
                )

            if "bird_density" in action:
                self.target_bird_density = float(
                    action["bird_density"]
                )

            if "silence_probability" in action:
                self.target_silence_probability = float(
                    action["silence_probability"]
                )

            if "master_volume" in action:
                self.target_master_volume = float(
                    action["master_volume"]
                )


    def stop(self):

        if self.stream is not None:

            self.stream.stop()
            self.stream.close()

            self.stream = None

            print(
                "AudioEngine stopped."
            )