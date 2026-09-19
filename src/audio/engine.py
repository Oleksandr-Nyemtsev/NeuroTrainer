import numpy as np
import sounddevice as sd


class AudioEngine:

    def __init__(
        self,
        sample_rate=44100,
        carrier_frequency=180.0,
        beat_frequency=8.0,
        binaural_volume=0.04,
        noise_volume=0.12,
        noise_type="pink",
        master_volume=0.8,
        device=None,
    ):

        self.sample_rate = sample_rate
        self.device = device

        # Поточні значення
        self.carrier_frequency = carrier_frequency
        self.beat_frequency = beat_frequency
        self.binaural_volume = binaural_volume
        self.noise_volume = noise_volume
        self.master_volume = master_volume

        # Цільові значення
        self.target_carrier = carrier_frequency
        self.target_beat = beat_frequency
        self.target_binaural_volume = binaural_volume
        self.target_noise_volume = noise_volume
        self.target_master_volume = master_volume

        # 0 = pink
        # 1 = brown
        self.noise_mix = (
            1.0 if noise_type == "brown" else 0.0
        )

        self.target_noise_mix = self.noise_mix

        self.phase_left = 0.0
        self.phase_right = 0.0

        self.stream = None

        self.noise_position = 0

        print("Generating noise buffers...")

        self.pink_noise = self._generate_noise(
            seconds=60,
            exponent=0.5,
        )

        self.brown_noise = self._generate_noise(
            seconds=60,
            exponent=1.0,
        )

        print("Audio buffers ready.")


    def _generate_noise(
        self,
        seconds,
        exponent,
    ):

        samples = int(
            self.sample_rate * seconds
        )

        rng = np.random.default_rng(
            42
        )

        white = rng.normal(
            0,
            1,
            samples,
        )

        spectrum = np.fft.rfft(
            white
        )

        frequencies = np.fft.rfftfreq(
            samples,
            d=1 / self.sample_rate,
        )

        frequencies[0] = 1.0

        spectrum /= (
            frequencies ** exponent
        )

        noise = np.fft.irfft(
            spectrum,
            n=samples,
        )

        noise /= (
            np.max(
                np.abs(noise)
            )
            + 1e-8
        )

        return noise.astype(
            np.float32
        )


    def _smooth(
        self,
        current,
        target,
        speed=0.01,
    ):

        return (
            current
            + (target - current)
            * speed
        )


    def _get_noise_block(
        self,
        frames,
    ):

        end = (
            self.noise_position
            + frames
        )

        buffer_length = len(
            self.pink_noise
        )

        if end <= buffer_length:

            pink = self.pink_noise[
                self.noise_position:end
            ]

            brown = self.brown_noise[
                self.noise_position:end
            ]

        else:

            first_part = (
                buffer_length
                - self.noise_position
            )

            pink = np.concatenate(
                [
                    self.pink_noise[
                        self.noise_position:
                    ],
                    self.pink_noise[
                        :frames - first_part
                    ],
                ]
            )

            brown = np.concatenate(
                [
                    self.brown_noise[
                        self.noise_position:
                    ],
                    self.brown_noise[
                        :frames - first_part
                    ],
                ]
            )

        self.noise_position = (
            end % buffer_length
        )

        noise = (
            pink * (1.0 - self.noise_mix)
            +
            brown * self.noise_mix
        )

        return noise


    def _callback(
        self,
        outdata,
        frames,
        time_info,
        status,
    ):

        if status:
            print(status)

        # Плавно рухаємося до нових параметрів

        self.carrier_frequency = self._smooth(
            self.carrier_frequency,
            self.target_carrier,
        )

        self.beat_frequency = self._smooth(
            self.beat_frequency,
            self.target_beat,
        )

        self.binaural_volume = self._smooth(
            self.binaural_volume,
            self.target_binaural_volume,
        )

        self.noise_volume = self._smooth(
            self.noise_volume,
            self.target_noise_volume,
        )

        self.master_volume = self._smooth(
            self.master_volume,
            self.target_master_volume,
        )

        self.noise_mix = self._smooth(
            self.noise_mix,
            self.target_noise_mix,
        )


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


        left_tone = np.sin(
            left_phase
        )

        right_tone = np.sin(
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


        noise = self._get_noise_block(
            frames
        )


        left = (
            left_tone
            * self.binaural_volume
            +
            noise
            * self.noise_volume
        )

        right = (
            right_tone
            * self.binaural_volume
            +
            noise
            * self.noise_volume
        )


        left *= self.master_volume
        right *= self.master_volume


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

        if "carrier_frequency" in action:

            self.target_carrier = float(
                action["carrier_frequency"]
            )

        if "beat_frequency" in action:

            self.target_beat = float(
                action["beat_frequency"]
            )

        if "binaural_volume" in action:

            self.target_binaural_volume = float(
                action["binaural_volume"]
            )

        if "noise_volume" in action:

            self.target_noise_volume = float(
                action["noise_volume"]
            )

        if "master_volume" in action:

            self.target_master_volume = float(
                action["master_volume"]
            )

        if "noise_type" in action:

            if action["noise_type"] == "brown":

                self.target_noise_mix = 1.0

            elif action["noise_type"] == "pink":

                self.target_noise_mix = 0.0


    def stop(self):

        if self.stream is not None:

            self.stream.stop()
            self.stream.close()

            self.stream = None

            print(
                "AudioEngine stopped."
            )