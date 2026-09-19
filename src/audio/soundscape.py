import numpy as np


class NaturalSoundscape:

    def __init__(
        self,
        sample_rate=44100,
        seed=42,
    ):

        self.sample_rate = sample_rate

        self.rng = np.random.default_rng(
            seed
        )

        # Стан генератора хвиль
        self.wave_phase = 0.0
        self.wave_speed = 0.07

        # Простий low-pass стан
        self.wave_filter_state = 0.0
        self.rain_filter_state = 0.0

        # Стан пташиного chirp
        self.bird_active = False
        self.bird_position = 0
        self.bird_length = 0
        self.bird_frequency = 2200.0
        self.bird_phase = 0.0


    def _lowpass(
        self,
        signal,
        state,
        alpha,
    ):

        output = np.empty_like(
            signal
        )

        current = state

        for i in range(len(signal)):

            current += alpha * (
                signal[i] - current
            )

            output[i] = current

        return output, current


    def _generate_wave(
        self,
        frames,
        volume,
        density,
    ):

        noise = self.rng.normal(
            0.0,
            1.0,
            frames,
        ).astype(np.float32)

        # Прибираємо різкий високочастотний шум
        filtered, self.wave_filter_state = (
            self._lowpass(
                noise,
                self.wave_filter_state,
                alpha=0.015,
            )
        )

        t = (
            np.arange(frames)
            / self.sample_rate
        )

        # Дуже повільна нерегулярна хвиля
        envelope = (
            0.5
            + 0.5
            * np.sin(
                2
                * np.pi
                * self.wave_speed
                * t
                + self.wave_phase
            )
        )

        envelope = (
            envelope ** 2.5
        )

        self.wave_phase += (
            2
            * np.pi
            * self.wave_speed
            * frames
            / self.sample_rate
        )

        self.wave_phase %= (
            2 * np.pi
        )

        wave = (
            filtered
            * envelope
            * volume
            * density
        )

        return wave


    def _generate_rain(
        self,
        frames,
        volume,
        density,
    ):

        rain = np.zeros(
            frames,
            dtype=np.float32,
        )

        # Ймовірність окремих крапель
        probability = (
            density
            * 0.0007
        )

        hits = (
            self.rng.random(frames)
            < probability
        )

        positions = np.where(
            hits
        )[0]

        for position in positions:

            length = min(
                int(
                    self.sample_rate
                    * self.rng.uniform(
                        0.015,
                        0.060,
                    )
                ),
                frames - position,
            )

            if length <= 0:
                continue

            envelope = np.exp(
                -np.linspace(
                    0,
                    7,
                    length,
                )
            )

            drop = (
                self.rng.normal(
                    0,
                    1,
                    length,
                )
                * envelope
            )

            rain[
                position:
                position + length
            ] += drop.astype(
                np.float32
            )

        rain, self.rain_filter_state = (
            self._lowpass(
                rain,
                self.rain_filter_state,
                alpha=0.12,
            )
        )

        return (
            rain
            * volume
        )


    def _start_bird(
        self,
    ):

        self.bird_active = True

        self.bird_position = 0

        self.bird_length = int(
            self.sample_rate
            * self.rng.uniform(
                0.25,
                0.60,
            )
        )

        self.bird_frequency = (
            self.rng.uniform(
                1600,
                2800,
            )
        )


    def _generate_bird(
        self,
        frames,
        volume,
        density,
    ):

        output = np.zeros(
            frames,
            dtype=np.float32,
        )

        if (
            not self.bird_active
            and self.rng.random()
            < density * 0.015
        ):
            self._start_bird()

        if not self.bird_active:
            return output

        remaining = (
            self.bird_length
            - self.bird_position
        )

        count = min(
            frames,
            remaining,
        )

        if count <= 0:

            self.bird_active = False

            return output

        local_t = (
            np.arange(count)
            + self.bird_position
        ) / self.sample_rate

        progress = (
            (
                np.arange(count)
                + self.bird_position
            )
            / self.bird_length
        )

        envelope = np.sin(
            np.pi
            * np.clip(
                progress,
                0,
                1,
            )
        ) ** 2

        modulation = (
            1.0
            + 0.08
            * np.sin(
                2
                * np.pi
                * 5.0
                * local_t
            )
        )

        phase = (
            2
            * np.pi
            * self.bird_frequency
            * local_t
            * modulation
        )

        chirp = (
            np.sin(
                phase
                + self.bird_phase
            )
            * envelope
            * volume
        )

        output[:count] = chirp.astype(
            np.float32
        )

        self.bird_position += count

        if (
            self.bird_position
            >= self.bird_length
        ):
            self.bird_active = False

        return output


    def render(
        self,
        frames,
        wave_volume=0.20,
        rain_volume=0.03,
        bird_volume=0.01,
        wave_density=0.50,
        rain_density=0.10,
        bird_density=0.02,
        silence_probability=0.05,
    ):

        # Іноді залишаємо природну паузу
        if (
            self.rng.random()
            < silence_probability
        ):
            return np.zeros(
                frames,
                dtype=np.float32,
            )

        wave = self._generate_wave(
            frames,
            wave_volume,
            wave_density,
        )

        rain = self._generate_rain(
            frames,
            rain_volume,
            rain_density,
        )

        bird = self._generate_bird(
            frames,
            bird_volume,
            bird_density,
        )

        result = (
            wave
            + rain
            + bird
        )

        # Захист від clipping
        result = np.clip(
            result,
            -1.0,
            1.0,
        )

        return result.astype(
            np.float32
        )