import numpy as np

from scipy.signal import (
    butter,
    sosfiltfilt,
    hilbert,
)


class SlowWaveDetector:

    def __init__(
        self,
        fs=256,
        low=0.5,
        high=1.2,
        min_amplitude=1e-6,
    ):

        self.fs = fs
        self.low = low
        self.high = high
        # Numerical floor in the supplied signal units.
        self.min_amplitude = min_amplitude


    @staticmethod
    def _undefined():
        return {"slow_wave_amplitude": 0.0, "slow_wave_phase": 0.0,
                "slow_wave_stability": 0.0, "rhythm_defined": False}

    def _bandpass(
        self,
        signal,
    ):

        sos = butter(
            N=4,
            Wn=[
                self.low,
                self.high,
            ],
            btype="bandpass",
            fs=self.fs,
            output="sos",
        )

        filtered = sosfiltfilt(
            sos,
            signal,
        )

        return filtered


    def analyze(
        self,
        eeg,
    ):
        """
        eeg shape:
        channels x samples
        """

        eeg = np.asarray(eeg, dtype=float)
        if eeg.ndim == 1:

            eeg = eeg[
                np.newaxis,
                :
            ]


        if (eeg.ndim != 2 or not eeg.shape[0] or eeg.shape[1] < 2 * self.fs
                or not np.isfinite(eeg).all()
                or np.any(np.std(eeg, axis=1) <= self.min_amplitude)):
            return self._undefined()

        amplitudes = []
        phases = []


        for channel in eeg:

            filtered = self._bandpass(
                channel
            )


            analytic_signal = hilbert(
                filtered
            )


            amplitude = np.abs(
                analytic_signal
            )


            phase = np.angle(
                analytic_signal
            )


            recent_samples = min(
                self.fs * 2,
                len(amplitude),
            )


            recent_amplitude = amplitude[
                -recent_samples:
            ]


            amplitudes.append(
                float(
                    np.mean(
                        recent_amplitude
                    )
                )
            )


            phases.append(
                float(
                    phase[-1]
                )
            )


        mean_amplitude = float(
            np.mean(
                amplitudes
            )
        )


        if not np.isfinite(mean_amplitude) or min(amplitudes) <= self.min_amplitude:
            return self._undefined()

        complex_phases = np.exp(
            1j * np.array(
                phases
            )
        )


        mean_phase = float(
            np.angle(
                np.mean(
                    complex_phases
                )
            )
        )


        phase_consistency = float(
            np.abs(
                np.mean(
                    complex_phases
                )
            )
        )


        return {
            "rhythm_defined": True,
            "slow_wave_amplitude":
                mean_amplitude,

            "slow_wave_phase":
                mean_phase,

            "slow_wave_stability":
                phase_consistency,
        }