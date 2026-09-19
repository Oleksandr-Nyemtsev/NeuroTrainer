import numpy as np

from scipy.signal import (
    welch,
    butter,
    sosfiltfilt,
    hilbert,
)


class AlphaTracker:

    def __init__(
        self,
        fs=256,
        alpha_min=7.5,
        alpha_max=12.5,
        filter_width=1.5,
    ):

        self.fs = fs

        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        self.filter_width = filter_width

        self.iaf = None


    def estimate_iaf(
        self,
        eeg,
    ):
        """
        Estimate Individual Alpha Frequency.

        eeg shape:
        channels x samples
        """

        if eeg.ndim == 1:
            eeg = eeg[np.newaxis, :]

        channel_peaks = []

        for channel in eeg:

            freqs, psd = welch(
                channel,
                fs=self.fs,
                nperseg=min(
                    len(channel),
                    self.fs * 4,
                ),
            )

            alpha_mask = (
                (freqs >= self.alpha_min)
                &
                (freqs <= self.alpha_max)
            )

            alpha_freqs = freqs[
                alpha_mask
            ]

            alpha_psd = psd[
                alpha_mask
            ]

            if len(alpha_freqs) == 0:
                continue

            peak_index = np.argmax(
                alpha_psd
            )

            peak_frequency = (
                alpha_freqs[
                    peak_index
                ]
            )

            channel_peaks.append(
                peak_frequency
            )


        if len(channel_peaks) == 0:

            self.iaf = None

            return None


        self.iaf = float(
            np.median(
                channel_peaks
            )
        )

        return self.iaf


    def _alpha_bandpass(
        self,
        signal,
    ):

        if self.iaf is None:

            raise RuntimeError(
                "IAF has not been estimated yet."
            )


        low = max(
            1.0,
            self.iaf
            - self.filter_width,
        )

        high = min(
            self.fs / 2 - 1,
            self.iaf
            + self.filter_width,
        )


        sos = butter(
            N=4,
            Wn=[low, high],
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
        Returns current alpha information.

        eeg shape:
        channels x samples
        """

        if self.iaf is None:

            self.estimate_iaf(
                eeg
            )


        if self.iaf is None:

            return {
                "iaf": None,
                "alpha_amplitude": 0.0,
                "alpha_phase": 0.0,
                "alpha_stability": 0.0,
            }


        if eeg.ndim == 1:
            eeg = eeg[np.newaxis, :]


        amplitudes = []
        phases = []


        for channel in eeg:

            filtered = (
                self._alpha_bandpass(
                    channel
                )
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


            # Беремо середню amplitude
            # з останньої секунди

            samples_last_second = (
                min(
                    self.fs,
                    len(amplitude),
                )
            )


            recent_amplitude = (
                amplitude[
                    -samples_last_second:
                ]
            )


            amplitudes.append(
                float(
                    np.mean(
                        recent_amplitude
                    )
                )
            )


            # Phase останньої точки
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


        # Circular mean для phase

        complex_phases = np.exp(
            1j
            * np.array(
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


        # Наскільки канали погоджуються
        # між собою по фазі

        phase_consistency = float(
            np.abs(
                np.mean(
                    complex_phases
                )
            )
        )


        return {
            "iaf": self.iaf,

            "alpha_amplitude":
                mean_amplitude,

            "alpha_phase":
                mean_phase,

            "alpha_stability":
                phase_consistency,
        }