from scipy.signal import butter, filtfilt


def bandpass_filter(signal, fs=256, low=1, high=40, order=4):
    b, a = butter(
        N=order,
        Wn=[low, high],
        btype="bandpass",
        fs=fs
    )

    filtered_signal = filtfilt(
        b,
        a,
        signal
    )

    return filtered_signal