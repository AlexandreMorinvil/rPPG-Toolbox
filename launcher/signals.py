"""Heart-rate estimation mirroring evaluation/post_process.py (scipy optional)."""

from functools import lru_cache

import numpy as np

try:
    import scipy.signal as _scipy_signal
except ImportError:  # pragma: no cover - optional dependency
    _scipy_signal = None

LOW_HZ = 0.6
HIGH_HZ = 3.3


@lru_cache(maxsize=16)
def _detrend_operator(length, smoothing=100):
    # Smoothness-priors detrending (Tarvainen et al.), as in the toolbox.
    identity = np.eye(length)
    second = np.zeros((length - 2, length))
    rows = np.arange(length - 2)
    second[rows, rows] = 1
    second[rows, rows + 1] = -2
    second[rows, rows + 2] = 1
    return identity - np.linalg.inv(identity + smoothing ** 2 * second.T @ second)


def detrend(signal, smoothing=100):
    signal = np.asarray(signal, dtype=np.float64)
    if signal.shape[-1] < 3:
        return signal - signal.mean(axis=-1, keepdims=True)
    if signal.shape[-1] > 2400:
        # Long signals: subtract a 2 s moving average instead of a dense solve.
        kernel = np.ones(61) / 61
        padded = np.pad(signal, (30, 30), mode="edge")
        return signal - np.convolve(padded, kernel, mode="valid")
    operator = _detrend_operator(signal.shape[-1], smoothing)
    return signal @ operator.T


def bandpass(signal, fs):
    signal = np.asarray(signal, dtype=np.float64)
    if _scipy_signal is not None and signal.shape[-1] > 9:
        b, a = _scipy_signal.butter(1, [LOW_HZ / fs * 2, HIGH_HZ / fs * 2], btype="bandpass")
        return _scipy_signal.filtfilt(b, a, signal, axis=-1)
    spectrum = np.fft.rfft(signal, axis=-1)
    freqs = np.fft.rfftfreq(signal.shape[-1], 1 / fs)
    spectrum[..., (freqs < LOW_HZ) | (freqs > HIGH_HZ)] = 0
    return np.fft.irfft(spectrum, n=signal.shape[-1], axis=-1)


def pulse_waveform(signal, fs, is_diff):
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    if is_diff:
        signal = np.cumsum(signal)
    return bandpass(detrend(signal), fs)


def spectrum(waveform, fs):
    n = len(waveform)
    nfft = 1 << max(0, (n - 1)).bit_length()
    power = np.abs(np.fft.rfft(waveform, n=nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, 1 / fs)
    return freqs, power


def fft_hr(waveform, fs):
    freqs, power = spectrum(waveform, fs)
    mask = (freqs >= LOW_HZ) & (freqs <= HIGH_HZ)
    if not np.any(mask) or not np.any(power[mask]):
        return float("nan")
    return float(freqs[mask][np.argmax(power[mask])] * 60)


def snr_db(waveform, fs, hr_bpm):
    if not np.isfinite(hr_bpm):
        return float("nan")
    freqs, power = spectrum(waveform, fs)
    f0 = hr_bpm / 60
    deviation = 6 / 60
    signal_mask = (np.abs(freqs - f0) <= deviation) | (np.abs(freqs - 2 * f0) <= deviation)
    band = (freqs >= LOW_HZ) & (freqs <= HIGH_HZ)
    signal_power = power[signal_mask].sum()
    noise_power = power[band & ~signal_mask].sum()
    if signal_power <= 0 or noise_power <= 0:
        return float("nan")
    return float(10 * np.log10(signal_power / noise_power))


def downsample(values, limit=4000):
    values = np.asarray(values, dtype=np.float64)
    if len(values) <= limit:
        return values
    index = np.linspace(0, len(values) - 1, limit).astype(int)
    return values[index]


def finite_list(values, digits=4):
    return [None if not np.isfinite(v) else round(float(v), digits) for v in np.asarray(values, dtype=np.float64)]


def looks_like_heart_rate(values):
    """True for label series in beats/min (vHRM) rather than a pulse waveform."""
    values = np.asarray(values, dtype=np.float64)
    values = values[..., 0] if values.ndim > 1 else values
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return False
    median = float(np.median(finite))
    return 25 <= median <= 250 and float(np.std(finite)) < 0.35 * median
