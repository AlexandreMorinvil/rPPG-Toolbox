"""Pulse-signal analysis shared by the launcher viewer and the saved highlight plots.

Mirrors evaluation/post_process.py (detrend, 0.6-3.3 Hz band-pass, FFT peak) with NumPy
only; SciPy is used when available. No torch import, so the launcher can use it as is.
"""

from functools import lru_cache

import numpy as np

try:
    import scipy.signal as _scipy_signal
except ImportError:  # pragma: no cover - optional dependency
    _scipy_signal = None

LOW_HZ = 0.6
HIGH_HZ = 3.3

# Causal Viterbi HR tracking (exploration/deep_learning_test/3-2_test_AI_models_jzc17.ipynb).
TRACKING_DEFAULTS = {
    "max_change_bpm_per_s": 8.0,
    "transition_weight": 2.0,
    "forgetting": 0.90,
    "acquisition_max_change_bpm_per_s": 60.0,
    "acquisition_transition_weight": 0.15,
    "acquisition_ramp_windows": 8,
    "acquisition_min_windows": 3,
    "confidence_threshold": 0.20,
    "confident_windows_required": 3,
}


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


def fill_nonfinite(signal):
    """Linearly interpolate NaN/inf samples so filters do not propagate them."""
    signal = np.asarray(signal, dtype=np.float64).reshape(-1).copy()
    finite = np.isfinite(signal)
    if finite.all() or finite.sum() < 2:
        return np.nan_to_num(signal) if not finite.all() else signal
    index = np.arange(len(signal))
    signal[~finite] = np.interp(index[~finite], index[finite], signal[finite])
    return signal


def pulse_waveform(signal, fs, is_diff):
    signal = fill_nonfinite(signal)
    if is_diff:
        signal = np.cumsum(signal)
    return bandpass(detrend(signal), fs)


def zscore(values):
    values = np.asarray(values, dtype=np.float64)
    centered = values - np.mean(values) if values.size else values
    std = np.std(centered) if values.size else 0.0
    return centered / std if std > 1e-12 else centered


def spectrum(waveform, fs, nfft=None):
    n = len(waveform)
    nfft = nfft or 1 << max(0, (n - 1)).bit_length()
    power = np.abs(np.fft.rfft(waveform, n=nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, 1 / fs)
    return freqs, power


def fft_hr(waveform, fs):
    freqs, power = spectrum(waveform, fs)
    mask = (freqs >= LOW_HZ) & (freqs <= HIGH_HZ)
    if not np.any(mask) or not np.any(power[mask]):
        return float("nan")
    return float(freqs[mask][np.argmax(power[mask])] * 60)


def band_spectrum(waveform, fs):
    """HR-band power and its bins in beats/min (one tracking observation)."""
    freqs, power = spectrum(waveform, fs)
    mask = (freqs >= LOW_HZ) & (freqs <= HIGH_HZ)
    return freqs[mask] * 60, power[mask]


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


def beats(waveform, fs):
    """Pulse peak indices (same rule as the vHRM IBI metrics: <=200 bpm, 10 % std prominence)."""
    waveform = np.asarray(waveform, dtype=np.float64)
    if waveform.size < 3:
        return np.asarray([], dtype=int)
    distance = max(1, int(fs * 60 / 200))
    prominence = max(np.std(waveform) * 0.1, np.finfo(np.float64).eps)
    if _scipy_signal is not None:
        peaks, _ = _scipy_signal.find_peaks(waveform, distance=distance, prominence=prominence)
        return peaks
    # Fallback: local maxima above the mean, thinned to the minimum distance.
    candidates = np.where((waveform[1:-1] > waveform[:-2]) & (waveform[1:-1] >= waveform[2:])
                          & (waveform[1:-1] > waveform.mean()))[0] + 1
    kept = []
    for index in candidates:
        if not kept or index - kept[-1] >= distance:
            kept.append(index)
        elif waveform[index] > waveform[kept[-1]]:
            kept[-1] = index
    return np.asarray(kept, dtype=int)


def beat_rate(peaks, fs):
    """Mean heart rate in beats/min from inter-beat intervals."""
    peaks = np.asarray(peaks)
    if peaks.size < 2:
        return float("nan")
    return float(60.0 / np.mean(np.diff(peaks) / fs))


def spectrogram(waveform, fs, window_s=10.0, step_s=1.0, fmax=4.0, max_columns=300):
    """Column-normalized power over time; returns (centre times, freqs in Hz, power[freq, time])."""
    waveform = np.asarray(waveform, dtype=np.float64)
    size = min(len(waveform), max(9, int(round(window_s * fs))))
    if size < 9:
        return np.asarray([]), np.asarray([]), np.zeros((0, 0))
    step = max(1, int(round(step_s * fs)))
    starts = np.arange(0, len(waveform) - size + 1, step)
    if len(starts) > max_columns:
        starts = starts[np.linspace(0, len(starts) - 1, max_columns).astype(int)]
    nfft = max(512, 1 << (size - 1).bit_length())
    taper = np.hanning(size)
    freqs = np.fft.rfftfreq(nfft, 1 / fs)
    keep = freqs <= fmax
    columns = []
    for start in starts:
        segment = waveform[start:start + size]
        power = np.abs(np.fft.rfft((segment - segment.mean()) * taper, n=nfft)) ** 2
        power = power[keep]
        columns.append(power / (power.max() or 1))
    times = (starts + size / 2) / fs
    return times, freqs[keep], np.stack(columns, axis=1)


def _state_confidence(scores):
    probabilities = np.exp(scores - np.max(scores))
    probabilities /= np.maximum(probabilities.sum(), 1e-12)
    if probabilities.size < 2:
        return 1.0
    best_two = np.partition(probabilities, -2)[-2:]
    return float(best_two[-1] - best_two[-2])


def track_hr(power_spectra, bpm_bins, times, **options):
    """Causal Viterbi HR tracking over per-window band spectra.

    Each estimate only uses the current and earlier windows. A fast "acquisition" phase allows
    large jumps until the tracker is confident, then changes are limited to max_change_bpm_per_s.
    Returns the tracked HR per window (beats/min).
    """
    settings = {**TRACKING_DEFAULTS, **options}
    power_spectra = np.asarray(power_spectra, dtype=np.float64)
    bpm_bins = np.asarray(bpm_bins, dtype=np.float64)
    if power_spectra.ndim != 2 or not len(power_spectra) or not bpm_bins.size:
        return np.asarray([])
    if len(power_spectra) == 1:
        return bpm_bins[np.argmax(power_spectra, axis=1)]
    times = np.asarray(times, dtype=np.float64)

    normalized = power_spectra / np.maximum(power_spectra.max(axis=1, keepdims=True), 1e-12)
    emission = np.log(np.maximum(np.nan_to_num(normalized), 1e-12))
    tracked = np.zeros(len(power_spectra), dtype=int)
    scores = emission[0] - emission[0].max()
    tracked[0] = int(np.argmax(scores))
    changes = np.abs(bpm_bins[:, None] - bpm_bins[None, :])
    locked = False
    confident = 1 if _state_confidence(scores) >= settings["confidence_threshold"] else 0

    for index in range(1, len(power_spectra)):
        elapsed = max(float(times[index] - times[index - 1]), 1e-6)
        ramp = 1.0 if locked else min(index / settings["acquisition_ramp_windows"], 1.0)
        rate = settings["acquisition_max_change_bpm_per_s"] + ramp * (
            settings["max_change_bpm_per_s"] - settings["acquisition_max_change_bpm_per_s"])
        weight = settings["acquisition_transition_weight"] + ramp * (
            settings["transition_weight"] - settings["acquisition_transition_weight"])
        allowed = max(rate * elapsed, 1e-6)
        penalty = weight * (np.maximum(changes - allowed, 0.0) / allowed) ** 2
        candidates = settings["forgetting"] * scores[None, :] - penalty
        scores = emission[index] + candidates.max(axis=1)
        scores -= scores.max()
        tracked[index] = int(np.argmax(scores))
        if not locked:
            confident = confident + 1 if _state_confidence(scores) >= settings["confidence_threshold"] else 0
            if index + 1 >= settings["acquisition_min_windows"] and \
                    confident >= settings["confident_windows_required"]:
                locked = True
    return bpm_bins[tracked]


def _finite_mean(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def recording_windows(prediction, fs, window_s, step_s, pred_diff, label=None, label_diff=False,
                      measured_hr=None, truth="measured", track=False):
    """Per-window heart rate of one recording.

    Rows: start/end (s), pred (FFT HR of the prediction), label_hr (FFT HR of the PPG label),
    measured_hr (mean measured HR, e.g. Zephyr), gt (the reference: measured when available and
    truth == "measured", else label), snr (dB, around the reference), tracked (when track=True).
    """
    prediction = fill_nonfinite(prediction)
    n = len(prediction)
    size = min(int(round(window_s * fs)) if window_s else n, n)
    step = max(int(round(step_s * fs)) if step_s else size, 1)
    if size < 9:
        return []
    use_measured = measured_hr is not None and truth == "measured"
    rows, spectra, bins = [], [], None
    for start in range(0, n - size + 1, step):
        segment = slice(start, start + size)
        wave = pulse_waveform(prediction[segment], fs, pred_diff)
        label_hr = fft_hr(pulse_waveform(label[segment], fs, label_diff), fs) if label is not None else float("nan")
        measured = _finite_mean(measured_hr[segment]) if measured_hr is not None else float("nan")
        reference = measured if use_measured else label_hr
        rows.append({"start": start / fs, "end": (start + size) / fs, "pred": fft_hr(wave, fs),
                     "gt": reference, "label_hr": label_hr, "measured_hr": measured,
                     "snr": snr_db(wave, fs, reference)})
        if track:
            bins, power = band_spectrum(wave, fs)
            spectra.append(power)
    if track and spectra:
        tracked = track_hr(spectra, bins, [(row["start"] + row["end"]) / 2 for row in rows])
        for row, value in zip(rows, tracked):
            row["tracked"] = float(value)
    return rows


def window_metrics(rows, key="pred"):
    """MAE/RMSE/bias/MAPE/Pearson/SNR/max jump/% within 5 bpm of rows from recording_windows."""
    pairs = np.array([(row["gt"], row[key]) for row in rows
                      if key in row and np.isfinite(row["gt"]) and np.isfinite(row[key])], dtype=np.float64)
    if not len(pairs):
        return {"windows": 0}
    error = pairs[:, 1] - pairs[:, 0]
    result = {
        "windows": int(len(pairs)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "bias": float(np.mean(error)),
        "mean_gt": float(np.mean(pairs[:, 0])),
        "mean_pred": float(np.mean(pairs[:, 1])),
        "max_jump": max_jump([row[key] for row in rows if key in row]),
        "within5": within(error, 5.0),
    }
    nonzero = pairs[:, 0] != 0
    if nonzero.any():
        result["mape"] = float(np.mean(np.abs(error[nonzero] / pairs[nonzero, 0])) * 100)
    if len(pairs) >= 3 and np.std(pairs[:, 0]) > 0 and np.std(pairs[:, 1]) > 0:
        result["pearson"] = float(np.corrcoef(pairs[:, 0], pairs[:, 1])[0, 1])
    snrs = [row["snr"] for row in rows if np.isfinite(row["snr"])]
    if snrs:
        result["snr"] = float(np.mean(snrs))
    return {k: v for k, v in result.items() if not (isinstance(v, float) and not np.isfinite(v))}


def max_jump(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.max(np.abs(np.diff(values)))) if values.size > 1 else float("nan")


def within(errors, tolerance=5.0):
    """Percentage of finite errors with |error| <= tolerance."""
    errors = np.asarray(errors, dtype=np.float64)
    errors = errors[np.isfinite(errors)]
    return float(np.mean(np.abs(errors) <= tolerance) * 100) if errors.size else float("nan")


def invalid_spans(valid, fs):
    """[(start_s, end_s)] runs where the boolean mask is False (e.g. no valid PPG label)."""
    invalid = ~np.asarray(valid, dtype=bool)
    if not invalid.any():
        return []
    edges = np.flatnonzero(np.diff(np.concatenate(([0], invalid.astype(int), [0]))))
    return [(float(a / fs), float(b / fs)) for a, b in zip(edges[::2], edges[1::2])]


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
