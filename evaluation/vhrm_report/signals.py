"""From a pulse signal to heart rate, beats, inter-beat intervals (RR) and HRV.

The same interval cleaning and HRV code is applied to the rPPG beats and to the ECG R-peaks,
so the estimate and the ground truth are always computed the same way.
"""

import bisect

import numpy as np
import pandas as pd
from scipy import signal as sps


def fill_nonfinite(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    finite = np.isfinite(values)
    if not finite.all() and finite.sum() >= 2:
        index = np.arange(len(values))
        values[~finite] = np.interp(index[~finite], index[finite], values[finite])
    return np.nan_to_num(values)


def prepare_bvp(bvp, fs, band_hz, order, is_derivative=False):
    """Detrend and zero-phase band-pass the predicted signal (integrated first if it is a derivative)."""
    values = fill_nonfinite(bvp)
    if is_derivative:
        values = np.cumsum(values)
    sos = sps.butter(order, band_hz, btype="bandpass", fs=fs, output="sos")
    return sps.sosfiltfilt(sos, sps.detrend(values))


def heart_rate_windows(filtered, fs, hr, band_hz):
    """Sliding-window Welch PSD peak, tracked with the toolbox's causal Viterbi tracker.

    Returns window centre times, the tracked HR and the untracked (independent PSD-peak) HR.
    """
    from evaluation.signal_analysis import track_hr  # rPPG-Toolbox

    size = int(round(hr["window_sec"] * fs))
    step = max(1, int(round(hr["step_sec"] * fs)))
    starts = np.arange(0, len(filtered) - size + 1, step)
    spectra = []
    for start in starts:
        freqs, power = sps.welch(filtered[start:start + size], fs=fs,
                                 nperseg=min(size, hr["welch_nperseg"]), nfft=hr["welch_nfft"])
        in_band = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
        spectra.append(power[in_band])
    bins = freqs[in_band] * 60
    spectra = np.asarray(spectra)
    times = (starts + size / 2) / fs
    untracked = bins[np.argmax(spectra, axis=1)]
    tracked = track_hr(spectra, bins, times, **hr["tracking_options"]) if hr["tracking"] else untracked
    return {"times": times, "hr": np.asarray(tracked, dtype=np.float64), "hr_untracked": untracked}


def detect_beats(filtered, fs, hr_times, hr_bpm, beats):
    """Pulse-peak times (s), guided by the tracked heart rate.

    Candidate peaks are taken in order of prominence; a candidate closer than
    refractory_fraction x the expected beat period (60 / tracked HR) to an accepted beat is
    dropped (dicrotic notches, noise ripples). Peak times are refined by parabolic interpolation,
    because 30 FPS alone only resolves 33 ms.
    """
    prominence = beats["min_prominence_std"] * float(np.std(filtered))
    candidates, properties = sps.find_peaks(filtered, prominence=max(prominence, 1e-12))
    if candidates.size == 0:
        return np.asarray([])
    expected_period = 60.0 / np.interp(candidates / fs, hr_times, hr_bpm)
    refractory = beats["refractory_fraction"] * expected_period
    kept = []
    for index in np.argsort(-properties["prominences"]):
        time = candidates[index] / fs
        position = bisect.bisect_left(kept, (time, 0))
        if position > 0 and time - kept[position - 1][0] < refractory[index]:
            continue
        if position < len(kept) and kept[position][0] - time < refractory[index]:
            continue
        kept.insert(position, (time, candidates[index]))
    peaks = np.asarray([sample for _, sample in kept])
    refined = peaks.astype(np.float64)
    inner = (peaks > 0) & (peaks < len(filtered) - 1)
    left, centre, right = filtered[peaks[inner] - 1], filtered[peaks[inner]], filtered[peaks[inner] + 1]
    curvature = left - 2 * centre + right
    with np.errstate(divide="ignore", invalid="ignore"):
        offset = np.where(curvature != 0, 0.5 * (left - right) / curvature, 0.0)
    refined[inner] += np.clip(offset, -0.5, 0.5)
    return refined / fs


def intervals(beat_times, ibi):
    """Inter-beat intervals (s) stamped at the ending beat, with a validity mask.

    An interval is rejected outside `range_sec`, or when it differs by more than
    `max_relative_deviation` from the running median of its neighbours (missed / extra beats).
    """
    beat_times = np.asarray(beat_times, dtype=np.float64)
    values = np.diff(beat_times)
    valid = (values >= ibi["range_sec"][0]) & (values <= ibi["range_sec"][1])
    median = (pd.Series(np.where(valid, values, np.nan))
              .rolling(ibi["median_window"], center=True, min_periods=3).median().to_numpy())
    with np.errstate(invalid="ignore"):
        valid &= np.isfinite(median) & (np.abs(values - median) <= ibi["max_relative_deviation"] * median)
    return {"times": beat_times[1:], "ibi": values, "valid": valid}


def hrv_values(times, values, valid, start, end, min_intervals):
    """RMSSD and SDNN (ms) of the valid intervals stamped inside [start, end)."""
    inside = (times >= start) & (times < end)
    usable = inside & valid
    if usable.sum() < min_intervals:
        return np.nan, np.nan
    # Successive differences only between two consecutive valid intervals.
    pairs = usable[1:] & usable[:-1]
    if pairs.sum() < min_intervals - 1:
        return np.nan, np.nan
    rmssd = np.sqrt(np.mean(np.square(np.diff(values)[pairs]))) * 1000
    sdnn = np.std(values[usable], ddof=1) * 1000
    return float(rmssd), float(sdnn)


def hrv_windows(series, duration, hrv):
    size = hrv["window_sec"]
    centres = np.arange(size / 2, duration - size / 2 + 1e-9, hrv["step_sec"])
    values = [hrv_values(series["times"], series["ibi"], series["valid"], centre - size / 2,
                         centre + size / 2, hrv["min_intervals"]) for centre in centres]
    values = np.asarray(values, dtype=np.float64).reshape(-1, 2)
    return {"times": centres, "rmssd": values[:, 0], "sdnn": values[:, 1]}


def interpolate(source_times, source_values, query_times, max_gap):
    """Linear interpolation; NaN outside the data and inside gaps longer than max_gap."""
    source_times = np.asarray(source_times, dtype=np.float64)
    source_values = np.asarray(source_values, dtype=np.float64)
    query_times = np.asarray(query_times, dtype=np.float64)
    result = np.full(query_times.shape, np.nan)
    if source_times.size < 2:
        return result
    right = np.clip(np.searchsorted(source_times, query_times), 1, source_times.size - 1)
    inside = ((query_times >= source_times[0]) & (query_times <= source_times[-1])
              & (source_times[right] - source_times[right - 1] <= max_gap))
    result[inside] = np.interp(query_times[inside], source_times, source_values)
    return result
