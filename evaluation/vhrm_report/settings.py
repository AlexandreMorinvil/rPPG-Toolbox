"""Analysis parameters of the report (the jzc_rppg_analysis values). Each report saves the values it used."""

import copy

DEFAULTS = {
    "inference": {
        "clip_overlap": 0.75,            # clips start every chunk_length * (1 - overlap) frames
        "overlap_window_floor": 1e-3,
        "zscore_clips": True,            # every predicted clip is z-scored before the overlap-add
    },
    "signal": {
        "bandpass_hz": [0.70, 3.50],     # 42-210 bpm
        "bandpass_order": 4,
    },
    "hr": {
        "window_sec": 8.0,
        "step_sec": 1.0,
        "welch_nperseg": 256,
        "welch_nfft": 2048,
        "tracking": True,                # causal Viterbi tracking of the spectral peak
        "tracking_options": {            # evaluation/signal_analysis.TRACKING_DEFAULTS
            "max_change_bpm_per_s": 8.0,
            "transition_weight": 2.0,
            "forgetting": 0.90,
            "acquisition_max_change_bpm_per_s": 60.0,
            "acquisition_transition_weight": 0.15,
            "acquisition_ramp_windows": 8,
            "acquisition_min_windows": 3,
            "confidence_threshold": 0.20,
            "confident_windows_required": 3,
        },
        "within_bpm": 5.0,
    },
    "beats": {
        "min_prominence_std": 0.10,      # candidate pulse peaks: prominence >= 0.1 x signal std
        "refractory_fraction": 0.60,     # two beats are >= 0.6 x the expected period (tracked HR) apart
    },
    "ibi": {
        "range_sec": [0.30, 2.00],       # plausible inter-beat intervals (30-200 bpm)
        "median_window": 11,             # beats in the running median
        "max_relative_deviation": 0.25,  # reject IBIs more than 25 % away from the running median
        "max_gap_sec": 2.0,              # no RR comparison where the estimate has no beat this close
        "rpeak_margin_sec": 5.0,         # R-peaks this far outside the segment still form its edge intervals
    },
    "hrv": {
        "window_sec": 30.0,
        "step_sec": 1.0,
        "min_intervals": 15,             # valid IBIs needed in a window
    },
}


def analysis_settings(overrides=None):
    """DEFAULTS merged with a {section: {key: value}} dictionary."""
    settings = copy.deepcopy(DEFAULTS)
    for section, values in (overrides or {}).items():
        settings.setdefault(section, {}).update(values)
    return settings
