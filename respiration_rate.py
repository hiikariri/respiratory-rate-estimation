"""
Estimate breathing rate from a single-lead ECG.

It works by tracking how the R-peak height rises and falls with each breath,
which gives a respiration signal pulled straight out of the ECG (this is called
ECG-derived respiration, or EDR). R-peak detection is reused from the sibling
``ECG`` library (``ecg_processor``).

How to use
----------
    from respiration_rate import respiration_rate

    res = respiration_rate(ecg, fs)     # ecg = samples, fs = sampling rate (Hz)
    res.rate                            # breathing rate, breaths per minute

What you get back (an ``RRResult``):
    res.rate            overall breathing rate (breaths/min)
    res.window_rates    rate for each window (default 60 s)
    res.edr             the breathing signal extracted from the ECG
    res.r_peaks         heartbeat positions used

By default the rate is read from the dominant frequency of the breathing signal
(method="psd"). The notebook's original breath-counting method is also
available:
    res = respiration_rate(ecg, fs, method="peak")

Other handy options: window=32 (window length, s), method="psd"/"peak".
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.signal import find_peaks, welch

# --- reuse the sibling ECG library for R-peak detection / band-pass filtering --
_ECG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ECG"
)
if _ECG_DIR not in sys.path:
    sys.path.insert(0, _ECG_DIR)
from ecg_processor import bandpass_filter, detect_r_peaks  # noqa: E402

__all__ = [
    "compute_edr",
    "count_breath_peaks",
    "respiration_rate_from_edr",
    "respiration_rate",
    "WindowRR",
    "RRResult",
    # defaults (exposed so callers/tests can introspect the algorithm)
    "EDR_LOWCUT",
    "EDR_HIGHCUT",
    "MIN_BREATH_INTERVAL",
    "HEIGHT_FRAC",
    "PROMINENCE_FRAC",
    "RESP_BAND",
]

# --- algorithm defaults (taken from respiratory_rate_estimation.ipynb) --------
EDR_LOWCUT = 5.0          # Hz, QRS band-pass low cut for amplitude extraction
EDR_HIGHCUT = 40.0        # Hz, QRS band-pass high cut
MIN_BREATH_INTERVAL = 1.17  # s, minimum spacing between breaths (-> max ~51 bpm)
HEIGHT_FRAC = 0.35        # peak height threshold = HEIGHT_FRAC * mean(|edr window|)
PROMINENCE_FRAC = 0.075   # prominence threshold = frac * signal_range
RESP_BAND = (10 / 60.0, 30 / 60.0)  # Hz (10–30 brpm), plausible respiration band for the PSD method
WINDOW = 32.0             # s, analysis window
MIN_R_PEAKS = 4           # need at least this many R peaks to build an EDR signal


# =============================================================================
# EDR signal
# =============================================================================
def compute_edr(
    ecg,
    fs,
    *,
    lowcut: float = EDR_LOWCUT,
    highcut: float = EDR_HIGHCUT,
    detector: str = "pan_tompkins",
    r_peaks: Optional[np.ndarray] = None,
    interp: str = "pchip",
    normalize: bool = True,
):
    """Derive a respiration signal from ECG R-peak amplitude modulation.

    Parameters
    ----------
    ecg : array_like
        Raw ECG samples (single lead).
    fs : float
        Sampling frequency, Hz.
    lowcut, highcut : float
        Band-pass band (Hz) applied before amplitude extraction. R-peak
        amplitudes are read from this filtered signal so that baseline wander
        does not contaminate the respiratory modulation.
    detector : str
        R-peak detector passed to :func:`ecg_processor.detect_r_peaks`
        (``"pan_tompkins"`` or ``"wavelet"``). Ignored if ``r_peaks`` is given.
    r_peaks : array_like, optional
        Pre-computed R-peak sample indices. If omitted they are detected.
    interp : str
        Interpolation of the irregular R-amplitude series onto a uniform grid:
        ``"pchip"`` (shape-preserving, no overshoot -- recommended) or
        ``"cubic"`` (the natural cubic spline used in the notebook; can
        overshoot near amplitude jumps and distort the respiration estimate).
    normalize : bool
        z-score the EDR signal (recommended; the peak-height threshold assumes
        it).

    Returns
    -------
    edr_time : np.ndarray
        Time axis (s) of the EDR signal, uniformly sampled at ``fs`` and
        spanning the first to last detected R peak.
    edr : np.ndarray
        The EDR signal (z-scored when ``normalize``).
    r_peaks : np.ndarray
        R-peak sample indices used.

    Raises
    ------
    ValueError
        If fewer than :data:`MIN_R_PEAKS` R peaks are available.
    """
    ecg = np.asarray(ecg, dtype=float)
    dt = 1.0 / fs

    filtered = bandpass_filter(ecg, lowcut, highcut, fs)

    if r_peaks is None:
        r_peaks = detect_r_peaks(filtered, fs, method=detector)
    r_peaks = np.asarray(r_peaks, dtype=int)
    r_peaks = r_peaks[(r_peaks >= 0) & (r_peaks < ecg.size)]
    r_peaks = np.unique(r_peaks)

    if r_peaks.size < MIN_R_PEAKS:
        raise ValueError(
            f"only {r_peaks.size} R peak(s) detected; need >= {MIN_R_PEAKS} "
            "to derive a respiration signal"
        )

    r_loc = r_peaks * dt            # R-peak times (s)
    r_amp = filtered[r_peaks]       # R-peak amplitudes (the modulated series)

    # Interpolate the (irregular) R-amplitude series onto a uniform grid so it
    # can be treated as a continuous respiration waveform. PCHIP is preferred:
    # unlike a natural cubic spline it does not overshoot at amplitude jumps
    # (e.g. an ectopic beat), which would otherwise create a spurious large
    # peak that inflates the prominence threshold and suppresses real breaths.
    if interp == "pchip":
        interpolator = PchipInterpolator(r_loc, r_amp)
    elif interp == "cubic":
        interpolator = CubicSpline(r_loc, r_amp)
    else:
        raise ValueError(f"unknown interp {interp!r}; use 'pchip' or 'cubic'")
    edr_time = np.arange(r_loc[0], r_loc[-1], dt)
    edr = interpolator(edr_time)

    if normalize:
        std = edr.std()
        edr = (edr - edr.mean()) / std if std > 0 else edr - edr.mean()

    return edr_time, edr, r_peaks


# =============================================================================
# Breath peak detection
# =============================================================================
def count_breath_peaks(
    edr_window,
    fs,
    *,
    min_breath_interval: float = MIN_BREATH_INTERVAL,
    height_frac: float = HEIGHT_FRAC,
    prominence_frac: float = PROMINENCE_FRAC,
):
    """Return indices of breath peaks within one EDR window using scipy's find_peaks."""
    edr_window = np.asarray(edr_window, dtype=float)
    if edr_window.size < 3:
        return np.array([], dtype=int)

    # 1. Height threshold based on mean absolute amplitude
    height_threshold = height_frac * np.mean(np.abs(edr_window))

    # 2. Prominence threshold based on the signal's peak-to-peak range
    # This is more robust than a 2-pass search because it's immune to finding zero initial peaks
    signal_range = np.ptp(edr_window) 
    prominence_threshold = prominence_frac * signal_range

    # 3. Distance threshold
    # Relaxed slightly (to 75% of minimum) to favor prominence over strict MPD locking
    distance = max(1, int(round((min_breath_interval * 0.75) * fs)))

    peaks, _ = find_peaks(
        edr_window,
        distance=distance,
        height=height_threshold,
        prominence=prominence_threshold,
    )
    return peaks


def _psd_rate(edr_window, fs, band=RESP_BAND):
    """Dominant respiration frequency (-> bpm) of a window via Welch PSD."""
    edr_window = np.asarray(edr_window, dtype=float)
    edr_window = edr_window - edr_window.mean()
    n = edr_window.size
    if n < 8:
        return float("nan")
    nperseg = min(n, max(64, n))  # one segment over the whole window
    f, pxx = welch(edr_window, fs=fs, nperseg=nperseg)
    mask = (f >= band[0]) & (f <= band[1])
    if not mask.any():
        return float("nan")
    f_peak = f[mask][int(np.argmax(pxx[mask]))]
    return float(f_peak * 60.0)


# =============================================================================
# Results containers
# =============================================================================
@dataclass
class WindowRR:
    """Respiratory rate for one analysis window."""
    index: int
    start: float          # s
    end: float            # s
    n_breaths: int        # breaths detected (peak method) in the window
    rate: float           # breaths/min

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class RRResult:
    """Outcome of :func:`respiration_rate`."""
    rate: float                       # overall breaths/min
    fs: float
    method: str
    windows: List[WindowRR] = field(default_factory=list)
    edr_time: np.ndarray = field(default_factory=lambda: np.array([]))
    edr: np.ndarray = field(default_factory=lambda: np.array([]))
    r_peaks: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def window_rates(self) -> np.ndarray:
        return np.array([w.rate for w in self.windows], dtype=float)

    @property
    def window_centers(self) -> np.ndarray:
        return np.array([(w.start + w.end) / 2 for w in self.windows], dtype=float)


# =============================================================================
# Windowed respiratory-rate estimation
# =============================================================================
def _make_windows(t_start, t_end, window, min_duration):
    """Non-overlapping ``window``-second spans covering [t_start, t_end).

    A short trailing remainder is kept only if it is at least ``min_duration``
    long (its rate is normalised by its actual duration). If the whole span is
    shorter than one window, a single window covers it.
    """
    spans = []
    s = t_start
    while s < t_end - 1e-9:
        e = min(s + window, t_end)
        if (e - s) >= min_duration or not spans:
            spans.append((s, e))
        s += window
    return spans or [(t_start, t_end)]


def respiration_rate_from_edr(
    edr_time,
    edr,
    fs,
    *,
    window: float = WINDOW,
    method: str = "psd",
    min_breath_interval: float = MIN_BREATH_INTERVAL,
    height_frac: float = HEIGHT_FRAC,
    prominence_frac: float = PROMINENCE_FRAC,
    resp_band=RESP_BAND,
    min_window_duration: Optional[float] = None,
):
    """Estimate respiratory rate from an existing EDR signal.

    ``method`` is ``"peak"`` (count breath peaks, the notebook method) or
    ``"psd"`` (dominant frequency in ``resp_band`` via Welch). Returns a list of
    :class:`WindowRR` and the overall rate (mean of window rates).
    """
    edr_time = np.asarray(edr_time, dtype=float)
    edr = np.asarray(edr, dtype=float)
    if edr.size == 0:
        return [], float("nan")

    if min_window_duration is None:
        min_window_duration = min(window, 10.0)

    spans = _make_windows(edr_time[0], edr_time[-1], window, min_window_duration)
    windows: List[WindowRR] = []

    for i, (start, end) in enumerate(spans):
        mask = (edr_time >= start) & (edr_time < end)
        seg = edr[mask]
        duration = end - start
        if seg.size < 3 or duration <= 0:
            continue

        if method == "peak":
            peaks = count_breath_peaks(
                seg, fs,
                min_breath_interval=min_breath_interval,
                height_frac=height_frac,
                prominence_frac=prominence_frac,
            )
            n_breaths = int(peaks.size)
            rate = n_breaths * 60.0 / duration
        elif method == "psd":
            rate = _psd_rate(seg, fs, band=resp_band)
            n_breaths = int(round(rate * duration / 60.0)) if np.isfinite(rate) else 0
        else:
            raise ValueError(f"unknown method {method!r}; use 'peak' or 'psd'")

        windows.append(WindowRR(index=i, start=start, end=end,
                                n_breaths=n_breaths, rate=rate))

    rates = np.array([w.rate for w in windows], dtype=float)
    rates = rates[np.isfinite(rates)]
    overall = float(rates.mean()) if rates.size else float("nan")
    return windows, overall


def respiration_rate(
    ecg,
    fs,
    *,
    window: float = WINDOW,
    method: str = "psd",
    lowcut: float = EDR_LOWCUT,
    highcut: float = EDR_HIGHCUT,
    detector: str = "pan_tompkins",
    r_peaks: Optional[np.ndarray] = None,
    interp: str = "pchip",
    min_breath_interval: float = MIN_BREATH_INTERVAL,
    height_frac: float = HEIGHT_FRAC,
    prominence_frac: float = PROMINENCE_FRAC,
    resp_band=RESP_BAND,
) -> RRResult:
    """Estimate respiratory rate from a raw ECG signal.

    Full pipeline: band-pass -> R-peak detection -> R-amplitude EDR ->
    windowed breath estimation.

    Parameters mirror :func:`compute_edr` and :func:`respiration_rate_from_edr`.
    Returns an :class:`RRResult`; ``.rate`` is the overall respiratory rate in
    breaths per minute (mean of the per-window rates).
    """
    edr_time, edr, peaks = compute_edr(
        ecg, fs, lowcut=lowcut, highcut=highcut,
        detector=detector, r_peaks=r_peaks, interp=interp, normalize=True,
    )
    windows, overall = respiration_rate_from_edr(
        edr_time, edr, fs,
        window=window, method=method,
        min_breath_interval=min_breath_interval,
        height_frac=height_frac, prominence_frac=prominence_frac,
        resp_band=resp_band,
    )
    return RRResult(
        rate=overall, fs=float(fs), method=method, windows=windows,
        edr_time=edr_time, edr=edr, r_peaks=peaks,
    )