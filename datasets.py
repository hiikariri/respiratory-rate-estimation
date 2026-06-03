"""
Loaders for the two respiration test datasets.

* BIDMC PPG & Respiration  -> CSV export under ``bidmc-ppg-and-respiration-dataset/bidmc_csv``
* dataset_primer_1         -> per-subject ``*_ecg.csv`` + ``*_metadata.json``

Each loader returns a small record object exposing the Lead-II ECG, its sampling
frequency, and the available respiratory-rate ground truth.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
BIDMC_DIR = os.path.join(_ROOT, "bidmc-ppg-and-respiration-dataset", "bidmc_csv")
PRIMER_DIR = os.path.join(_ROOT, "dataset_primer_1")


@dataclass
class BidmcRecord:
    name: str
    ecg: np.ndarray             # Lead II
    fs: float                   # Hz
    time: np.ndarray            # s
    breath_samples_ann1: np.ndarray
    breath_samples_ann2: np.ndarray
    monitor_resp: Optional[float]  # mean monitor RESP numeric (bpm), reference only

    @property
    def duration(self) -> float:
        return float(self.time[-1] - self.time[0])

    @property
    def reference_rate(self) -> float:
        """Gold-standard RR (bpm): mean breath count of the two annotators over
        the recording, the reference used by the BIDMC source paper."""
        n = np.mean([self.breath_samples_ann1.size, self.breath_samples_ann2.size])
        return n / self.duration * 60.0

    def breaths_in(self, start_s: float, end_s: float) -> float:
        """Mean annotated breath count of the two annotators in [start_s, end_s)."""
        counts = []
        for ann in (self.breath_samples_ann1, self.breath_samples_ann2):
            t = ann / self.fs
            counts.append(int(np.sum((t >= start_s) & (t < end_s))))
        return float(np.mean(counts))


@dataclass
class PrimerRecord:
    name: str
    ecg: np.ndarray             # ECG_Raw
    fs: float                   # Hz
    time: np.ndarray            # s
    reference_rate: Optional[float]   # bpm, from metadata ground truth
    heart_rate: Optional[float]       # bpm, from metadata (reference only)

    @property
    def duration(self) -> float:
        return float(self.time[-1] - self.time[0])


def _fs_from_time(time: np.ndarray) -> float:
    """Sampling frequency from a time axis.

    Uses the total span over (n-1) intervals rather than the median sample
    step: the BIDMC CSV time column is rounded to 0.01 s, so its per-sample
    diffs alternate 0.008 / 0.01 / 0.0 and the median wrongly yields 100 Hz
    instead of the true 125 Hz.
    """
    time = np.asarray(time, dtype=float)
    return float((time.size - 1) / (time[-1] - time[0]))


def load_bidmc_record(signals_path: str) -> BidmcRecord:
    name = os.path.basename(signals_path).replace("_Signals.csv", "")
    sig = pd.read_csv(signals_path)
    sig.columns = sig.columns.str.strip()
    time = sig["Time [s]"].to_numpy(dtype=float)
    ecg = sig["II"].to_numpy(dtype=float)
    fs = _fs_from_time(time)

    breaths_path = signals_path.replace("_Signals.csv", "_Breaths.csv")
    br = pd.read_csv(breaths_path)
    ann1 = br.iloc[:, 0].dropna().to_numpy(dtype=int)
    ann2 = br.iloc[:, 1].dropna().to_numpy(dtype=int)

    monitor_resp = None
    num_path = signals_path.replace("_Signals.csv", "_Numerics.csv")
    if os.path.exists(num_path):
        num = pd.read_csv(num_path)
        num.columns = num.columns.str.strip()
        if "RESP" in num.columns:
            monitor_resp = float(num["RESP"].mean())

    return BidmcRecord(name, ecg, fs, time, ann1, ann2, monitor_resp)


def load_bidmc(limit: Optional[int] = None) -> List[BidmcRecord]:
    paths = sorted(glob.glob(os.path.join(BIDMC_DIR, "bidmc_*_Signals.csv")))
    if limit:
        paths = paths[:limit]
    return [load_bidmc_record(p) for p in paths]


def load_primer_record(ecg_path: str) -> PrimerRecord:
    name = os.path.basename(ecg_path).replace("_ecg.csv", "")
    df = pd.read_csv(ecg_path)
    df.columns = df.columns.str.strip()
    time = df["Time (s)"].to_numpy(dtype=float)
    ecg = df["ECG_Raw (V)"].to_numpy(dtype=float)

    meta_path = ecg_path.replace("_ecg.csv", "_metadata.json")
    rr = hr = None
    fs = _fs_from_time(time)
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            meta = json.load(fh)
        gt = meta.get("ground_truth", {})
        rr = _to_float(gt.get("respiration_rate_bpm"))
        hr = _to_float(gt.get("heart_rate_bpm"))
        specs = meta.get("signal_specs", {}).get("ecg", {})
        if specs.get("sampling_rate_hz"):
            fs = float(specs["sampling_rate_hz"])

    return PrimerRecord(name, ecg, fs, time, rr, hr)


def load_primer(limit: Optional[int] = None) -> List[PrimerRecord]:
    paths = sorted(glob.glob(os.path.join(PRIMER_DIR, "*_ecg.csv")))
    if limit:
        paths = paths[:limit]
    return [load_primer_record(p) for p in paths]


def _to_float(value) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
