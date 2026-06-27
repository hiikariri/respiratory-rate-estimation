"""
Evaluate the respiratory-rate algorithm on the PhysioNet "Simultaneous
physiological measurements with five devices" dataset.

The dataset records 13 participants performing four tasks (Rest, Walking,
2-Back, Running). Each WFDB record bundles several device ECGs plus a Hexoskin
chest-band breathing rate, which we use as the reference.

Pipeline under test (RR/respiration_rate.py): single-lead ECG -> band-pass ->
R-peak detection -> ECG-derived respiration (EDR) -> windowed breathing rate.

For every analysis window we compare the estimate against the mean Hexoskin
breathing rate over the same span, and aggregate the absolute error per task
phase.

By default it sweeps the PSD respiration band's upper edge (--resp-hi 30 40 45
breaths/min) using the detector's own R-peaks. The 30 brpm edge is the library
default; the wider edges let the Walking/Running phases reach rates the default
band clips. One extra diagnostic config feeds the gold .atr R-peaks at the
widest band, isolating R-peak-detection error from the band/estimation error.

Usage
-----
    python evaluate_simultaneous.py                    # sweep 30/40/45 brpm
    python evaluate_simultaneous.py --resp-hi 30 50    # custom upper edges
    python evaluate_simultaneous.py --lead HEXOSKIN/ECG_I --limit 3
"""
from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import wfdb

from respiration_rate import compute_edr, respiration_rate_from_edr
# respiration_rate puts the sibling ECG/ dir on sys.path at import, so the SQA
# engine resolves from there.
from ecg_sqa import ECGSQAEngine  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DATA_DIR = os.path.join(
    _ROOT, "simultaneous_data",
    "simultaneous-physiological-measurements-with-five-devices-at-"
    "different-cognitive-and-physical-loads-1.0.2", "generated_data",
)
RESULTS_DIR = os.path.join(_HERE, "results")

REF_CHANNEL = "HEXOSKIN/breathing_rate"
DEFAULT_LEAD = "FAROS/ECG"
# Canonical phase order; markers in the .aux file name the start of each.
PHASE_KEYS = ["Rest", "Walking", "2-Back", "Running"]


@dataclass
class Record:
    name: str
    ecg: np.ndarray
    fs: float
    ref_br: np.ndarray            # reference breathing rate, per sample (brpm)
    phases: List[Tuple[str, float, float]]   # (label, start_s, end_s)
    gold_rpeaks: np.ndarray       # .atr beat sample indices


def _phase_label(note: str) -> Optional[str]:
    """Map an .aux note ('FAROS_Marker/Walking', 'Manual/Running') to a phase."""
    tail = note.split("/")[-1].strip()
    return tail if tail in PHASE_KEYS else None


def _read_phases(rec_base: str, fs: float, sig_len: int) -> List[Tuple[str, float, float]]:
    """Phase spans from the .aux event annotations.

    The first marker for each phase wins (Walking/2-Back have duplicate
    FAROS/Manual markers a few seconds apart). Each phase runs until the next
    phase's start, the last until end of record.
    """
    aux = wfdb.rdann(rec_base, "aux")
    starts: List[Tuple[str, float]] = []
    seen = set()
    for samp, note in zip(aux.sample, aux.aux_note):
        label = _phase_label(note)
        if label and label not in seen:
            starts.append((label, samp / fs))
            seen.add(label)
    starts.sort(key=lambda x: x[1])
    spans = []
    for i, (label, s) in enumerate(starts):
        e = starts[i + 1][1] if i + 1 < len(starts) else sig_len / fs
        spans.append((label, s, e))
    return spans


def load_record(rec_base: str, lead: str) -> Record:
    rec = wfdb.rdrecord(rec_base)
    name = os.path.basename(rec_base)
    fs = float(rec.fs)
    if lead not in rec.sig_name:
        raise ValueError(f"{name}: lead {lead!r} not found")
    ecg = rec.p_signal[:, rec.sig_name.index(lead)].astype(float)
    ref_br = rec.p_signal[:, rec.sig_name.index(REF_CHANNEL)].astype(float)
    phases = _read_phases(rec_base, fs, rec.sig_len)
    try:
        gold = np.asarray(wfdb.rdann(rec_base, "atr").sample, dtype=int)
    except Exception:
        gold = np.array([], dtype=int)
    return Record(name, ecg, fs, ref_br, phases, gold)


def _ref_rate(ref_br: np.ndarray, fs: float, start_s: float, end_s: float) -> float:
    """Mean reference breathing rate over [start_s, end_s) (brpm), NaN-safe."""
    a, b = int(round(start_s * fs)), int(round(end_s * fs))
    a, b = max(0, a), min(ref_br.size, b)
    if b <= a:
        return float("nan")
    seg = ref_br[a:b]
    seg = seg[np.isfinite(seg) & (seg > 0)]
    return float(seg.mean()) if seg.size else float("nan")


def _assign_phase(t: float, phases: List[Tuple[str, float, float]]) -> Optional[str]:
    for label, s, e in phases:
        if s <= t < e:
            return label
    return None


def compute_edrs(rec: Record) -> Dict[str, Optional[Tuple[np.ndarray, np.ndarray]]]:
    """EDR signal from detected R-peaks and from the gold .atr beats.

    R-peak detection is the expensive step, so each EDR is computed once here
    and reused across every band the sweep tries.
    """
    edrs: Dict[str, Optional[Tuple[np.ndarray, np.ndarray]]] = {}
    try:
        t, edr, _ = compute_edr(rec.ecg, rec.fs)
        edrs["detected"] = (t, edr)
    except ValueError:
        edrs["detected"] = None
    if rec.gold_rpeaks.size:
        try:
            t, edr, _ = compute_edr(rec.ecg, rec.fs, r_peaks=rec.gold_rpeaks)
            edrs["gold"] = (t, edr)
        except ValueError:
            edrs["gold"] = None
    else:
        edrs["gold"] = None
    return edrs


_TIER_RANK = {"bad": 0, "hr": 1, "diag": 2}


def _window_sqa(rec: Record, start: float, end: float, cache: dict) -> dict:
    """ECG signal quality for the segment under one analysis window.

    Runs :class:`ECGSQAEngine` on the raw ECG slice and reports whether it is
    acceptable, its iScore, and its tier ("bad"/"hr"/"diag"). Cached per
    (start, end) so the band sweep doesn't reassess identical windows.
    """
    key = (round(start, 3), round(end, 3))
    if key in cache:
        return cache[key]
    a, b = int(round(start * rec.fs)), int(round(end * rec.fs))
    a, b = max(0, a), min(rec.ecg.size, b)
    seg = rec.ecg[a:b]
    if seg.size < int(5 * rec.fs):          # too short for a meaningful verdict
        res = dict(acceptable=False, iscore=float("nan"), tier=None, stage=0)
    else:
        try:
            r = ECGSQAEngine(seg, rec.fs).assess()
            res = dict(acceptable=(r["quality"] == "acceptable"),
                       iscore=(r["iscore"] if r["iscore"] is not None else float("nan")),
                       tier=r["iscore_quality"], stage=r["stage_reached"])
        except Exception:
            res = dict(acceptable=False, iscore=float("nan"), tier=None, stage=0)
    cache[key] = res
    return res


def evaluate_config(rec: Record, edrs, *, source: str, method: str,
                    window: float, resp_band,
                    sqa_cache: Optional[dict] = None,
                    sqa_tier: str = "hr") -> pd.DataFrame:
    """Per-window estimate vs reference for one config (reuses a cached EDR).

    When ``sqa_cache`` is given, each row is tagged with the ECG quality of its
    window (``sqa_ok`` honours ``sqa_tier``: "hr" keeps any acceptable window,
    "diag" keeps only diagnostic-quality ones).
    """
    edr = edrs.get(source)
    if edr is None:
        return pd.DataFrame()
    edr_t, edr_sig = edr
    windows, _ = respiration_rate_from_edr(edr_t, edr_sig, rec.fs, window=window,
                                           method=method, resp_band=resp_band)
    min_rank = _TIER_RANK.get(sqa_tier, 1)
    rows = []
    for w in windows:
        center = 0.5 * (w.start + w.end)
        phase = _assign_phase(center, rec.phases)
        ref = _ref_rate(rec.ref_br, rec.fs, w.start, w.end)
        if phase is None or not np.isfinite(ref) or not np.isfinite(w.rate):
            continue
        row = dict(record=rec.name, phase=phase, start=w.start,
                   end=w.end, est=w.rate, ref=ref,
                   abs_err=abs(w.rate - ref), err=w.rate - ref)
        if sqa_cache is not None:
            sqa = _window_sqa(rec, w.start, w.end, sqa_cache)
            ok = sqa["acceptable"] and _TIER_RANK.get(sqa["tier"], -1) >= min_rank
            row.update(sqa_ok=bool(ok), iscore=sqa["iscore"],
                       sqa_tier=sqa["tier"], sqa_stage=sqa["stage"])
        rows.append(row)
    return pd.DataFrame(rows)


def _summary(df: pd.DataFrame) -> pd.DataFrame:
    """MAE / bias / n per phase (+ ALL) for one config's per-window table.

    If the table carries SQA tags, also report coverage (``cov``) and the
    MAE/bias over only the SQA-accepted windows (``mae_kept``/``bias_kept``).
    """
    if df.empty:
        return pd.DataFrame()
    has_sqa = "sqa_ok" in df.columns
    parts = []
    for phase in PHASE_KEYS + ["ALL"]:
        sub = df if phase == "ALL" else df[df["phase"] == phase]
        if sub.empty:
            continue
        row = dict(phase=phase, n=len(sub),
                   mae=sub["abs_err"].mean(), bias=sub["err"].mean(),
                   ref_mean=sub["ref"].mean(), est_mean=sub["est"].mean())
        if has_sqa:
            kept = sub[sub["sqa_ok"]]
            row.update(kept=len(kept), cov=len(kept) / len(sub),
                       mae_kept=(kept["abs_err"].mean() if len(kept) else float("nan")),
                       bias_kept=(kept["err"].mean() if len(kept) else float("nan")))
        parts.append(row)
    return pd.DataFrame(parts)


_FMT = {"mae": "{:.2f}".format, "bias": "{:+.2f}".format,
        "ref_mean": "{:.1f}".format, "est_mean": "{:.1f}".format,
        "cov": "{:.0%}".format, "mae_kept": "{:.2f}".format,
        "bias_kept": "{:+.2f}".format}


def _print_summary(name: str, summ: pd.DataFrame) -> None:
    if summ.empty:
        return
    if "cov" in summ.columns:        # SQA on: show coverage + gated accuracy
        cols = ["phase", "n", "kept", "cov", "mae", "mae_kept", "bias", "bias_kept"]
    else:
        cols = ["phase", "n", "mae", "bias", "ref_mean", "est_mean"]
    cols = [c for c in cols if c in summ.columns]
    print(f"\n[{name}]")
    print(summ[cols].to_string(
        index=False, formatters={k: v for k, v in _FMT.items() if k in cols}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lead", default=DEFAULT_LEAD, help=f"ECG channel (default {DEFAULT_LEAD})")
    ap.add_argument("--window", type=float, default=32.0, help="analysis window, s")
    ap.add_argument("--resp-lo", type=float, default=10.0,
                    help="respiration band lower edge, brpm (default 10)")
    ap.add_argument("--resp-hi", type=float, nargs="+", default=[30.0, 40.0, 45.0],
                    help="respiration band upper edge(s) to sweep, brpm "
                         "(default: 30 40 45; 30 is the library default)")
    ap.add_argument("--no-sqa", dest="sqa", action="store_false",
                    help="disable ECG signal-quality gating of windows")
    ap.add_argument("--sqa-tier", choices=["hr", "diag"], default="hr",
                    help="min ECG quality to keep a window: 'hr' (any acceptable) "
                         "or 'diag' (diagnostic only). Default: hr")
    ap.add_argument("--limit", type=int, default=None, help="first N records only")
    ap.set_defaults(sqa=True)
    args = ap.parse_args()

    bases = sorted({p[:-4] for p in glob.glob(os.path.join(DATA_DIR, "x*.hea"))})
    if args.limit:
        bases = bases[: args.limit]
    if not bases:
        raise SystemExit(f"no records found in {DATA_DIR}")

    lo_hz = args.resp_lo / 60.0
    configs = [(f"psd {args.resp_lo:g}-{hi:g}",
                dict(source="detected", method="psd", resp_band=(lo_hz, hi / 60.0)))
               for hi in args.resp_hi]
    widest = max(args.resp_hi)
    configs.append((f"gold psd {args.resp_lo:g}-{widest:g}",
                    dict(source="gold", method="psd",
                         resp_band=(lo_hz, widest / 60.0))))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Records: {len(bases)} | lead: {args.lead} | window: {args.window:g}s | "
          f"reference: {REF_CHANNEL}")
    print(f"PSD band sweep (brpm): lo={args.resp_lo:g}, hi={args.resp_hi}")
    print(f"SQA gating: {'on (tier >= ' + args.sqa_tier + ')' if args.sqa else 'off'}\n")

    per_config: Dict[str, List[pd.DataFrame]] = {name: [] for name, _ in configs}
    for base in bases:
        rec = load_record(base, args.lead)
        edrs = compute_edrs(rec)
        # one SQA verdict per (window span); shared across the PSD bands.
        sqa_cache: dict = {} if args.sqa else None
        msg = [rec.name]
        for name, kw in configs:
            df = evaluate_config(rec, edrs, window=args.window,
                                 sqa_cache=sqa_cache, sqa_tier=args.sqa_tier, **kw)
            per_config[name].append(df)
            mae = df["abs_err"].mean() if not df.empty else float("nan")
            msg.append(f"{name} MAE={mae:5.2f}")
        print("  " + " | ".join(msg))

    print("\n" + "=" * 72)
    title = "Per-phase accuracy (breaths/min)   MAE = mean abs error"
    if args.sqa:
        title += "   |   kept/cov = SQA-accepted windows"
    print(title)
    print("=" * 72)
    all_windows = []
    for name, _ in configs:
        df = pd.concat([d for d in per_config[name] if not d.empty],
                       ignore_index=True) if any(not d.empty for d in per_config[name]) else pd.DataFrame()
        if df.empty:
            continue
        df["config"] = name
        all_windows.append(df)
        _print_summary(name, _summary(df))

    if all_windows:
        windows_df = pd.concat(all_windows, ignore_index=True)
        win_path = os.path.join(RESULTS_DIR, "simultaneous_windows.csv")
        windows_df.to_csv(win_path, index=False)
        summ_df = pd.concat(
            [(_summary(windows_df[windows_df["config"] == n]).assign(config=n))
             for n in windows_df["config"].unique()],
            ignore_index=True)
        summ_path = os.path.join(RESULTS_DIR, "simultaneous_summary.csv")
        summ_df.to_csv(summ_path, index=False)
        print(f"\nSaved per-window results -> {win_path}")
        print(f"Saved phase summary      -> {summ_path}")


if __name__ == "__main__":
    main()
