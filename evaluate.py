"""
Evaluate the EDR respiratory-rate library on the BIDMC and dataset_primer_1
datasets.

Run:
    python evaluate.py              # both datasets, peak method
    python evaluate.py --method psd
    python evaluate.py --dataset bidmc --limit 10

Per-record estimates are written to ``RR/results/`` and accuracy metrics
(MAE / RMSE / bias / correlation against ground truth) are printed.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import datasets as ds
from respiration_rate import WINDOW, respiration_rate

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _metrics(est, ref):
    """MAE, RMSE, bias and Pearson r over paired estimate/reference arrays."""
    est = np.asarray(est, dtype=float)
    ref = np.asarray(ref, dtype=float)
    ok = np.isfinite(est) & np.isfinite(ref)
    est, ref = est[ok], ref[ok]
    if est.size == 0:
        return dict(n=0, mae=np.nan, rmse=np.nan, bias=np.nan, corr=np.nan)
    err = est - ref
    corr = (np.corrcoef(est, ref)[0, 1]
            if est.size > 1 and est.std() > 0 and ref.std() > 0 else np.nan)
    return dict(n=int(est.size),
                mae=float(np.mean(np.abs(err))),
                rmse=float(np.sqrt(np.mean(err ** 2))),
                bias=float(np.mean(err)),
                corr=float(corr))


def _print_metrics(title, m):
    print(f"\n{title}")
    print("-" * len(title))
    print(f"  n        : {m['n']}")
    print(f"  MAE      : {m['mae']:.2f} bpm")
    print(f"  RMSE     : {m['rmse']:.2f} bpm")
    print(f"  Bias     : {m['bias']:+.2f} bpm (estimate - reference)")
    print(f"  Pearson r: {m['corr']:.3f}")


def evaluate_bidmc(method="peak", limit=None, window=WINDOW):
    records = ds.load_bidmc(limit=limit)
    print(f"\n=== BIDMC ({len(records)} recordings, method={method}) ===")

    rec_rows = []          # one row per recording (overall)
    win_est, win_ref = [], []   # pooled per-window pairs

    for rec in records:
        try:
            res = respiration_rate(rec.ecg, rec.fs, method=method, window=window)
        except ValueError as exc:
            print(f"  {rec.name}: skipped ({exc})")
            rec_rows.append(dict(record=rec.name, est_rate=np.nan,
                                 ref_rate=rec.reference_rate,
                                 monitor_resp=rec.monitor_resp, n_rpeaks=0))
            continue

        # Per-window reference from breath annotations, paired with estimate.
        for w in res.windows:
            ref_count = rec.breaths_in(w.start, w.end)
            win_ref.append(ref_count * 60.0 / w.duration)
            win_est.append(w.rate)

        rec_rows.append(dict(
            record=rec.name,
            est_rate=res.rate,
            ref_rate=rec.reference_rate,
            monitor_resp=rec.monitor_resp,
            n_rpeaks=int(res.r_peaks.size),
        ))

    df = pd.DataFrame(rec_rows)
    _save(df, "bidmc", method)

    rec_m = _metrics(df["est_rate"], df["ref_rate"])
    win_m = _metrics(win_est, win_ref)
    _print_metrics("Per-recording RR vs annotated reference", rec_m)
    _print_metrics(f"Per-window RR ({int(window)} s) vs annotations", win_m)
    return df


def evaluate_primer(method="peak", limit=None, window=WINDOW):
    records = ds.load_primer(limit=limit)
    print(f"\n=== dataset_primer_1 ({len(records)} recordings, method={method}) ===")

    rows = []
    for rec in records:
        try:
            res = respiration_rate(rec.ecg, rec.fs, method=method, window=window)
            est = res.rate
            n_rpeaks = int(res.r_peaks.size)
        except ValueError as exc:
            print(f"  {rec.name}: skipped ({exc})")
            est, n_rpeaks = np.nan, 0
        rows.append(dict(record=rec.name, est_rate=est,
                         ref_rate=rec.reference_rate,
                         heart_rate_ref=rec.heart_rate, n_rpeaks=n_rpeaks))

    df = pd.DataFrame(rows)
    _save(df, "primer", method)

    m = _metrics(df["est_rate"], df["ref_rate"])
    _print_metrics("RR vs metadata ground truth", m)
    return df


def _save(df, dataset, method):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{dataset}_{method}.csv")
    df.to_csv(path, index=False)
    print(f"  -> wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["bidmc", "primer", "both"],
                    default="both")
    ap.add_argument("--method", choices=["peak", "psd"], default="psd")
    ap.add_argument("--limit", type=int, default=None,
                    help="limit number of recordings (for a quick run)")
    ap.add_argument("--window", type=float, default=WINDOW,
                    help="analysis window in seconds")
    args = ap.parse_args()

    if args.dataset in ("bidmc", "both"):
        evaluate_bidmc(args.method, args.limit, args.window)
    if args.dataset in ("primer", "both"):
        evaluate_primer(args.method, args.limit, args.window)


if __name__ == "__main__":
    main()
