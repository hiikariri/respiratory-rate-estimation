"""
GUI for ECG-derived respiratory-rate estimation.

Load an ECG recording from a dataset, then view:
  * the ECG with detected R-peaks, and
  * the ECG-derived respiration (EDR) signal with the detected breaths,
    the estimated respiratory rate, and the error (MAE) against ground truth.

Supports both bundled dataset layouts:
  * BIDMC          -> ``*_Signals.csv`` (+ ``*_Breaths.csv`` ground truth)
  * dataset_primer -> ``*_ecg.csv``     (+ ``*_metadata.json`` ground truth)

Run:
    python rr_gui.py
"""
import re
import sys
from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from scipy.signal import welch

import datasets as ds
from respiration_rate import (respiration_rate, count_breath_peaks,
                              bandpass_filter, EDR_LOWCUT, EDR_HIGHCUT, RESP_BAND)

DEFAULT_DATASET = ds.BIDMC_DIR


# --------------------------------------------------------------------------- #
# Data helpers (bridge the two dataset layouts to one interface)
# --------------------------------------------------------------------------- #
def discover_records(folder):
    """Return ``[(kind, path, name), ...]`` for every record in ``folder``."""
    folder = Path(folder)
    recs = []
    for f in sorted(folder.glob("*_Signals.csv")):
        recs.append(("bidmc", f, f.name[: -len("_Signals.csv")]))
    for f in sorted(folder.glob("*_ecg.csv")):
        recs.append(("primer", f, f.name[: -len("_ecg.csv")]))
    return recs


def load_record(kind, path):
    if kind == "bidmc":
        return ds.load_bidmc_record(str(path))
    return ds.load_primer_record(str(path))


def short_label(name):
    """Trim the ``data_`` prefix and ``_YYYYMMDD_HHMMSS`` timestamp (primer)."""
    s = re.sub(r"_\d{8}_\d{6}$", "", name)
    return s[len("data_"):] if s.startswith("data_") else s


def window_ref_rate(rec, start, end):
    """Ground-truth respiratory rate (bpm) for a window, or NaN."""
    if isinstance(rec, ds.BidmcRecord):
        dur = end - start
        return rec.breaths_in(start, end) * 60.0 / dur if dur > 0 else np.nan
    return rec.reference_rate if rec.reference_rate else np.nan  # primer: constant


def ref_breath_times(rec):
    """Times (s) of annotated breaths for overlay, or None when unavailable."""
    if isinstance(rec, ds.BidmcRecord):
        return rec.breath_samples_ann1 / rec.fs
    return None


def breath_peak_times(res):
    """Times (s) of detected breath peaks across all windows of an RRResult."""
    times = []
    for w in res.windows:
        mask = (res.edr_time >= w.start) & (res.edr_time < w.end)
        seg, seg_t = res.edr[mask], res.edr_time[mask]
        peaks = count_breath_peaks(seg, res.fs)
        times.extend(seg_t[peaks])
    return np.asarray(times)


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #
class RRGui(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ECG-Derived Respiratory Rate")
        self.resize(1280, 900)
        self.dataset_dir = Path(DEFAULT_DATASET)
        self.records = []          # [(kind, path, name), ...]
        self._axes = None          # (ax_ecg, ax_edr) for live view cropping
        self._tmax = 0.0
        self._last_plot = None     # cached args of the last _plot, for redraws

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        controls = QtWidgets.QGridLayout()
        root.addLayout(controls)

        # Dataset folder
        controls.addWidget(QtWidgets.QLabel("Dataset folder"), 0, 0)
        self.dataset_edit = QtWidgets.QLineEdit(str(self.dataset_dir))
        controls.addWidget(self.dataset_edit, 0, 1, 1, 5)
        browse_btn = QtWidgets.QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_dataset)
        controls.addWidget(browse_btn, 0, 6)

        # Record + navigation
        controls.addWidget(QtWidgets.QLabel("Record"), 1, 0)
        self.record_combo = QtWidgets.QComboBox()
        controls.addWidget(self.record_combo, 1, 1)
        self.prev_btn = QtWidgets.QPushButton("◀ Prev")
        self.prev_btn.clicked.connect(lambda: self.step_record(-1))
        controls.addWidget(self.prev_btn, 1, 2)
        self.next_btn = QtWidgets.QPushButton("Next ▶")
        self.next_btn.clicked.connect(lambda: self.step_record(1))
        controls.addWidget(self.next_btn, 1, 3)
        self.record_combo.currentIndexChanged.connect(self._update_nav_buttons)
        refresh_btn = QtWidgets.QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_records)
        controls.addWidget(refresh_btn, 1, 4)

        # ECG source for the plot (display only; detection always uses the
        # internal 5-40 Hz band-pass regardless of this choice)
        controls.addWidget(QtWidgets.QLabel("ECG source"), 1, 5)
        self.source_combo = QtWidgets.QComboBox()
        self.source_combo.addItems(["Raw", "Filtered"])
        self.source_combo.currentIndexChanged.connect(self.update_source)
        controls.addWidget(self.source_combo, 1, 6)

        # Method + window
        controls.addWidget(QtWidgets.QLabel("Method"), 2, 0)
        self.method_combo = QtWidgets.QComboBox()
        self.method_combo.addItems(["psd", "peak"])   # psd first: more accurate
        controls.addWidget(self.method_combo, 2, 1)
        controls.addWidget(QtWidgets.QLabel("Window (s)"), 2, 2)
        self.window_spin = QtWidgets.QDoubleSpinBox()
        self.window_spin.setRange(10.0, 600.0)
        self.window_spin.setSingleStep(2.0)
        self.window_spin.setValue(32.0)
        controls.addWidget(self.window_spin, 2, 3)

        # View window (crops both plots without recomputing)
        controls.addWidget(QtWidgets.QLabel("View start (s)"), 3, 0)
        self.view_start_spin = QtWidgets.QDoubleSpinBox()
        self.view_start_spin.setRange(0.0, 1_000_000.0)
        self.view_start_spin.setSingleStep(5.0)
        controls.addWidget(self.view_start_spin, 3, 1)
        controls.addWidget(QtWidgets.QLabel("View width (s)"), 3, 2)
        self.view_width_spin = QtWidgets.QDoubleSpinBox()
        self.view_width_spin.setRange(2.0, 1_000_000.0)
        self.view_width_spin.setSingleStep(5.0)
        self.view_width_spin.setValue(60.0)
        controls.addWidget(self.view_width_spin, 3, 3)
        self.view_start_spin.valueChanged.connect(self.update_view)
        self.view_width_spin.valueChanged.connect(self.update_view)

        # Buttons
        self.plot_btn = QtWidgets.QPushButton("Estimate && Plot")
        self.plot_btn.clicked.connect(self.estimate_and_plot)
        controls.addWidget(self.plot_btn, 2, 4, 1, 1)
        self.batch_btn = QtWidgets.QPushButton("Evaluate all (MAE)")
        self.batch_btn.clicked.connect(self.evaluate_all)
        controls.addWidget(self.batch_btn, 3, 4, 1, 1)

        # Metrics
        self.metrics_label = QtWidgets.QLabel("Select a record and click Estimate && Plot.")
        self.metrics_label.setStyleSheet("font-weight: bold;")
        root.addWidget(self.metrics_label)

        # Figure
        self.figure = Figure(figsize=(12, 8), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        root.addWidget(self.toolbar)
        root.addWidget(self.canvas)

        self.refresh_records()

    # ----------------------------------------------------------------- #
    def browse_dataset(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select dataset folder", str(self.dataset_dir))
        if path:
            self.dataset_edit.setText(path)
            self.refresh_records()

    def refresh_records(self):
        self.dataset_dir = Path(self.dataset_edit.text().strip())
        self.record_combo.clear()
        if not self.dataset_dir.exists():
            self.metrics_label.setText(f"Folder not found: {self.dataset_dir}")
            self.records = []
            return
        self.records = discover_records(self.dataset_dir)
        self.record_combo.addItems([short_label(name) for _, _, name in self.records])
        if self.records:
            self.metrics_label.setText(
                f"{len(self.records)} records loaded. Pick one and Estimate && Plot.")
        else:
            self.metrics_label.setText(
                f"No '*_Signals.csv' or '*_ecg.csv' files in {self.dataset_dir}")
        self._update_nav_buttons()

    def step_record(self, delta):
        n = self.record_combo.count()
        if n == 0:
            return
        i = min(n - 1, max(0, self.record_combo.currentIndex() + delta))
        if i != self.record_combo.currentIndex():
            self.record_combo.setCurrentIndex(i)
            self.estimate_and_plot()
        self._update_nav_buttons()

    def _update_nav_buttons(self):
        n = self.record_combo.count()
        i = self.record_combo.currentIndex()
        self.prev_btn.setEnabled(n > 0 and i > 0)
        self.next_btn.setEnabled(n > 0 and i < n - 1)

    # ----------------------------------------------------------------- #
    def estimate_and_plot(self):
        i = self.record_combo.currentIndex()
        if i < 0 or i >= len(self.records):
            QtWidgets.QMessageBox.warning(self, "No record", "No record selected.")
            return

        kind, path, name = self.records[i]
        method = self.method_combo.currentText()
        window = self.window_spin.value()
        try:
            rec = load_record(kind, path)
            res = respiration_rate(rec.ecg, rec.fs, method=method, window=window)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Failed", str(exc))
            return

        # Per-window estimate vs ground truth -> MAE for this recording.
        est, ref = [], []
        for w in res.windows:
            r = window_ref_rate(rec, w.start, w.end)
            if np.isfinite(r):
                est.append(w.rate)
                ref.append(r)
        est, ref = np.asarray(est), np.asarray(ref)
        mae = float(np.mean(np.abs(est - ref))) if est.size else float("nan")
        ref_overall = rec.reference_rate if rec.reference_rate else float("nan")

        self.metrics_label.setText(
            f"{name} | fs {rec.fs:.0f} Hz | R-peaks {res.r_peaks.size} | "
            f"RR estimated: {res.rate:.2f} bpm   "
            f"RR reference: {ref_overall:.2f} bpm   "
            f"MAE: {mae:.2f} bpm ({est.size} window{'s' if est.size != 1 else ''})"
        )
        self._last_plot = (rec, res, name, method, mae, ref_overall)
        self._plot(rec, res, name, method, mae, ref_overall)

    def _plot(self, rec, res, name, method, mae, ref_overall):
        time = rec.time
        source = self.source_combo.currentText()
        ecg = np.asarray(rec.ecg, dtype=float)
        if source == "Filtered":
            # Same band-pass the detector/EDR use, so the R-peak markers (which
            # are located on the filtered signal) sit exactly on the peaks.
            ecg = bandpass_filter(ecg, EDR_LOWCUT, EDR_HIGHCUT, rec.fs)

        self.figure.clear()
        ax_ecg = self.figure.add_subplot(3, 1, 1)
        ax_edr = self.figure.add_subplot(3, 1, 2, sharex=ax_ecg)
        ax_psd = self.figure.add_subplot(3, 1, 3)   # frequency domain (own x-axis)

        # --- ECG with R-peaks ---
        ax_ecg.plot(time, ecg, lw=0.7, color="tab:blue", label=f"ECG ({source})")
        if res.r_peaks.size:
            ax_ecg.scatter(time[res.r_peaks], ecg[res.r_peaks], s=16, c="red",
                           marker="x", zorder=3, label="R-peaks")
        ax_ecg.set_title(f"{name} — ECG ({source}) with detected R-peaks")
        ax_ecg.set_ylabel("ECG amplitude")
        ax_ecg.grid(alpha=0.3)
        ax_ecg.legend(loc="upper right")

        # --- EDR with detected breaths ---
        ax_edr.plot(res.edr_time, res.edr, lw=1.0, color="tab:green",
                    label="EDR (respiration from ECG)")
        bpt = breath_peak_times(res)
        if bpt.size:
            yi = np.interp(bpt, res.edr_time, res.edr)
            ax_edr.scatter(bpt, yi, s=28, c="tab:red", marker="x", zorder=3,
                           label=f"detected breaths ({bpt.size})")
        ref_t = ref_breath_times(rec)
        if ref_t is not None:
            ref_t = ref_t[(ref_t >= res.edr_time[0]) & (ref_t <= res.edr_time[-1])]
            for j, t in enumerate(ref_t):
                ax_edr.axvline(t, color="tab:blue", ls=":", alpha=0.35,
                               label="annotated breaths" if j == 0 else None)
        ax_edr.set_title(
            f"EDR — RR estimated = {res.rate:.1f} bpm  |  "
            f"RR reference = {ref_overall:.1f} bpm  |  MAE = {mae:.2f} bpm  "
            f"(method = {method})"
        )
        ax_edr.set_xlabel("Time (s)")
        ax_edr.set_ylabel("EDR (z-score)")
        ax_edr.grid(alpha=0.3)
        ax_edr.legend(loc="upper right", fontsize=8)

        # --- EDR spectrum (this is what the PSD method reads the rate from) ---
        edr0 = res.edr - res.edr.mean()
        win_s = res.windows[0].duration if res.windows else (res.edr_time[-1] - res.edr_time[0])
        nperseg = int(min(max(8, round(win_s * res.fs)), edr0.size))
        f, pxx = welch(edr0, fs=res.fs, nperseg=nperseg)
        bpm = f * 60.0
        ax_psd.plot(bpm, pxx, color="tab:purple", lw=1.2, label="EDR power spectrum")
        lo, hi = RESP_BAND[0] * 60.0, RESP_BAND[1] * 60.0
        ax_psd.axvspan(lo, hi, color="0.6", alpha=0.15,
                       label=f"respiration band ({lo:.0f}–{hi:.0f} bpm)")
        band = (f >= RESP_BAND[0]) & (f <= RESP_BAND[1])
        if band.any():
            f_peak = bpm[band][int(np.argmax(pxx[band]))]
            ax_psd.scatter([f_peak], [pxx[band].max()], s=40, c="tab:purple",
                           zorder=4, label=f"spectral peak = {f_peak:.1f} bpm")
        ax_psd.axvline(res.rate, color="tab:green", ls="--", lw=1.3,
                       label=f"RR estimated = {res.rate:.1f}")
        if np.isfinite(ref_overall):
            ax_psd.axvline(ref_overall, color="tab:blue", ls=":", lw=1.6,
                           label=f"RR reference = {ref_overall:.1f}")
        ax_psd.set_xlim(0, max(60.0, hi * 1.4))
        ax_psd.set_xlabel("Frequency (breaths/min)")
        ax_psd.set_ylabel("Power")
        ax_psd.set_title("EDR spectrum (Welch PSD) — dominant frequency in band = respiratory rate")
        ax_psd.grid(alpha=0.3)
        ax_psd.legend(loc="upper right", fontsize=8)

        self._axes = (ax_ecg, ax_edr)
        self._tmax = float(time[-1])
        self._apply_view_window()
        self.canvas.draw_idle()

    # ----------------------------------------------------------------- #
    def _apply_view_window(self):
        if not self._axes:
            return
        start = max(0.0, self.view_start_spin.value())
        end = min(start + self.view_width_spin.value(), self._tmax)
        if start >= end:                       # out of range -> whole record
            start, end = 0.0, self._tmax
        self._axes[0].set_xlim(start, end)     # shared x updates both subplots

    def update_view(self):
        if not self._axes:
            return
        self._apply_view_window()
        self.canvas.draw_idle()

    def update_source(self):
        """Redraw the current record when the Raw/Filtered choice changes."""
        if self._last_plot is not None:
            self._plot(*self._last_plot)

    # ----------------------------------------------------------------- #
    def evaluate_all(self):
        """Run over every record in the folder and report the dataset-level MAE."""
        if not self.records:
            QtWidgets.QMessageBox.warning(self, "No records", "No records found.")
            return
        method = self.method_combo.currentText()
        window = self.window_spin.value()
        self._axes = None   # batch plot replaces axes; disable view cropping

        progress = QtWidgets.QProgressDialog(
            "Estimating RR for all records...", "Cancel", 0, len(self.records), self)
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setWindowTitle("Evaluate all")
        progress.show()

        names, ests, refs = [], [], []
        for k, (kind, path, name) in enumerate(self.records):
            if progress.wasCanceled():
                break
            progress.setValue(k)
            try:
                rec = load_record(kind, path)
                res = respiration_rate(rec.ecg, rec.fs, method=method, window=window)
                ref = rec.reference_rate
                if not (ref and np.isfinite(ref)):
                    continue
                names.append(short_label(name))
                ests.append(res.rate)
                refs.append(ref)
            except Exception:
                continue
        progress.close()

        ests, refs = np.asarray(ests), np.asarray(refs)
        if ests.size == 0:
            QtWidgets.QMessageBox.warning(self, "No results", "No valid records.")
            return

        err = ests - refs
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        bias = float(np.mean(err))
        corr = (float(np.corrcoef(ests, refs)[0, 1])
                if ests.size > 1 and ests.std() and refs.std() else float("nan"))
        self.metrics_label.setText(
            f"Dataset [{method}] | records {ests.size} | "
            f"MAE {mae:.2f} | RMSE {rmse:.2f} | bias {bias:+.2f} | r {corr:.3f} bpm"
        )

        # --- Plot: scatter (est vs ref) + per-record abs error ---
        self.figure.clear()
        ax1 = self.figure.add_subplot(2, 1, 1)
        ax2 = self.figure.add_subplot(2, 1, 2)

        ax1.scatter(refs, ests, s=22, alpha=0.75, color="tab:green")
        lo, hi = float(min(refs.min(), ests.min())), float(max(refs.max(), ests.max()))
        ax1.plot([lo, hi], [lo, hi], "--", color="tab:red", lw=1, label="y = x")
        ax1.set_xlabel("Reference RR (bpm)")
        ax1.set_ylabel("Estimated RR (bpm)")
        ax1.set_title(f"Estimated vs reference RR — all records ({method})")
        ax1.legend(loc="best")
        ax1.grid(alpha=0.3)

        idx = np.arange(ests.size)
        ax2.bar(idx, np.abs(err), color="tab:orange", alpha=0.85)
        ax2.axhline(mae, ls="--", color="tab:red", label=f"MAE = {mae:.2f} bpm")
        ax2.set_xticks(idx)
        ax2.set_xticklabels(names, rotation=90, fontsize=7)
        ax2.set_ylabel("Abs error (bpm)")
        ax2.set_title("Per-record absolute RR error")
        ax2.legend(loc="best")
        ax2.grid(alpha=0.3, axis="y")

        self.canvas.draw_idle()


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = RRGui()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
