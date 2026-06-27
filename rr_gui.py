"""
GUI for ECG-derived respiratory-rate estimation.

Load an ECG recording from a dataset, then view:
  * the ECG with detected R-peaks, and
  * the ECG-derived respiration (EDR) signal with the detected breaths,
    the estimated respiratory rate, and the error (MAE / MAPE) against ground truth.

Supports three bundled dataset layouts:
  * BIDMC          -> ``*_Signals.csv`` (+ ``*_Breaths.csv`` ground truth)
  * dataset_primer -> ``*_ecg.csv``     (+ ``*_metadata.json`` ground truth)
  * simultaneous   -> WFDB ``*.hea``    (five-devices set; Hexoskin breathing
                      rate is the reference, with the four task phases shaded
                      and a selectable ECG lead). Point the folder at the
                      ``generated_data`` directory (or the dataset root).

Run:
    python rr_gui.py
"""
import json
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

# ECG Signal Quality Assessment lives in the sibling ECG project. Make it
# importable so bad-quality recordings can be auto-excluded during curation.
# Kept optional: if the ECG project (or its deps) is unavailable the GUI still
# runs, only the SQA auto-exclude button is disabled.
sys.path.append(str(Path(__file__).resolve().parent.parent / "ECG"))
try:
    from ecg_sqa import ECGSQAEngine
    _SQA_IMPORT_ERROR = ""
except Exception as _exc:                       # optional dependency
    ECGSQAEngine = None
    _SQA_IMPORT_ERROR = str(_exc)

DEFAULT_DATASET = ds.BIDMC_DIR


# --------------------------------------------------------------------------- #
# Data helpers (bridge the two dataset layouts to one interface)
# --------------------------------------------------------------------------- #
def discover_records(folder):
    """Return ``[(kind, path, name), ...]`` for every record in ``folder``.

    Finds the two CSV layouts plus WFDB ``*.hea`` records (the five-devices
    set); if a ``generated_data`` subfolder exists it is searched too, so the
    dataset root can be selected directly.
    """
    folder = Path(folder)
    recs = []
    for f in sorted(folder.glob("*_Signals.csv")):
        recs.append(("bidmc", f, f.name[: -len("_Signals.csv")]))
    for f in sorted(folder.glob("*_ecg.csv")):
        recs.append(("primer", f, f.name[: -len("_ecg.csv")]))
    hea_dirs = [folder]
    sub = folder / "generated_data"
    if sub.is_dir():
        hea_dirs.append(sub)
    for d in hea_dirs:
        for f in sorted(d.glob("*.hea")):
            recs.append(("simultaneous", f, f.stem))
    return recs


def load_record(kind, path, lead=None):
    if kind == "bidmc":
        return ds.load_bidmc_record(str(path))
    if kind == "primer":
        return ds.load_primer_record(str(path))
    return ds.load_simultaneous_record(str(path), lead=lead or ds.DEFAULT_SIM_LEAD)


def short_label(name):
    """Trim the ``data_`` prefix and ``_YYYYMMDD_HHMMSS`` timestamp (primer)."""
    s = re.sub(r"_\d{8}_\d{6}$", "", name)
    return s[len("data_"):] if s.startswith("data_") else s


def window_ref_rate(rec, start, end):
    """Ground-truth respiratory rate (bpm) for a window, or NaN."""
    if isinstance(rec, ds.BidmcRecord):
        dur = end - start
        return rec.breaths_in(start, end) * 60.0 / dur if dur > 0 else np.nan
    if isinstance(rec, ds.SimultaneousRecord):
        return rec.ref_rate_in(start, end)       # mean Hexoskin BR in the window
    return rec.reference_rate if rec.reference_rate else np.nan  # primer: constant


# Task-phase shading for the five-devices dataset.
_PHASE_COLORS = {"Rest": "tab:blue", "Walking": "tab:green",
                 "2-Back": "tab:orange", "Running": "tab:red"}


def shade_phases(ax, phases, *, annotate=False):
    """Shade task-phase spans on a time-axis plot (no-op when ``phases`` empty)."""
    for label, s, e in phases or []:
        ax.axvspan(s, e, color=_PHASE_COLORS.get(label, "0.5"), alpha=0.06, zorder=0)
        if annotate:
            ax.text(0.5 * (s + e), 0.99, label, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=8, color="0.3",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                              alpha=0.5))


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
        # Curation state (persisted per folder in .rr_excluded.json):
        self.overrides = {}        # name -> bool manual decision (True=exclude, keep=False)
        self.sqa_bad = set()       # names ECG SQA judged "unacceptable" (cached)
        self.sqa_assessed = False  # SQA has been run for the current folder
        self.auto_sqa = ECGSQAEngine is not None   # auto-exclude bad ECG via SQA
        self._busy = False         # reentrancy guard (SQA's processEvents pumps the loop)
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
        self.record_combo.currentIndexChanged.connect(self._sync_exclude_check)
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

        # ECG lead (multi-lead WFDB datasets) + respiration-band upper edge.
        # Both feed the estimate, so they take effect on the next Estimate/Evaluate.
        controls.addWidget(QtWidgets.QLabel("ECG lead"), 5, 0)
        self.lead_combo = QtWidgets.QComboBox()
        self.lead_combo.addItems(ds.SIMULTANEOUS_LEADS)
        self.lead_combo.setToolTip(
            "ECG channel to analyse for multi-lead WFDB records (five-devices "
            "set). Ignored for single-lead CSV datasets.")
        controls.addWidget(self.lead_combo, 5, 1)
        controls.addWidget(QtWidgets.QLabel("Resp band hi (bpm)"), 5, 2)
        self.resp_hi_spin = QtWidgets.QDoubleSpinBox()
        self.resp_hi_spin.setRange(20.0, 60.0)
        self.resp_hi_spin.setSingleStep(1.0)
        self.resp_hi_spin.setValue(RESP_BAND[1] * 60.0)   # library default (30)
        self.resp_hi_spin.setToolTip(
            "Upper edge of the PSD respiration band. Library default is 30; "
            "widen to ~45 for exercise data so fast breathing is not clipped.")
        controls.addWidget(self.resp_hi_spin, 5, 3)

        # Outlier rejection for "Evaluate all": drop records whose abs error
        # vs the reference exceeds the threshold from the aggregate stats/plot.
        self.outlier_check = QtWidgets.QCheckBox("Exclude if error >")
        self.outlier_check.setToolTip(
            "In 'Evaluate all', drop records whose absolute RR error exceeds "
            "this many bpm from the aggregate MAE/MAPE/RMSE/plot.")
        controls.addWidget(self.outlier_check, 5, 4)
        self.outlier_spin = QtWidgets.QDoubleSpinBox()
        self.outlier_spin.setRange(0.1, 100.0)
        self.outlier_spin.setSingleStep(0.5)
        self.outlier_spin.setValue(3.0)
        self.outlier_spin.setSuffix(" bpm")
        controls.addWidget(self.outlier_spin, 5, 5)

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

        # --- Curation ----------------------------------------------------- #
        # Automatic: ECG Signal Quality Assessment flags unacceptable recordings
        # and excludes them on load; the manual toggle below overrides per record.
        self.auto_sqa_check = QtWidgets.QCheckBox("Auto-exclude bad ECG (SQA)")
        self.auto_sqa_check.toggled.connect(self.toggle_auto_sqa)
        if ECGSQAEngine is None:
            self.auto_sqa_check.setEnabled(False)
            self.auto_sqa_check.setToolTip(f"ECG SQA engine unavailable: {_SQA_IMPORT_ERROR}")
        else:
            self.auto_sqa_check.setToolTip(
                "Automatically assess every record's ECG quality and exclude the "
                "unacceptable ones. Runs once per folder; manual edits override it.")
        controls.addWidget(self.auto_sqa_check, 4, 0, 1, 3)

        # Manual: include/exclude the current record, overriding the SQA verdict.
        self.exclude_check = QtWidgets.QCheckBox("Exclude this record (manual)")
        self.exclude_check.setToolTip(
            "Manually exclude/include the selected record. Overrides the SQA "
            "verdict and is remembered in the dataset folder.")
        self.exclude_check.toggled.connect(self.toggle_excluded)
        controls.addWidget(self.exclude_check, 2, 5, 1, 2)
        self.exclude_info = QtWidgets.QLabel("Excluded: 0 / 0")
        controls.addWidget(self.exclude_info, 3, 5, 1, 2)

        # Re-assess the whole folder now / discard manual edits.
        self.sqa_btn = QtWidgets.QPushButton("Re-run SQA")
        self.sqa_btn.clicked.connect(self.rerun_sqa)
        if ECGSQAEngine is None:
            self.sqa_btn.setEnabled(False)
            self.sqa_btn.setToolTip(f"ECG SQA engine unavailable: {_SQA_IMPORT_ERROR}")
        else:
            self.sqa_btn.setToolTip("Re-assess every record's ECG quality now.")
        controls.addWidget(self.sqa_btn, 4, 4, 1, 1)
        self.clear_excl_btn = QtWidgets.QPushButton("Reset manual edits")
        self.clear_excl_btn.setToolTip(
            "Discard manual overrides and follow the SQA verdicts only.")
        self.clear_excl_btn.clicked.connect(self.reset_overrides)
        controls.addWidget(self.clear_excl_btn, 4, 5, 1, 2)

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

        # Defer the first load so the window paints before the auto-SQA pass runs.
        QtCore.QTimer.singleShot(0, self.refresh_records)

    # ----------------------------------------------------------------- #
    def browse_dataset(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select dataset folder", str(self.dataset_dir))
        if path:
            self.dataset_edit.setText(path)
            self.refresh_records()

    def refresh_records(self):
        if self._busy:                          # ignore reentrant calls (processEvents)
            return
        self.dataset_dir = Path(self.dataset_edit.text().strip())
        self._load_excluded()
        self._sync_auto_sqa_check()
        self.record_combo.clear()
        if not self.dataset_dir.exists():
            self.metrics_label.setText(f"Folder not found: {self.dataset_dir}")
            self.records = []
            self._update_exclude_status()
            return
        self.records = discover_records(self.dataset_dir)
        # Automatic curation: assess ECG quality once per folder (cached after).
        if (self.auto_sqa and not self.sqa_assessed
                and ECGSQAEngine is not None and self.records):
            self._run_sqa()
        self.record_combo.addItems([self._combo_label(name) for _, _, name in self.records])
        if self.records:
            self.metrics_label.setText(
                f"{len(self.records)} records loaded. Pick one and Estimate && Plot.")
        else:
            self.metrics_label.setText(
                f"No '*_Signals.csv' or '*_ecg.csv' files in {self.dataset_dir}")
        self._update_nav_buttons()
        self._sync_exclude_check()
        self._update_exclude_status()

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
    # Curation: exclude bad-data records (persisted per dataset folder)
    # ----------------------------------------------------------------- #
    def _excluded_path(self):
        """JSON file (in the dataset folder) that remembers the exclusions."""
        return self.dataset_dir / ".rr_excluded.json"

    def _load_excluded(self):
        """Load curation state for the current folder (resets to defaults first)."""
        self.overrides = {}
        self.sqa_bad = set()
        self.sqa_assessed = False
        self.auto_sqa = ECGSQAEngine is not None
        path = self._excluded_path()
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
        except (ValueError, OSError):
            return                              # corrupt/unreadable -> defaults
        if isinstance(data, list):
            # legacy format: a plain list of manually excluded record names
            self.overrides = {name: True for name in data}
        elif isinstance(data, dict):
            self.overrides = {k: bool(v) for k, v in data.get("overrides", {}).items()}
            self.sqa_bad = set(data.get("sqa_bad", []))
            self.sqa_assessed = bool(data.get("sqa_assessed", False))
            if ECGSQAEngine is not None:
                self.auto_sqa = bool(data.get("auto_sqa", True))

    def _save_excluded(self):
        data = {
            "auto_sqa": self.auto_sqa,
            "sqa_assessed": self.sqa_assessed,
            "sqa_bad": sorted(self.sqa_bad),
            "overrides": dict(sorted(self.overrides.items())),
        }
        try:
            self._excluded_path().write_text(json.dumps(data, indent=2))
        except OSError as exc:
            self.metrics_label.setText(f"Could not save exclusions: {exc}")

    def _is_excluded(self, name):
        """Effective verdict: manual override if present, else the SQA flag."""
        if name in self.overrides:
            return self.overrides[name]
        return self.auto_sqa and name in self.sqa_bad

    def _combo_label(self, name):
        """Display label for a record, flagged with ✖ when excluded."""
        label = short_label(name)
        return f"✖ {label}" if self._is_excluded(name) else label

    def _current_name(self):
        i = self.record_combo.currentIndex()
        return self.records[i][2] if 0 <= i < len(self.records) else None

    def toggle_excluded(self, checked):
        """Record a manual decision for the current record (overrides SQA)."""
        i = self.record_combo.currentIndex()
        name = self._current_name()
        if name is None:
            return
        self.overrides[name] = bool(checked)
        self.record_combo.setItemText(i, self._combo_label(name))
        self._save_excluded()
        self._update_exclude_status()

    def _sync_exclude_check(self):
        """Reflect the current record's exclusion state in the checkbox."""
        name = self._current_name()
        self.exclude_check.blockSignals(True)   # avoid re-triggering toggle_excluded
        self.exclude_check.setEnabled(name is not None)
        self.exclude_check.setChecked(name is not None and self._is_excluded(name))
        self.exclude_check.blockSignals(False)

    def _sync_auto_sqa_check(self):
        """Reflect the loaded auto-SQA flag in its checkbox without re-triggering."""
        self.auto_sqa_check.blockSignals(True)
        self.auto_sqa_check.setChecked(self.auto_sqa)
        self.auto_sqa_check.blockSignals(False)

    def _update_exclude_status(self):
        names = [name for _, _, name in self.records]
        excl = [n for n in names if self._is_excluded(n)]
        manual = sum(1 for n in excl if self.overrides.get(n) is True)
        sqa = len(excl) - manual
        self.exclude_info.setText(
            f"Excluded: {len(excl)} / {len(names)}  (SQA {sqa}, manual {manual})")

    def _refresh_combo_labels(self):
        """Repaint every combo entry's ✖ flag from the current exclusion state."""
        for i, (_, _, name) in enumerate(self.records):
            self.record_combo.setItemText(i, self._combo_label(name))

    def _run_sqa(self):
        """Assess every record's ECG quality; cache the unacceptable ones.

        Returns the number of records that failed to load/assess. Updates
        ``self.sqa_bad`` and marks the folder assessed, then persists.
        """
        if self._busy:                          # already assessing -> no reentry
            return 0
        self._busy = True
        try:
            progress = QtWidgets.QProgressDialog(
                "Assessing ECG signal quality...", "Cancel", 0, len(self.records), self)
            progress.setWindowModality(QtCore.Qt.WindowModal)
            progress.setWindowTitle("ECG SQA")
            progress.show()

            bad, n_failed = set(), 0
            for k, (kind, path, name) in enumerate(self.records):
                if progress.wasCanceled():
                    break
                progress.setValue(k)
                QtWidgets.QApplication.processEvents()
                try:
                    rec = load_record(kind, path, self._selected_lead())
                    res = ECGSQAEngine(rec.ecg, rec.fs).assess()
                except Exception:
                    n_failed += 1
                    continue
                if res["quality"] == "unacceptable":
                    bad.add(name)
            progress.close()
        finally:
            self._busy = False

        self.sqa_bad = bad
        self.sqa_assessed = True
        self._save_excluded()
        return n_failed

    def rerun_sqa(self):
        """Force a fresh SQA pass over the folder (manual overrides preserved)."""
        if ECGSQAEngine is None:
            QtWidgets.QMessageBox.warning(
                self, "SQA unavailable",
                f"Could not import the ECG SQA engine:\n{_SQA_IMPORT_ERROR}")
            return
        if not self.records:
            QtWidgets.QMessageBox.warning(self, "No records", "No records found.")
            return
        n_failed = self._run_sqa()
        self._refresh_combo_labels()
        self._sync_exclude_check()
        self._update_exclude_status()
        self.metrics_label.setText(
            f"SQA: assessed {len(self.records)} record(s) | "
            f"{len(self.sqa_bad)} unacceptable (auto-excluded)"
            f"{f' | {n_failed} failed' if n_failed else ''}. Manual overrides kept."
        )

    def toggle_auto_sqa(self, checked):
        """Enable/disable automatic SQA exclusion (runs the pass on first enable)."""
        self.auto_sqa = bool(checked)
        if checked and not self.sqa_assessed and ECGSQAEngine is not None and self.records:
            self._run_sqa()
        self._save_excluded()
        self._refresh_combo_labels()
        self._sync_exclude_check()
        self._update_exclude_status()

    def reset_overrides(self):
        """Discard all manual edits so the curation follows the SQA verdicts."""
        if not self.overrides:
            QtWidgets.QMessageBox.information(
                self, "No manual edits", "There are no manual curation edits to reset.")
            return
        if QtWidgets.QMessageBox.question(
                self, "Reset manual edits",
                f"Discard {len(self.overrides)} manual override(s) and follow the "
                f"SQA verdicts only?"
        ) != QtWidgets.QMessageBox.Yes:
            return
        self.overrides = {}
        self._save_excluded()
        self._refresh_combo_labels()
        self._sync_exclude_check()
        self._update_exclude_status()

    # ----------------------------------------------------------------- #
    def _selected_lead(self):
        """ECG lead for WFDB records (ignored by the single-lead CSV loaders)."""
        return self.lead_combo.currentText() or ds.DEFAULT_SIM_LEAD

    def _resp_band(self):
        """PSD respiration band (Hz) from the fixed low edge and the hi spinbox."""
        return (RESP_BAND[0], self.resp_hi_spin.value() / 60.0)

    def estimate_and_plot(self):
        i = self.record_combo.currentIndex()
        if i < 0 or i >= len(self.records):
            QtWidgets.QMessageBox.warning(self, "No record", "No record selected.")
            return

        kind, path, name = self.records[i]
        method = self.method_combo.currentText()
        window = self.window_spin.value()
        try:
            rec = load_record(kind, path, self._selected_lead())
            res = respiration_rate(rec.ecg, rec.fs, method=method, window=window,
                                   resp_band=self._resp_band())
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
        nz = ref != 0                            # MAPE undefined where ref == 0
        mape = (float(np.mean(np.abs((est[nz] - ref[nz]) / ref[nz]))) * 100.0
                if nz.any() else float("nan"))
        ref_overall = rec.reference_rate if rec.reference_rate else float("nan")

        excluded_tag = "  ⚠ EXCLUDED (bad data)" if self._is_excluded(name) else ""
        lead_tag = f" ({rec.lead})" if isinstance(rec, ds.SimultaneousRecord) else ""
        self.metrics_label.setText(
            f"{name}{lead_tag} | fs {rec.fs:.0f} Hz | R-peaks {res.r_peaks.size} | "
            f"RR estimated: {res.rate:.2f} bpm   "
            f"RR reference: {ref_overall:.2f} bpm   "
            f"MAE: {mae:.2f} bpm   "
            f"MAPE: {mape:.1f}% ({est.size} window{'s' if est.size != 1 else ''})"
            f"{excluded_tag}"
        )
        self._last_plot = (rec, res, name, method, mae, mape, ref_overall)
        self._plot(rec, res, name, method, mae, mape, ref_overall)

    def _plot(self, rec, res, name, method, mae, mape, ref_overall):
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

        # --- task-phase shading (five-devices dataset; no-op otherwise) ---
        phases = getattr(rec, "phases", None)
        shade_phases(ax_ecg, phases, annotate=True)
        shade_phases(ax_edr, phases)

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
            f"RR reference = {ref_overall:.1f} bpm  |  MAE = {mae:.2f} bpm  |  "
            f"MAPE = {mape:.1f}%  (method = {method})"
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
        band_hz = self._resp_band()
        lo, hi = band_hz[0] * 60.0, band_hz[1] * 60.0
        ax_psd.axvspan(lo, hi, color="0.6", alpha=0.15,
                       label=f"respiration band ({lo:.0f}–{hi:.0f} bpm)")
        band = (f >= band_hz[0]) & (f <= band_hz[1])
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
        n_excluded = 0
        for k, (kind, path, name) in enumerate(self.records):
            if progress.wasCanceled():
                break
            progress.setValue(k)
            if self._is_excluded(name):      # curated out (SQA or manual)
                n_excluded += 1
                continue
            try:
                rec = load_record(kind, path, self._selected_lead())
                res = respiration_rate(rec.ecg, rec.fs, method=method, window=window,
                                       resp_band=self._resp_band())
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

        n_outliers = 0
        if self.outlier_check.isChecked():
            threshold = self.outlier_spin.value()
            keep = np.abs(ests - refs) <= threshold
            n_outliers = int((~keep).sum())
            if keep.any():
                names = [n for n, k in zip(names, keep) if k]
                ests, refs = ests[keep], refs[keep]
            else:
                QtWidgets.QMessageBox.warning(
                    self, "No results",
                    f"All records exceed the {threshold:.1f} bpm error threshold.")
                return

        err = ests - refs
        mae = float(np.mean(np.abs(err)))
        nz = refs != 0                           # MAPE undefined where ref == 0
        mape = (float(np.mean(np.abs(err[nz] / refs[nz]))) * 100.0
                if nz.any() else float("nan"))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        bias = float(np.mean(err))
        corr = (float(np.corrcoef(ests, refs)[0, 1])
                if ests.size > 1 and ests.std() and refs.std() else float("nan"))
        # R2 score (coefficient of determination): 1 - SS_res/SS_tot, with the
        # reference as truth. Negative when the estimator beats predicting the
        # mean. Differs from r**2 (which ignores bias/scale).
        ss_tot = float(np.sum((refs - refs.mean()) ** 2))
        r2 = float(1.0 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else float("nan")
        outlier_tag = f" (outliers {n_outliers})" if n_outliers else ""
        self.metrics_label.setText(
            f"Dataset [{method}] | records {ests.size} (excluded {n_excluded}){outlier_tag} | "
            f"MAE {mae:.2f} | MAPE {mape:.1f}% | RMSE {rmse:.2f} | "
            f"bias {bias:+.2f} | r {corr:.3f} | R² {r2:.3f}"
        )

        self._save_label_distribution(refs)

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

    def _save_label_distribution(self, refs):
        """Save a histogram of the reference RR (label) values actually used
        in the last 'Evaluate all' run to a PNG in the dataset folder.
        Built off-screen (not shown on the GUI canvas); overwritten each run."""
        fig = Figure(figsize=(8, 5), tight_layout=True)
        ax = fig.add_subplot(1, 1, 1)
        n_bins = min(15, max(5, refs.size // 2))
        ax.hist(refs, bins=n_bins, color="tab:blue", alpha=0.7, label="reference RR (label)")
        ax.axvline(float(refs.mean()), color="tab:blue", ls="--", lw=1.3,
                   label=f"mean = {refs.mean():.1f} bpm")
        ax.axvline(float(np.median(refs)), color="tab:cyan", ls=":", lw=1.3,
                   label=f"median = {np.median(refs):.1f} bpm")
        ax.set_xlabel("Reference RR (bpm)")
        ax.set_ylabel("Count")
        ax.set_title(f"Reference RR Distribution — Total Subject ({refs.size})")
        ax.legend(loc="best")
        ax.grid(alpha=0.3, axis="y")

        path = self.dataset_dir / "rr_label_distribution.png"
        try:
            fig.savefig(path, dpi=150)
        except OSError as exc:
            self.metrics_label.setText(f"{self.metrics_label.text()}  [save failed: {exc}]")


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = RRGui()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
