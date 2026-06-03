# Respiratory-Rate (RR) Library

Estimate breathing rate (breaths per minute) from a single-lead ECG. It works
by tracking how the R-peak height rises and falls with each breath — no
separate breathing sensor needed. R-peak detection comes from the sibling
[`ECG`](../ECG) library.

## How to use

```python
from respiration_rate import respiration_rate

res = respiration_rate(ecg, fs)        # ecg = samples, fs = sampling rate in Hz

print(res.rate)                        # breathing rate in breaths/min
```

That's it. `ecg` is a 1-D array of ECG samples and `fs` is its sampling rate.
By default it uses the PSD method with a 32-second window.

### A bit more from the result

```python
res.rate            # overall breathing rate (breaths/min)
res.window_rates    # rate for each 60-second window
res.edr             # the breathing signal pulled out of the ECG
res.r_peaks         # heartbeat (R-peak) positions used
```

### Common options

```python
respiration_rate(
    ecg, fs,
    method="psd",     # "psd" (default, recommended) or "peak" (notebook method)
    window=32,        # window length in seconds
)
```

## Try it on the sample datasets

```bash
python evaluate.py --method psd          # run on BIDMC + dataset_primer_1
python evaluate.py --dataset bidmc       # just one dataset
```

This prints the accuracy and saves per-recording numbers to `results/`.

## GUI

```bash
python rr_gui.py
```

Pick a dataset folder and a record to see three stacked plots: the ECG (with
detected R-peaks), the EDR breathing signal (with detected breaths and the
estimated rate / MAE against the ground truth), and the EDR's Welch power
spectrum — where the dominant frequency inside the shaded respiration band is
the estimated rate. "Evaluate all" scores the whole folder. The "ECG source"
dropdown switches the ECG plot between the raw signal and the 5–40 Hz
band-passed signal used for detection (display only — it never changes the
result). Works with both the BIDMC (`*_Signals.csv`) and dataset_primer_1
(`*_ecg.csv`) layouts.

## How well it works

On the **BIDMC** dataset (real recordings with a true breathing reference) the
PSD method is accurate to about **1.9 breaths/min** on average.

On **dataset_primer_1** the estimate does not match the reference value — that
dataset's breathing-rate label does not line up with its ECG, so it is not a
fair test of the method (the heart-rate it recovers from the same ECG is
correct).

## Files

| File | What it is |
|------|------------|
| `respiration_rate.py` | The library |
| `datasets.py` | Loads the two sample datasets |
| `evaluate.py` | Runs and scores the library on a dataset |
| `rr_gui.py` | GUI to load a recording and plot ECG + EDR with RR and MAE |
