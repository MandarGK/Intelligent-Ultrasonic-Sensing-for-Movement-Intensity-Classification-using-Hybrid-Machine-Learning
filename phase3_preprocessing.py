"""
Phase 3 — Preprocessing (single-file consolidated version).

Everything that was previously split across core/manifest.py, io.py,
robust_onset.py, alignment.py, signal_features.py, spectrogram.py,
splitting.py, scaling.py, and datasets.py now lives in this one file,
so the whole of Phase 3 runs from a single command. Logic is unchanged
from the original modules except for anonymized P1-P5 subject naming and automatic backup of an existing Phase 3 output.

Usage
-----
    python phase3_preprocessing.py --data-dir /path/to/Data \
        [--out-dir artifacts/phase3] [--chunksize 500] [--val-fraction 0.2]

Requires: numpy, pandas, scipy. torch is only needed if you import the
Dataset classes at the bottom of this file from elsewhere (e.g. Phase 5) --
it is NOT required just to run preprocessing, so the import is optional.

Output (under --out-dir):
    binary/{stem}_full.npy         one per input CSV: (n_rows, 25000) float32 memmap
    binary/{stem}_aligned.npy      one per input CSV: (n_rows, 4900) float32 memmap
                                    (rows with failed alignment are all-NaN)
    master_manifest.csv            one row per recording: metadata + features +
                                    onset/alignment info + Env1 train/val split
    bonus_pd_manifest.csv          Env2 subject PD, held out from the main pipeline
    env1_block_split_log.csv       which time-blocks went to train vs val (reproducibility)
    scaler.json                    amplitude scaler fit on Env1-train only
"""

import argparse
import json
import re
import sys
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import hilbert, stft

# =============================================================================
# SECTION 1 — manifest.py: filename -> metadata parser
# =============================================================================
#
# Expected filename convention:
#   Room1_EMPTY_5000.csv, Room1_with_Chair_5000.csv,
#   Room2_EMPTY_5000.csv, Room2_with_chair_5000.csv,
#   ENV1_HUMAN_P1_5000.csv, ENV1_HUMAN_P2_5000.csv,
#   ENV1_HUMAN_SITTING_P1_5000.csv, ENV1_HUMAN_SITTING_P2_5000.csv,
#   ENV2_HUMAN_P3_5000.csv, ENV2_HUMAN_P4_5000.csv, ENV2_HUMAN_P5_5000.csv,
#   ENV2_HUMAN_SITTING_P3_5000.csv, ENV2_HUMAN_SITTING_P4_5000.csv,
#   ENV2_HUMAN_SITTING_P5_5000.csv
#
# Subject naming is intentionally anonymized:
#   P1 = Person 1, P2 = Person 2, P3 = Person 3, P4 = Person 4, P5 = Person 5.
# P5 remains the bonus/held-out Env2 subject.
#
# Fails loudly (raises) rather than guessing on any filename that doesn't
# match -- a silent misparse is a mislabeling risk (Section 15).

MAIN_ENV2_SUBJECTS = {"P3", "P4"}
BONUS_ENV2_SUBJECTS = {"P5"}


def parse_filename(filename: str):
    stem = filename.replace(".csv", "")
    upper = stem.upper()

    if upper.startswith("ROOM1") or "ENV1" in upper:
        environment = "Env1"
    elif upper.startswith("ROOM2") or "ENV2" in upper:
        environment = "Env2"
    else:
        raise ValueError(f"Cannot determine environment from filename: {filename}")

    if "EMPTY" in upper:
        condition, class_label, subject, activity = "EmptyRoom", "NON_HUMAN", None, None
        group = None
    elif "CHAIR" in upper:
        condition, class_label, subject, activity = "ChairTable", "NON_HUMAN", None, None
        group = None
    elif "HUMAN" in upper:
        class_label = "HUMAN"
        activity = "Reading" if "SITTING" in upper else "Standing"
        condition = "HumanReading" if activity == "Reading" else "HumanStanding"
        tokens = re.split(r"[_]", stem)
        strip_tokens = {"ENV1", "ENV2", "ROOM1", "ROOM2", "HUMAN", "SITTING", "5000"}
        subject_tokens = [t.upper() for t in tokens if t.upper().startswith("P") and t[1:].isdigit()]
        if not subject_tokens:
            raise ValueError(
                f"Could not extract anonymized subject token P1-P5 from filename: {filename}"
            )

        subject = subject_tokens[0]
        if subject not in {"P1", "P2", "P3", "P4", "P5"}:
            raise ValueError(
                f"Unrecognized subject '{subject}' in {filename}. "
                "Expected one of P1, P2, P3, P4, P5."
            )

        if environment == "Env1":
            if subject not in {"P1", "P2"}:
                raise ValueError(
                    f"Env1 file {filename} must use P1 or P2, not {subject}."
                )
            group = "main"
        else:
            if subject in MAIN_ENV2_SUBJECTS:
                group = "main"
            elif subject in BONUS_ENV2_SUBJECTS:
                group = "bonus"
            else:
                raise ValueError(
                    f"Unrecognized Env2 subject '{subject}' in {filename} -- "
                    f"expected P3, P4, or P5."
                )
    else:
        raise ValueError(f"Cannot determine condition from filename: {filename}")

    return {
        "environment": environment, "condition": condition, "class_label": class_label,
        "subject": subject, "activity": activity, "subject_group": group,
    }


def build_file_manifest(csv_filenames):
    """Returns a list of dicts, one per input filename, with parsed metadata.
    Raises on the first unparseable filename (fail loudly, per Section 15)."""
    records = []
    for fname in csv_filenames:
        parsed = parse_filename(fname)
        parsed["source_file"] = fname
        records.append(parsed)
    return records


# =============================================================================
# SECTION 2 — robust_onset.py: robust first-arrival echo onset detection
# =============================================================================
#
# Validated against real data: corrected the earlier global-peak method's
# mistakes in 57.7% of sampled recordings overall (97.1% for HumanReading
# specifically), and confirmed physically consistent via a height vs onset
# calibration (R^2 = 0.961 against known subject heights).
#
# Known limitation (documented, not silently hidden): `num_events` is noisy
# on real data likely genuine room reverberation tail plus some threshold
# sensitivity and should NOT be used as a model feature without further
# tuning. Only `onset_index` / `onset_time_ms` (the FIRST event) is validated
# for use.

def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def detect_first_arrival(
    samples: np.ndarray,
    fs_hz: float,
    smooth_window: int = 25,
    k_mad: float = 8.0,
    min_duration_samples: int = 10,
):
    samples = samples.astype(np.float64)
    samples = samples - samples.mean()

    analytic = hilbert(samples)
    envelope = np.abs(analytic)
    envelope = _smooth(envelope, smooth_window)

    med = np.median(envelope)
    mad = np.median(np.abs(envelope - med))
    threshold = med + k_mad * mad

    above = envelope > threshold

    if not above.any():
        return {
            "onset_index": None, "onset_time_ms": None,
            "threshold_used": threshold, "noise_floor_median": med,
            "num_events": 0,
        }

    diffs = np.diff(above.astype(int))
    starts = np.where(diffs == 1)[0] + 1
    ends = np.where(diffs == -1)[0] + 1
    if above[0]:
        starts = np.insert(starts, 0, 0)
    if above[-1]:
        ends = np.append(ends, len(above))

    valid_runs = [(s, e) for s, e in zip(starts, ends) if (e - s) >= min_duration_samples]

    if not valid_runs:
        return {
            "onset_index": None, "onset_time_ms": None,
            "threshold_used": threshold, "noise_floor_median": med,
            "num_events": 0,
        }

    onset_index = valid_runs[0][0]
    onset_time_ms = (onset_index / fs_hz) * 1000.0

    return {
        "onset_index": onset_index, "onset_time_ms": onset_time_ms,
        "threshold_used": threshold, "noise_floor_median": med,
        "num_events": len(valid_runs),
    }


# =============================================================================
# SECTION 3 — alignment.py: position-normalized alignment (for RQ6)
# =============================================================================
#
# Why this exists: echo onset time directly encodes target height/distance
# beneath the (vertically-mounted, overhead) sensor. Standing and Reading
# differ hugely in height by design median onset ~0-1.2ms for Standing
# vs ~3.5-4.2ms for Reading (Env2, robust-corrected). A model using the
# FULL window (absolute position intact) could trivially separate these by
# distance alone, never touching the actual page-turning micro-motion
# signature the research question is about.
#
# This crops a FIXED-LENGTH window starting at each recording's own
# detected onset, removing absolute position as a usable cue -- forcing
# any model trained on this branch to learn from echo SHAPE rather than
# WHEN it arrives.
#
# Crop length: the 12.8ms (25,000-sample) window is fixed. The
# latest-onset condition (EmptyRoom, median ~9.8ms robust-corrected)
# leaves the least room after onset about 3ms / ~5,900 samples.
# ALIGNED_WINDOW_SAMPLES is set below that so every condition can supply
# a full, non truncated crop; recordings where onset arrives so late a
# full crop isn't available anyway are flagged rather than silently
# zero-padded with a false signal.

FS_HZ = 1_953_125.0
ALIGNED_WINDOW_SAMPLES = 4_900  # ~2.5ms; fits safely within every condition's post-onset budget
FULL_WINDOW_SAMPLES = 25_000


def align_recording(samples: np.ndarray, fs_hz: float = FS_HZ,
                     window_samples: int = ALIGNED_WINDOW_SAMPLES):
    """
    Returns:
      aligned_waveform : np.ndarray of length `window_samples`, or None if
                          no onset was detected or not enough samples remained
                          after onset to fill the window.
      onset_info       : dict from detect_first_arrival, plus 'truncated' flag.
    """
    onset_info = detect_first_arrival(samples, fs_hz)

    if onset_info["onset_index"] is None:
        return None, {**onset_info, "truncated": None, "reason": "no_onset_detected"}

    start = onset_info["onset_index"]
    end = start + window_samples

    if end > len(samples):
        # Not enough samples after onset -- flag rather than pad with zeros,
        # since zero-padding here would be a fabricated signal (Section 15).
        return None, {**onset_info, "truncated": True, "reason": "insufficient_samples_after_onset"}

    aligned = samples[start:end]
    return aligned, {**onset_info, "truncated": False, "reason": None}


# =============================================================================
# SECTION 4 — signal_features.py: time/frequency-domain features (classical ML)
# =============================================================================
#
# Phase 2 fix, confirmed necessary from real data: the original
# dominant_frequency_hz used an UNRESTRICTED global argmax over the full
# 0-976kHz spectrum. On real data this returned exactly 0.0 Hz for 97.4%
# of recordings the short, impulsive burst's broadband envelope
# dominates low-frequency bins, drowning out the genuine ~40kHz carrier
# peak that IS visible once the search is restricted to a physically
# justified band. Every frequency domain feature below now searches only
# within CARRIER_BAND_HZ, centered on the confirmed ~40kHz SRF02 carrier,
# per Section 11.C's requirement that band choice be justified, not guessed.
#
# One-line justification per feature (Section 9 requirement):
#   mean, std, rms, peak, peak_to_peak, crest_factor:
#     Detect presence/absence and variability of reflected energy a
#     static surface should show low variance vs. a human body (breathing,
#     micro motion) or page-turning.
#   dominant_frequency_hz, band_energy (20-60kHz):
#     Ultrasonic Doppler shifts from movement (even small page turning
#     motion) show up as sidebands around the carrier frequency, which
#     time domain stats alone won't capture.

CARRIER_BAND_HZ = (20_000.0, 60_000.0)  # justified: SRF02 carrier confirmed ~40kHz in Phase 2 EDA


def compute_time_domain_features(samples: np.ndarray) -> dict:
    x = samples.astype(np.float64)
    mean = x.mean()
    std = x.std()
    rms = np.sqrt(np.mean(x ** 2))
    peak = np.max(np.abs(x))
    peak_to_peak = x.max() - x.min()
    crest_factor = peak / rms if rms > 0 else 0.0
    return {
        "mean": mean, "std": std, "rms": rms,
        "peak": peak, "peak_to_peak": peak_to_peak, "crest_factor": crest_factor,
    }


def compute_frequency_domain_features(samples: np.ndarray, fs_hz: float = FS_HZ,
                                       band_hz=CARRIER_BAND_HZ) -> dict:
    x = samples.astype(np.float64)
    x = x - x.mean()

    n = len(x)
    spectrum = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs_hz)

    band_mask = (freqs >= band_hz[0]) & (freqs <= band_hz[1])

    if not band_mask.any():
        return {"dominant_frequency_hz": 0.0, "band_energy": 0.0, "total_energy": float(np.sum(spectrum ** 2))}

    band_spectrum = spectrum[band_mask]
    band_freqs = freqs[band_mask]

    dominant_idx = np.argmax(band_spectrum)
    dominant_frequency_hz = band_freqs[dominant_idx]
    band_energy = np.sum(band_spectrum ** 2)
    total_energy = np.sum(spectrum ** 2)

    return {
        "dominant_frequency_hz": float(dominant_frequency_hz),
        "band_energy": float(band_energy),
        "total_energy": float(total_energy),
        "band_energy_fraction": float(band_energy / total_energy) if total_energy > 0 else 0.0,
    }


def compute_all_features(samples: np.ndarray, fs_hz: float = FS_HZ) -> dict:
    feats = {}
    feats.update(compute_time_domain_features(samples))
    feats.update(compute_frequency_domain_features(samples, fs_hz))
    return feats


# =============================================================================
# SECTION 5 — spectrogram.py: STFT for the 2D CNN (band-limited)
# =============================================================================
#
# Parameter justification (Section 11.C requires this, not arbitrary values):
#   Frequency range 20-60kHz: Phase 2 EDA confirmed the SRF02 carrier sits
#     at ~40kHz with essentially all discriminative energy concentrated
#     there; the remaining ~936kHz of Nyquist bandwidth is not
#     scientifically motivated to include and would waste model capacity.
#   nperseg=256: at Fs=1,953,125 Hz this gives a frequency resolution of
#     ~7.6kHz/bin -- enough to resolve the ~40kHz carrier and nearby
#     Doppler sidebands from motion.
#   noverlap=224 (87.5%): the aligned window is short (4,900 samples /
#     2.5ms); high overlap is needed to get more than a handful of time
#     bins out of that short a signal.

SPEC_FREQ_BAND_HZ = (20_000.0, 60_000.0)
SPEC_NPERSEG = 256
SPEC_NOVERLAP = 224


def compute_band_limited_spectrogram(samples: np.ndarray, fs_hz: float = FS_HZ,
                                       band_hz=SPEC_FREQ_BAND_HZ,
                                       nperseg: int = SPEC_NPERSEG, noverlap: int = SPEC_NOVERLAP):
    x = samples.astype(np.float64)
    x = x - x.mean()

    freqs, times, Zxx = stft(x, fs=fs_hz, nperseg=nperseg, noverlap=noverlap)
    magnitude = np.abs(Zxx)

    band_mask = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    band_magnitude = magnitude[band_mask, :]
    band_freqs = freqs[band_mask]

    return band_magnitude.astype(np.float32), band_freqs, times


# =============================================================================
# SECTION 6 — io.py: chunked CSV -> memmap conversion + per-row features
# =============================================================================
#
# Phase 3 needs to run onset detection + feature extraction once per
# recording, but Phase 4/5 (training 5 models, with hyperparameter search)
# will need to read the same raw waveforms repeatedly, many times, over
# many epochs. Re-parsing 700MB text CSVs on every access is impractical.
#
# This streams each CSV exactly ONCE (chunked, so memory stays bounded
# regardless of file size), and while doing so:
#   - writes the full 25,000-sample raw waveform for every row into a
#     per-file memmap .npy array (position-aware branch input)
#   - runs onset detection + alignment, writing the aligned crop into a
#     second memmap .npy array where alignment succeeded (position-
#     normalized branch input; rows where alignment failed are marked
#     invalid in the metadata table, not silently zero-filled)
#   - computes time/frequency-domain features for the classical models
#   - returns a per-row metadata DataFrame

N_HEADER_COLS = 17


def _count_rows(csv_path: Path, chunksize: int = 500) -> int:
    total = 0
    for chunk in pd.read_csv(csv_path, header=None, chunksize=chunksize):
        total += len(chunk)
    return total


def convert_file(csv_path: Path, out_dir: Path, chunksize: int = 500, verbose: bool = True):
    """
    Streams `csv_path` once. Writes:
      out_dir/{stem}_full.npy     -- (n_rows, FULL_WINDOW_SAMPLES) float32 memmap
      out_dir/{stem}_aligned.npy  -- (n_rows, ALIGNED_WINDOW_SAMPLES) float32 memmap
                                      (rows with failed alignment are all-NaN)
    Returns a per-row metadata DataFrame (not yet written to disk -- caller
    combines across files and writes the master table).
    """
    stem = csv_path.stem
    if verbose:
        print(f"  Counting rows in {csv_path.name} ...", end=" ", flush=True)
    n_rows = _count_rows(csv_path, chunksize=chunksize)
    if verbose:
        print(f"{n_rows} rows")

    full_path = out_dir / f"{stem}_full.npy"
    aligned_path = out_dir / f"{stem}_aligned.npy"

    full_mm = np.lib.format.open_memmap(
        full_path, mode="w+", dtype=np.float32, shape=(n_rows, FULL_WINDOW_SAMPLES)
    )
    aligned_mm = np.lib.format.open_memmap(
        aligned_path, mode="w+", dtype=np.float32, shape=(n_rows, ALIGNED_WINDOW_SAMPLES)
    )

    rows_meta = []
    row_cursor = 0

    for chunk in pd.read_csv(csv_path, header=None, chunksize=chunksize):
        if chunk.shape[1] != N_HEADER_COLS + FULL_WINDOW_SAMPLES:
            raise ValueError(
                f"{csv_path.name}: unexpected column count {chunk.shape[1]} "
                f"(expected {N_HEADER_COLS + FULL_WINDOW_SAMPLES})"
            )

        sample_block = chunk.iloc[:, N_HEADER_COLS:].to_numpy(dtype=np.float32)
        header_block = chunk.iloc[:, :N_HEADER_COLS].to_numpy()

        for i in range(sample_block.shape[0]):
            samples = sample_block[i]
            full_mm[row_cursor] = samples

            aligned, onset_info = align_recording(samples, fs_hz=FS_HZ)
            if aligned is not None:
                aligned_mm[row_cursor] = aligned.astype(np.float32)
                alignment_ok = True
            else:
                aligned_mm[row_cursor] = np.nan
                alignment_ok = False

            feats = compute_all_features(samples, fs_hz=FS_HZ)

            rows_meta.append({
                "source_file": csv_path.name,
                "row_index": row_cursor,
                "sample_frequency_hz": header_block[i][6],
                "adc_resolution": header_block[i][7],
                "onset_index": onset_info.get("onset_index"),
                "onset_time_ms": onset_info.get("onset_time_ms"),
                "alignment_ok": alignment_ok,
                "alignment_reason": onset_info.get("reason"),
                **feats,
            })
            row_cursor += 1

        if verbose:
            print(f"    ... {row_cursor}/{n_rows} rows processed", end="\r", flush=True)

    full_mm.flush()
    aligned_mm.flush()
    if verbose:
        print(f"    done: {row_cursor}/{n_rows} rows -> {full_path.name}, {aligned_path.name}" + " " * 10)

    n_failed = sum(1 for r in rows_meta if not r["alignment_ok"])
    if verbose and n_failed:
        print(f"    NOTE: {n_failed}/{n_rows} rows failed alignment (see alignment_reason in metadata)")

    return pd.DataFrame(rows_meta)


# =============================================================================
# SECTION 7 — splitting.py: group-aware Env1 train/val split
# =============================================================================
#
# Why "recording_id" alone isn't a valid split unit here: in the original
# leakage design (Section 7), splitting was meant to happen at the
# recording_id (group) level so that windows chopped from one continuous
# recording never span both train and val. But in THIS dataset, each row
# already IS one independent 12.8ms ping -- there's no continuous stream
# to chop. So grouping by recording_id (= one row) degenerates to a plain
# random row-level split, which does NOT protect against the leakage that
# actually exists here: consecutive pings are only ~66ms apart (Phase 1
# finding) and are very likely correlated -- slow body sway, breathing
# phase, minor repositioning -- so a model could exploit "this looks like
# the recording two rows away in the training set" rather than learning
# anything condition-general.
#
# Fix: group consecutive rows into fixed-size BLOCKS (default 50 rows =
# ~3.3 seconds of real time at the ~66ms inter-ping rate) and split at the
# BLOCK level, not the row level. No block is split across train/val.

BLOCK_SIZE_ROWS = 50


def assign_blocks(df: pd.DataFrame, block_size: int = BLOCK_SIZE_ROWS) -> pd.DataFrame:
    """Adds a `block_id` column: {source_file}_{block_index}, grouping
    consecutive row_index values within each file into fixed-size blocks."""
    df = df.copy()
    df["block_id"] = df["source_file"] + "_" + (df["row_index"] // block_size).astype(str)
    return df


def group_aware_train_val_split(df: pd.DataFrame, val_fraction: float = 0.2,
                                 seed: int = 42, block_size: int = BLOCK_SIZE_ROWS):
    """
    Splits an Env1-only dataframe into train/val at the block level,
    stratified as best as possible by `condition` so both non-human
    sub-conditions and all human sub-conditions appear in both splits.

    Returns the input df with a new 'split' column ('train' or 'val'),
    plus the exact list of block_ids assigned to each split (for the
    reproducibility log).
    """
    assert (df["environment"] == "Env1").all(), \
        "group_aware_train_val_split must only be called on Env1 data -- Env2 is test-only, never split here."

    df = assign_blocks(df, block_size)
    rng = np.random.default_rng(seed)

    split_assignment = {}
    for condition, group in df.groupby("condition"):
        blocks = np.array(group["block_id"].unique().tolist(), dtype=object)
        rng.shuffle(blocks)
        n_val = max(1, int(len(blocks) * val_fraction))
        val_blocks = set(blocks[:n_val])
        for b in blocks:
            split_assignment[b] = "val" if b in val_blocks else "train"

    df["split"] = df["block_id"].map(split_assignment)

    # Reproducibility artifact: which blocks (not individual rows) went where
    block_log = pd.DataFrame(
        [{"block_id": k, "split": v} for k, v in split_assignment.items()]
    )

    return df, block_log


# =============================================================================
# SECTION 8 — scaling.py: amplitude scaler with enforced fit/transform boundary
# =============================================================================
#
# Section 15 leakage rule: any fitted parameter (scaler mean/std) must be
# fit ONLY on Env1 train, then applied (never refit) to Env1 val and all of
# Env2. Implemented so that calling fit() on anything other than
# Env1-train data raises an error, rather than silently allowing the
# mistake (e.g. calling fit_transform() on the whole dataset before splitting).

class Env1TrainOnlyScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self._fitted_on = None  # sanity tag, must be 'Env1_train'

    def fit(self, samples_2d: np.ndarray, source_tag: str):
        if source_tag != "Env1_train":
            raise RuntimeError(
                f"Refusing to fit scaler on '{source_tag}'. "
                "Per Section 15, the scaler must be fit ONLY on Env1 train data. "
                "Use .transform() for Env1 val and all of Env2."
            )
        self.mean_ = samples_2d.mean()
        self.std_ = samples_2d.std()
        self._fitted_on = source_tag
        return self

    def transform(self, samples_2d: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Scaler has not been fit yet.")
        return (samples_2d - self.mean_) / (self.std_ if self.std_ > 0 else 1.0)

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"mean": float(self.mean_), "std": float(self.std_),
                       "fitted_on": self._fitted_on}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        obj = cls()
        obj.mean_ = d["mean"]
        obj.std_ = d["std"]
        obj._fitted_on = d["fitted_on"]
        return obj


# =============================================================================
# SECTION 9 — datasets.py: PyTorch Dataset classes (used by Phase 4/5, not
# by this script's own main() -- kept here so this one file is the single
# source of truth for Phase 3 outputs. torch import is optional/lazy so
# running preprocessing itself never requires torch to be installed.
# =============================================================================
#
# Two branches:
#   PositionAware*Dataset       : full 25,000-sample window, absolute timing
#                                  intact. Legitimate for RQ1 (human vs
#                                  non-human), since this sensor's real
#                                  detection mechanism is range-based
#                                  (confirmed via the height calibration).
#   PositionNormalized*Dataset  : cropped to the aligned 4,900-sample window
#                                  anchored at each recording's own detected
#                                  onset. Absolute distance is not
#                                  recoverable from this input -- required
#                                  for RQ6 (Standing vs Reading) so the
#                                  model can't shortcut via posture height.
#
# Rows where alignment failed are excluded from the position-normalized
# datasets at construction time, not silently included with fabricated
# (zero-padded) data.

CLASS_TO_INT = {"NON_HUMAN": 0, "HUMAN": 1}

try:
    import torch
    from torch.utils.data import Dataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    Dataset = object  # fallback base class so class defs below don't error


if _TORCH_AVAILABLE:

    class _BaseRecordingDataset(Dataset):
        def __init__(self, metadata_df, binary_dir: Path, scaler: Env1TrainOnlyScaler, array_suffix: str):
            self.meta = metadata_df.reset_index(drop=True)
            self.binary_dir = Path(binary_dir)
            self.scaler = scaler
            self.array_suffix = array_suffix
            self._memmap_cache = {}

        def __len__(self):
            return len(self.meta)

        def _get_memmap(self, source_file: str):
            if source_file not in self._memmap_cache:
                stem = Path(source_file).stem
                path = self.binary_dir / f"{stem}_{self.array_suffix}.npy"
                self._memmap_cache[source_file] = np.load(path, mmap_mode="r")
            return self._memmap_cache[source_file]

        def _get_waveform(self, row):
            mm = self._get_memmap(row["source_file"])
            waveform = np.array(mm[int(row["row_index"])])  # copy out of memmap
            waveform = self.scaler.transform(waveform)
            return waveform

        def _get_label(self, row):
            return CLASS_TO_INT[row["class_label"]]

    class PositionAwareWaveformDataset(_BaseRecordingDataset):
        """Input for 1D CNN, position-aware branch (RQ1)."""

        def __init__(self, metadata_df, binary_dir: Path, scaler: Env1TrainOnlyScaler):
            super().__init__(metadata_df, binary_dir, scaler, array_suffix="full")

        def __getitem__(self, idx):
            row = self.meta.iloc[idx]
            waveform = self._get_waveform(row)
            label = self._get_label(row)
            return torch.tensor(waveform, dtype=torch.float32).unsqueeze(0), torch.tensor(label, dtype=torch.long)

    class PositionNormalizedWaveformDataset(_BaseRecordingDataset):
        """Input for 1D CNN, position-normalized branch (RQ6). Only includes
        rows where alignment succeeded."""

        def __init__(self, metadata_df, binary_dir: Path, scaler: Env1TrainOnlyScaler):
            valid = metadata_df[metadata_df["alignment_ok"] == True].copy()  # noqa: E712
            super().__init__(valid, binary_dir, scaler, array_suffix="aligned")

        def __getitem__(self, idx):
            row = self.meta.iloc[idx]
            waveform = self._get_waveform(row)
            label = self._get_label(row)
            return torch.tensor(waveform, dtype=torch.float32).unsqueeze(0), torch.tensor(label, dtype=torch.long)

    class _BaseSpectrogramDataset(_BaseRecordingDataset):
        def __getitem__(self, idx):
            row = self.meta.iloc[idx]
            waveform = self._get_waveform(row)
            spec, _, _ = compute_band_limited_spectrogram(waveform)
            label = self._get_label(row)
            return torch.tensor(spec, dtype=torch.float32).unsqueeze(0), torch.tensor(label, dtype=torch.long)

    class PositionAwareSpectrogramDataset(_BaseSpectrogramDataset):
        """Input for 2D CNN, position-aware branch (RQ1)."""

        def __init__(self, metadata_df, binary_dir: Path, scaler: Env1TrainOnlyScaler):
            super().__init__(metadata_df, binary_dir, scaler, array_suffix="full")

    class PositionNormalizedSpectrogramDataset(_BaseSpectrogramDataset):
        """Input for 2D CNN, position-normalized branch (RQ6)."""

        def __init__(self, metadata_df, binary_dir: Path, scaler: Env1TrainOnlyScaler):
            valid = metadata_df[metadata_df["alignment_ok"] == True].copy()  # noqa: E712
            super().__init__(valid, binary_dir, scaler, array_suffix="aligned")


# =============================================================================
# SECTION 10 — experiments/phase3_preprocessing.py: the runnable pipeline
# =============================================================================
#
# Runs once per file (streamed, bounded memory regardless of file size):
#   1. Parse filename -> metadata
#   2. Convert raw CSV -> memmap binary (full window + aligned crop), while
#      computing onset detection and time/frequency features per row
#   3. Combine into one master per-recording metadata table
#   4. Group-aware Env1 train/val split -- Env2 is never split, held out
#      entirely for final test
#   5. Fit the amplitude scaler on Env1-train ONLY, save as frozen artifact
#
# Nothing here fabricates a result: Env2 subject PD is discovered and
# validated but explicitly excluded from the main manifest (kept in a
# separate bonus table); rows that fail alignment are marked, not dropped
# silently.

def backup_existing_phase3(out_dir: Path) -> Path | None:
    """
    Back up the existing Phase 3 output directory.

    Behavior:
      - If phase3 does not exist: do nothing.
      - If phase3 is empty: delete it.
      - If phase3 contains files:
          1. Create a timestamped backup.
          2. ONLY after the backup succeeds, delete phase3.
      - Uses pigz -1 if available.
      - Falls back to an uncompressed TAR if pigz is unavailable.
    """

    # Nothing to back up
    if not out_dir.exists():
        return None

    # Make sure the path is actually a directory
    if not out_dir.is_dir():
        raise RuntimeError(
            f"Expected Phase 3 output path to be a directory: {out_dir}"
        )

    # Check whether the directory contains anything
    has_content = any(out_dir.iterdir())

    # If the directory exists but is empty, remove it directly.
    if not has_content:
        shutil.rmtree(out_dir)
        print(f"Removed empty Phase 3 output directory: {out_dir}")
        return None

    # ------------------------------------------------------------------
    # Existing Phase 3 directory contains files.
    # Create a timestamped backup.
    # ------------------------------------------------------------------

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Check whether pigz is installed
    pigz = shutil.which("pigz")

    if pigz:
        extension = ".tar.gz"
    else:
        extension = ".tar"

    backup_path = (
        out_dir.parent
        / f"{out_dir.name}_backup_{timestamp}{extension}"
    )

    # Avoid overwriting an existing backup created in the same second
    counter = 1

    while backup_path.exists():
        backup_path = (
            out_dir.parent
            / f"{out_dir.name}_backup_{timestamp}_{counter}{extension}"
        )
        counter += 1

    print(f"Existing Phase 3 output found: {out_dir}")

    # ------------------------------------------------------------------
    # Create backup using pigz
    # ------------------------------------------------------------------

    if pigz:
        print(
            f"Creating compressed backup with pigz -1: "
            f"{backup_path}"
        )

        parent = out_dir.parent

        tar_proc = subprocess.Popen(
            [
                "tar",
                "-cf",
                "-",
                "-C",
                str(parent),
                out_dir.name,
            ],
            stdout=subprocess.PIPE,
        )

        try:
            with backup_path.open("wb") as output_file:

                pigz_proc = subprocess.Popen(
                    [
                        pigz,
                        "-1",
                    ],
                    stdin=tar_proc.stdout,
                    stdout=output_file,
                )

                # We no longer need the parent's copy of this pipe.
                tar_proc.stdout.close()

                pigz_return = pigz_proc.wait()
                tar_return = tar_proc.wait()

            # Check both processes
            if pigz_return != 0 or tar_return != 0:
                raise RuntimeError(
                    f"Backup failed "
                    f"(tar={tar_return}, pigz={pigz_return})"
                )

        except Exception:

            # Remove incomplete backup
            backup_path.unlink(missing_ok=True)

            # Make sure tar is not left running
            if tar_proc.poll() is None:
                tar_proc.kill()
                tar_proc.wait()

            raise

    # ------------------------------------------------------------------
    # Create backup without pigz
    # ------------------------------------------------------------------

    else:
        print(
            "pigz not found; creating uncompressed TAR backup: "
            f"{backup_path}"
        )

        parent = out_dir.parent

        result = subprocess.run(
            [
                "tar",
                "-cf",
                str(backup_path),
                "-C",
                str(parent),
                out_dir.name,
            ],
            check=False,
        )

        # Check whether tar succeeded
        if result.returncode != 0:

            # Remove incomplete backup
            backup_path.unlink(missing_ok=True)

            raise RuntimeError(
                f"Backup failed (tar={result.returncode})"
            )

    # ------------------------------------------------------------------
    # Backup succeeded
    # ------------------------------------------------------------------

    print(f"Backup complete: {backup_path}")

    # IMPORTANT:
    # Delete the old Phase 3 directory ONLY after the backup
    # has completed successfully.
    try:
        shutil.rmtree(out_dir)

    except Exception as exc:
        raise RuntimeError(
            f"Backup succeeded at {backup_path}, but failed to remove "
            f"old Phase 3 directory {out_dir}: {exc}"
        ) from exc

    print(
        f"Removed old Phase 3 output directory: {out_dir}"
    )

    return backup_path


def main():
    ap = argparse.ArgumentParser(description="Phase 3 preprocessing (single-file)")
    ap.add_argument("--data-dir", type=str, required=True, help="Folder containing the raw CSV files")
    ap.add_argument("--out-dir", type=str, default="artifacts/phase3", help="Output folder for all Phase 3 artifacts")
    ap.add_argument("--chunksize", type=int, default=500, help="Rows per chunk when streaming each CSV")
    ap.add_argument("--val-fraction", type=float, default=0.2, help="Fraction of Env1 time-blocks held out for validation")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    # Safety: preserve the previous Phase 3 run before writing new artifacts.
    backup_existing_phase3(out_dir)

    binary_dir = out_dir / "binary"
    binary_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(csv_files)} files.\n")

    file_manifest = build_file_manifest([f.name for f in csv_files])
    manifest_by_name = {m["source_file"]: m for m in file_manifest}

    all_meta = []
    for fpath in csv_files:
        print(f"Converting {fpath.name} ...")
        file_meta = manifest_by_name[fpath.name]
        row_meta = convert_file(fpath, binary_dir, chunksize=args.chunksize)
        for col, val in file_meta.items():
            row_meta[col] = val
        row_meta["recording_id"] = row_meta["source_file"] + "::row" + row_meta["row_index"].astype(str)
        all_meta.append(row_meta)
        print()

    master = pd.concat(all_meta, ignore_index=True)

    # Bonus subject (PD) held out from main pipeline, kept in a separate table
    is_bonus = master["subject_group"] == "bonus"
    bonus = master[is_bonus].copy()
    main_df = master[~is_bonus].copy()

    print(f"Main dataset: {len(main_df)} recordings. Bonus (PD): {len(bonus)} recordings.")

    # Group-aware split, Env1 only
    env1 = main_df[main_df["environment"] == "Env1"].copy()
    env2 = main_df[main_df["environment"] == "Env2"].copy()  # test-only, untouched

    env1_split, block_log = group_aware_train_val_split(env1, val_fraction=args.val_fraction)
    env2["split"] = "test"

    final = pd.concat([env1_split, env2], ignore_index=True)

    # Fit scaler on Env1-train raw ADC values ONLY (a sample, since full waveform
    # arrays are large; a random subset of rows is enough for mean/std)
    train_rows = env1_split[env1_split["split"] == "train"]
    print(f"\nFitting scaler on {len(train_rows)} Env1-train recordings...")

    sample_vals = []
    for _, row in train_rows.sample(min(len(train_rows), 200), random_state=42).iterrows():
        stem = Path(row["source_file"]).stem
        mm = np.load(binary_dir / f"{stem}_full.npy", mmap_mode="r")
        sample_vals.append(np.array(mm[int(row["row_index"])]))
    sample_vals = np.concatenate(sample_vals)

    scaler = Env1TrainOnlyScaler()
    scaler.fit(sample_vals, source_tag="Env1_train")
    scaler.save(out_dir / "scaler.json")
    print(f"Scaler fit: mean={scaler.mean_:.3f}, std={scaler.std_:.3f} -> saved to {out_dir/'scaler.json'}")

    # Save artifacts
    final.to_csv(out_dir / "master_manifest.csv", index=False)
    bonus.to_csv(out_dir / "bonus_pd_manifest.csv", index=False)
    block_log.to_csv(out_dir / "env1_block_split_log.csv", index=False)

    print(f"\nSaved: {out_dir/'master_manifest.csv'}, bonus_pd_manifest.csv, env1_block_split_log.csv, scaler.json")
    print("\nSplit summary:")
    print(final.groupby(["environment", "condition", "split"], dropna=False).size())

    print("\nAlignment success rate by condition:")
    print(final.groupby("condition")["alignment_ok"].mean())

    print("\nPhase 3 complete.")


if __name__ == "__main__":
    main()
