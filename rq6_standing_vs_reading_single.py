#!/usr/bin/env python3
"""
RQ6 single-file runner: Standing vs Reading, HUMAN-only.

Standalone consolidation of the RQ6 package (no src/, core/, experiments/).

Scientific design
-----------------
RQ6 asks whether Standing vs Reading can be distinguished using HUMAN
recordings, and whether that distinction survives removal of the obvious
posture/distance shortcut.

  position_aware       -> full waveform/window; trivial baseline (posture/
                          onset/distance differ between Standing and Reading).
  position_normalized  -> aligned crop; the actual micro-motion test.

The position-aware result MUST NOT be interpreted as evidence of page-turning.

Target: Standing -> 0, Reading -> 1

Leakage rules:
  * scaler is loaded from Phase 3 and must have been fit on Env1 train only.
  * Env1 train: fitting. Env1 val: early stopping / hyperparameter selection.
  * Env2 test: used once, after the model is frozen.
  * HUMAN-only filtering is enforced inside every RQ6 dataset.

SHORT COMMANDS (each runs BOTH branches)
----------------------------------------
    python3 rq6_standing_vs_reading_single.py develop CNN1       # CNN1D x {position_normalized, position_aware}
    python3 rq6_standing_vs_reading_single.py test    CNN1
    python3 rq6_standing_vs_reading_single.py develop CNN2       # CNN2D x {position_normalized, position_aware}
    python3 rq6_standing_vs_reading_single.py test    CNN2
    python3 rq6_standing_vs_reading_single.py develop CLASSICAL  # shape_invariant (phase4) + full (phase3)
    python3 rq6_standing_vs_reading_single.py test    CLASSICAL

Only one branch:
    python3 rq6_standing_vs_reading_single.py develop CNN1 --branch position_aware

Output layout (artifacts/rq6 by default):
    artifacts/rq6/position_normalized_cnn1d/       artifacts/rq6/position_aware_cnn1d/
    artifacts/rq6/position_normalized_cnn2d/       artifacts/rq6/position_aware_cnn2d/
    artifacts/rq6/position_normalized_classical/   artifacts/rq6/position_aware_classical/

Each develop run backs up and clears ONLY its own folder, so CNN1 / CNN2 /
CLASSICAL results never overwrite each other.

The long form still works, with defaults so most flags can be omitted:
    python3 rq6_standing_vs_reading_single.py develop --model-family cnn \
        --branch position_normalized --architecture cnn1d
"""

from __future__ import annotations

import os

# macOS/XGBoost stability: limit native thread pools before importing xgboost.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import argparse
import copy
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    Dataset = object
    DataLoader = None
    TORCH_AVAILABLE = False

try:
    from scipy.signal import stft
    SCIPY_AVAILABLE = True
except Exception:
    stft = None
    SCIPY_AVAILABLE = False

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix,
    )
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except Exception:
    JOBLIB_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTIVITY_TO_INT = {"Standing": 0, "Reading": 1}
INT_TO_ACTIVITY = {0: "Standing", 1: "Reading"}

FS_HZ = 1_953_125.0
FREQ_BAND_HZ = (20_000.0, 60_000.0)
NPERSEG = 256
NOVERLAP = 224

FEATURE_COLUMNS = [
    "mean", "std", "rms", "peak", "peak_to_peak", "crest_factor",
    "dominant_frequency_hz", "band_energy", "band_energy_fraction",
]
SHAPE_FEATURE_COLUMNS = [
    "std_over_peak", "peak_to_peak_over_peak", "crest_factor",
    "dominant_frequency_hz", "band_energy_fraction",
]
SHAPE_FEATURE_COLUMNS_V2 = [
    "std_over_peak", "peak_to_peak_over_peak", "crest_factor",
    "dominant_frequency_hz",
]
FEATURE_SETS = {
    "full": FEATURE_COLUMNS,
    "aligned": FEATURE_COLUMNS,
    "shape_invariant": SHAPE_FEATURE_COLUMNS,
    "shape_invariant_v2": SHAPE_FEATURE_COLUMNS_V2,
}

HYPERPARAM_GRID = [
    {
        "base_channels": 16, "num_blocks": 3, "dropout": 0.3,
        "epochs": 30, "batch_size": 64, "lr": 1e-3, "patience": 5,
    },
    {
        "base_channels": 32, "num_blocks": 3, "dropout": 0.4,
        "epochs": 30, "batch_size": 64, "lr": 5e-4, "patience": 5,
    },
]

CLASSICAL_HYPERPARAM_GRIDS = {
    "random_forest": [
        {"n_estimators": 100, "max_depth": None},
        {"n_estimators": 200, "max_depth": 10},
    ],
    "linear": [{"C": 0.1}, {"C": 1.0}, {"C": 10.0}],
    "xgboost": [
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1},
        {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.05},
    ],
}

# ----- CLI defaults / shortcut configuration (no ML logic) -----
DEFAULT_MANIFEST = "artifacts/phase3/master_manifest.csv"
DEFAULT_SHAPE_MANIFEST = "artifacts/phase4/shape_feature_manifest.csv"
DEFAULT_BINARY_DIR = "artifacts/phase3/binary"
DEFAULT_SCALER = "artifacts/phase3/scaler.json"
DEFAULT_OUT_ROOT = "artifacts/rq6"

SHORTCUTS = ("CNN1", "CNN2", "CLASSICAL")
SHORTCUT_ARCHITECTURE = {"CNN1": "cnn1d", "CNN2": "cnn2d"}
SHORTCUT_BRANCH_ORDER = ["position_normalized", "position_aware"]
# branch -> classical feature set used by the CLASSICAL shortcut
CLASSICAL_BRANCH_FEATURE_SET = {
    "position_normalized": "shape_invariant",
    "position_aware": "full",
}


# ---------------------------------------------------------------------------
# Device / performance helpers
# ---------------------------------------------------------------------------

def get_device() -> "torch.device":
    """Prefer CUDA, then Apple MPS, then CPU."""
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is not installed. Install the Phase 5/RQ6 requirements "
            "before running CNN commands."
        )
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_torch() -> None:
    if not TORCH_AVAILABLE:
        return
    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def print_device_info() -> None:
    if not TORCH_AVAILABLE:
        print("PyTorch: NOT INSTALLED")
        return
    device = get_device()
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch device: {device}")
    if device.type == "mps":
        print("Apple Silicon MPS acceleration: ENABLED")
    elif device.type == "cuda":
        print("CUDA acceleration: ENABLED")
    else:
        print("GPU acceleration: unavailable; using CPU")


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int):
    if num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


# ---------------------------------------------------------------------------
# Evaluation / results
# ---------------------------------------------------------------------------

def compute_metrics_group(y_true, y_pred, y_prob=None) -> dict:
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn is required for evaluation.")
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = None
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics.update({"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)})
    return metrics


def build_predictions_table(meta_df, y_true, y_pred, y_prob, model_name):
    cols = [
        "environment", "condition", "class_label", "subject", "activity",
        "source_file", "row_index", "recording_id",
    ]
    missing = [c for c in cols if c not in meta_df.columns]
    if missing:
        raise ValueError(f"Manifest is missing required metadata columns: {missing}")
    out = meta_df[cols].copy()
    out["model"] = model_name
    out["y_true"] = np.asarray(y_true)
    out["y_pred"] = np.asarray(y_pred)
    out["y_prob"] = np.asarray(y_prob)
    return out


def aggregate_table(predictions_df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    rows = []
    for keys, group in predictions_df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        metrics = compute_metrics_group(
            group["y_true"], group["y_pred"], group["y_prob"]
        )
        row = dict(zip(group_cols, keys))
        row.update(metrics)
        row["n"] = int(len(group))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

class Env1TrainOnlyScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self._fitted_on = None

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        obj = cls()
        obj.mean_ = d["mean"]
        obj.std_ = d["std"]
        obj._fitted_on = d["fitted_on"]
        if obj._fitted_on != "Env1_train":
            raise RuntimeError(
                f"Scaler at {path} says fitted_on={obj._fitted_on!r}; "
                "RQ6 requires a scaler fit only on Env1_train."
            )
        return obj

    def transform(self, samples):
        if self.mean_ is None:
            raise RuntimeError("Scaler has not been loaded/fitted.")
        return (samples - self.mean_) / (self.std_ if self.std_ > 0 else 1.0)


# ---------------------------------------------------------------------------
# CNN architectures
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:

    class CNN1D(nn.Module):
        def __init__(self, base_channels=16, num_blocks=3, dropout=0.3):
            super().__init__()
            layers = []
            in_ch = 1
            ch = base_channels
            for i in range(num_blocks):
                kernel = 25 if i == 0 else 9
                layers += [
                    nn.Conv1d(in_ch, ch, kernel_size=kernel, padding=kernel // 2),
                    nn.BatchNorm1d(ch),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(kernel_size=4),
                ]
                in_ch = ch
                ch *= 2
            self.conv = nn.Sequential(*layers)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(in_ch, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            return self.head(self.pool(self.conv(x))).squeeze(-1)

    class CNN2D(nn.Module):
        def __init__(self, base_channels=16, num_blocks=3, dropout=0.3):
            super().__init__()
            layers = []
            in_ch = 1
            ch = base_channels
            for _ in range(num_blocks):
                layers += [
                    nn.Conv2d(in_ch, ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(ch),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=(1, 2)),
                ]
                in_ch = ch
                ch *= 2
            self.conv = nn.Sequential(*layers)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(in_ch, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            return self.head(self.pool(self.conv(x))).squeeze(-1)

else:
    CNN1D = None
    CNN2D = None


# ---------------------------------------------------------------------------
# Spectrogram
# ---------------------------------------------------------------------------

def compute_band_limited_spectrogram(samples, fs_hz=FS_HZ,
                                     band_hz=FREQ_BAND_HZ,
                                     nperseg=NPERSEG, noverlap=NOVERLAP):
    if not SCIPY_AVAILABLE:
        raise RuntimeError("scipy is required for CNN2D.")
    x = np.asarray(samples, dtype=np.float64)
    x = x - x.mean()
    freqs, times, zxx = stft(x, fs=fs_hz, nperseg=nperseg, noverlap=noverlap)
    magnitude = np.abs(zxx)
    mask = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    return magnitude[mask, :].astype(np.float32), freqs[mask], times


# ---------------------------------------------------------------------------
# RQ6 datasets
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:

    class _BaseRQ6Dataset(Dataset):
        def __init__(self, metadata_df, binary_dir, scaler,
                     array_suffix, amplitude_normalize=False):
            human = metadata_df[
                metadata_df["class_label"].astype(str).str.upper() == "HUMAN"
            ].copy()
            if array_suffix == "aligned":
                if "alignment_ok" not in human.columns:
                    raise ValueError(
                        "Aligned/position_normalized branch requires an "
                        "'alignment_ok' column in the manifest."
                    )
                human = human[human["alignment_ok"] == True].copy()  # noqa: E712

            human["y"] = human["activity"].map(ACTIVITY_TO_INT)
            human = human[human["y"].notna()].copy()
            human["y"] = human["y"].astype(int)
            self.meta = human.reset_index(drop=True)
            self.binary_dir = Path(binary_dir)
            self.scaler = scaler
            self.array_suffix = array_suffix
            self.amplitude_normalize = amplitude_normalize

            self._source_files = self.meta["source_file"].astype(str).to_numpy()
            self._row_indices = self.meta["row_index"].astype(int).to_numpy()
            self._labels = self.meta["y"].astype(np.int64).to_numpy()
            self._memmap_cache = {}

        def __len__(self):
            return len(self.meta)

        def _get_memmap(self, source_file):
            mm = self._memmap_cache.get(source_file)
            if mm is None:
                stem = Path(source_file).stem
                path = self.binary_dir / f"{stem}_{self.array_suffix}.npy"
                if not path.exists():
                    raise FileNotFoundError(
                        f"Binary array not found: {path}\n"
                        f"Check --binary-dir and --branch."
                    )
                mm = np.load(path, mmap_mode="r")
                self._memmap_cache[source_file] = mm
            return mm

        def _get_waveform(self, idx):
            mm = self._get_memmap(self._source_files[idx])
            waveform = np.asarray(mm[self._row_indices[idx]], dtype=np.float32)
            if self.amplitude_normalize:
                peak = float(np.max(np.abs(waveform)))
                waveform = waveform / (peak + 1e-9)
            else:
                waveform = self.scaler.transform(waveform).astype(np.float32, copy=False)
            return waveform

        def label_at(self, idx):
            return int(self._labels[idx])

    class RQ6WaveformDataset(_BaseRQ6Dataset):
        def __getitem__(self, idx):
            waveform = self._get_waveform(idx)
            return torch.from_numpy(waveform).unsqueeze(0), torch.tensor(
                self.label_at(idx), dtype=torch.long
            )

    class RQ6SpectrogramDataset(_BaseRQ6Dataset):
        def __getitem__(self, idx):
            waveform = self._get_waveform(idx)
            spec, _, _ = compute_band_limited_spectrogram(waveform)
            return torch.from_numpy(spec).unsqueeze(0), torch.tensor(
                self.label_at(idx), dtype=torch.long
            )

else:
    RQ6WaveformDataset = None
    RQ6SpectrogramDataset = None


# ---------------------------------------------------------------------------
# CNN training / wrapper
# ---------------------------------------------------------------------------

def train_cnn(model, train_dataset, val_dataset, epochs=30, batch_size=64,
              lr=1e-3, patience=5, num_workers=0, verbose=True):
    device = get_device()
    model = model.to(device)

    train_loader = make_loader(train_dataset, batch_size, True, num_workers)
    val_loader = make_loader(val_dataset, batch_size, False, num_workers)

    if verbose:
        print(
            f"    Device: {device} | DataLoader workers: {num_workers} | "
            f"{len(train_dataset)} train / {len(val_dataset)} val | "
            f"{len(train_loader)} batches/epoch"
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0
    history = []

    for epoch in range(epochs):
        model.train()
        train_sum = 0.0
        train_n = 0
        n_batches = len(train_loader)
        print_every = max(1, n_batches // 20)

        for batch_idx, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_sum += float(loss.item()) * xb.size(0)
            train_n += xb.size(0)

            if verbose and (
                batch_idx % print_every == 0 or batch_idx == n_batches - 1
            ):
                print(
                    f"    epoch {epoch+1}/{epochs}  "
                    f"batch {batch_idx+1}/{n_batches}  "
                    f"running_train_loss={train_sum/train_n:.4f}" + " " * 20,
                    end="\r", flush=True
                )

        if verbose:
            print()
        train_loss = train_sum / max(train_n, 1)

        model.eval()
        val_sum = 0.0
        val_n = 0
        val_correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True).float()
                logits = model(xb)
                loss = criterion(logits, yb)
                val_sum += float(loss.item()) * xb.size(0)
                val_n += xb.size(0)
                val_correct += int(
                    ((torch.sigmoid(logits) > 0.5).float() == yb).sum().item()
                )

        val_loss = val_sum / max(val_n, 1)
        val_acc = val_correct / max(val_n, 1)
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })
        print(
            f"    epoch {epoch+1}/{epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(
                    f"    early stopping at epoch {epoch+1} "
                    f"(no val improvement for {patience} epochs)"
                )
                break

    model.load_state_dict(best_state)
    return model, history, best_val_loss


def predict_cnn(model, dataset, batch_size=64, num_workers=0):
    device = get_device()
    model = model.to(device)
    model.eval()
    loader = make_loader(dataset, batch_size, False, num_workers)
    probs_all = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            probs_all.append(torch.sigmoid(model(xb)).cpu().numpy())
    if not probs_all:
        return np.array([], dtype=int), np.array([], dtype=float)
    probs = np.concatenate(probs_all)
    return (probs > 0.5).astype(int), probs


def save_cnn_checkpoint(path, architecture, model_hp, train_hp,
                        state_dict, best_val_loss):
    torch.save({
        "architecture": architecture,
        "model_hp": model_hp,
        "train_hp": train_hp,
        "state_dict": state_dict,
        "best_val_loss": best_val_loss,
    }, path)


def load_cnn_checkpoint(path):
    d = torch.load(path, map_location="cpu")
    architecture = d["architecture"]
    model_hp = d["model_hp"]
    model_cls = CNN1D if architecture == "cnn1d" else CNN2D
    model = model_cls(**model_hp)
    model.load_state_dict(d["state_dict"])
    return model, d


# ---------------------------------------------------------------------------
# Classical models
# ---------------------------------------------------------------------------

def make_classical_model(model_key, hp):
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn is required for classical RQ6.")
    if model_key == "random_forest":
        return RandomForestClassifier(random_state=42, **hp)
    if model_key == "linear":
        return LogisticRegression(max_iter=2000, random_state=42, **hp)
    if model_key == "xgboost":
        if not XGBOOST_AVAILABLE:
            raise RuntimeError("xgboost is not installed.")
        return XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            n_jobs=1,
            tree_method="hist",
            device="cpu",
            verbosity=0,
            **hp,
        )
    raise ValueError(model_key)


def fit_classical(model_key, hp, x_train, y_train, x_val, y_val):
    model = make_classical_model(model_key, hp)
    if model_key == "xgboost":
        x_train_xgb = np.ascontiguousarray(np.asarray(x_train, dtype=np.float32))
        y_train_xgb = np.ascontiguousarray(np.asarray(y_train, dtype=np.int32))
        model.fit(x_train_xgb, y_train_xgb, verbose=False)
    else:
        model.fit(x_train, y_train)
    return model


def classical_predict(model_key, model, x):
    if model_key == "linear":
        x = model["_scaler"].transform(x)
        estimator = model["estimator"]
        return estimator.predict(x), estimator.predict_proba(x)[:, 1]
    return model.predict(x), model.predict_proba(x)[:, 1]


def save_classical_model(model_key, model, path):
    if not JOBLIB_AVAILABLE:
        raise RuntimeError("joblib is required to save classical models.")
    joblib.dump(model, path)


def load_classical_model(path):
    if not JOBLIB_AVAILABLE:
        raise RuntimeError("joblib is required to load classical models.")
    return joblib.load(path)


def load_classical_arrays(manifest_path, feature_cols, split_name):
    df = pd.read_csv(manifest_path, low_memory=False)
    required = {"environment", "split", "class_label", "activity", *feature_cols}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"Manifest {manifest_path} is missing required columns: {missing}"
        )

    df = df[
        (df["environment"] == "Env1") &
        (df["split"] == split_name) &
        (df["class_label"].astype(str).str.upper() == "HUMAN")
    ].copy()
    df["y"] = df["activity"].map(ACTIVITY_TO_INT)
    df = df[df["y"].notna()].copy()

    bad = df[feature_cols].isna().any(axis=1)
    if bad.any():
        print(f"  WARNING: dropping {int(bad.sum())} rows with missing features")
        df = df[~bad]

    return df


def standardize_classical_for_linear(x_train, x_val):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    return scaler.fit_transform(x_train), scaler.transform(x_val), scaler


# ---------------------------------------------------------------------------
# Dataset / manifest helpers
# ---------------------------------------------------------------------------

def validate_manifest_for_cnn(manifest_path):
    df = pd.read_csv(manifest_path, low_memory=False)
    needed = {
        "environment", "split", "class_label", "activity",
        "source_file", "row_index", "recording_id",
    }
    missing = sorted(needed - set(df.columns))
    if missing:
        raise ValueError(
            f"Manifest {manifest_path} is missing required CNN columns: {missing}"
        )
    return df


def print_activity_balance(meta, name):
    counts = meta["activity"].value_counts().to_dict()
    print(f"{name}: n={len(meta)}  activity balance={counts}")


def ensure_out_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Output-directory resolution / backup (plumbing only)
# ---------------------------------------------------------------------------

def cnn_manifest(args):
    return args.manifest or DEFAULT_MANIFEST


def classical_manifest(args):
    if args.manifest:
        return args.manifest
    if args.feature_set.startswith("shape_invariant"):
        return args.shape_manifest
    return DEFAULT_MANIFEST


def cnn_branch_dir(args) -> Path:
    root = Path(args.out_dir or DEFAULT_OUT_ROOT)
    return root / f"{args.branch}_{args.architecture}"


def classical_dir(args) -> Path:
    """
    Dedicated classical output folder.
    - model_family == "classical" with explicit --out-dir -> used as-is
      (the shortcut passes the exact folder).
    - otherwise -> <root>/<position_*>_classical so it never collides with CNN output.
    """
    if args.out_dir and args.model_family == "classical":
        return Path(args.out_dir)
    root = Path(args.out_dir or DEFAULT_OUT_ROOT)
    branch = "position_aware" if args.feature_set in ("full",) else "position_normalized"
    return root / f"{branch}_classical"


def backup_and_clear_output_dir(out_dir):
    """
    Back up `out_dir` to a timestamped tar(.gz) beside it, then delete it only
    after the backup succeeded. Called ONLY on the specific folder being
    re-developed, never on the whole artifacts/rq6 root.
    """
    out_dir = Path(out_dir)

    if not out_dir.exists():
        return None

    if not out_dir.is_dir():
        raise RuntimeError(f"Output path exists but is not a directory: {out_dir}")

    if not any(out_dir.iterdir()):
        shutil.rmtree(out_dir)
        return None

    parent = out_dir.parent
    parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tar_path = shutil.which("tar")
    pigz_path = shutil.which("pigz")

    if tar_path is None:
        raise RuntimeError("Cannot back up existing output because 'tar' was not found.")

    ext = ".tar.gz" if pigz_path else ".tar"
    backup_path = parent / f"{out_dir.name}_backup_{timestamp}{ext}"
    counter = 1
    while backup_path.exists():
        backup_path = parent / f"{out_dir.name}_backup_{timestamp}_{counter}{ext}"
        counter += 1

    print("\n" + "=" * 70)
    print("RQ6 OUTPUT BACKUP")
    print("=" * 70)
    print(f"Existing output : {out_dir}")
    print(f"Backup file     : {backup_path}")
    print()

    try:
        if pigz_path:
            with open(backup_path, "wb") as backup_file:
                tar_proc = subprocess.Popen(
                    [tar_path, "-cf", "-", "-C", str(parent), out_dir.name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                pigz_proc = subprocess.Popen(
                    [pigz_path],
                    stdin=tar_proc.stdout,
                    stdout=backup_file,
                    stderr=subprocess.PIPE,
                )
                tar_proc.stdout.close()

                pigz_stderr = pigz_proc.stderr.read().decode("utf-8", errors="replace")
                pigz_returncode = pigz_proc.wait()
                tar_stderr = tar_proc.stderr.read().decode("utf-8", errors="replace")
                tar_returncode = tar_proc.wait()

            if tar_returncode != 0 or pigz_returncode != 0:
                raise RuntimeError(
                    "Backup failed.\n"
                    f"tar return code: {tar_returncode}\n"
                    f"pigz return code: {pigz_returncode}\n"
                    f"tar stderr: {tar_stderr.strip()}\n"
                    f"pigz stderr: {pigz_stderr.strip()}"
                )
        else:
            result = subprocess.run(
                [tar_path, "-cf", str(backup_path), "-C", str(parent), out_dir.name],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "Backup failed.\n"
                    f"tar return code: {result.returncode}\n"
                    f"tar stderr: {result.stderr.strip()}"
                )
    except Exception:
        if backup_path.exists():
            try:
                backup_path.unlink()
            except Exception:
                pass
        raise

    print(f"Backup completed successfully:\n  {backup_path}")

    try:
        shutil.rmtree(out_dir)
    except Exception as exc:
        raise RuntimeError(
            "Backup succeeded, but old output directory could not "
            f"be removed: {out_dir}"
        ) from exc

    print(f"Removed old RQ6 output directory:\n  {out_dir}")
    print("=" * 70 + "\n")

    return backup_path


# ---------------------------------------------------------------------------
# CNN develop/test
# ---------------------------------------------------------------------------

def run_cnn_develop(args):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for CNN RQ6.")
    if not SCIPY_AVAILABLE and args.architecture == "cnn2d":
        raise RuntimeError("scipy is required for CNN2D.")
    configure_torch()
    print_device_info()

    out_dir = cnn_branch_dir(args)
    backup_and_clear_output_dir(out_dir)  # only this branch/architecture folder
    frozen_dir = out_dir / "frozen_models"
    frozen_dir.mkdir(parents=True, exist_ok=True)

    df = validate_manifest_for_cnn(cnn_manifest(args))
    env1 = df[df["environment"] == "Env1"].copy()
    train_meta = env1[env1["split"] == "train"]
    val_meta = env1[env1["split"] == "val"]

    scaler = Env1TrainOnlyScaler.load(args.scaler_path)
    array_suffix = "full" if args.branch == "position_aware" else "aligned"
    dataset_cls = RQ6WaveformDataset if args.architecture == "cnn1d" else RQ6SpectrogramDataset

    train_ds = dataset_cls(
        train_meta, args.binary_dir, scaler,
        array_suffix=array_suffix,
        amplitude_normalize=args.amplitude_normalize,
    )
    val_ds = dataset_cls(
        val_meta, args.binary_dir, scaler,
        array_suffix=array_suffix,
        amplitude_normalize=args.amplitude_normalize,
    )

    print(
        f"\nRQ6 CNN development\n"
        f"Branch={args.branch}  Architecture={args.architecture}  "
        f"amplitude_normalize={args.amplitude_normalize}"
    )
    print_activity_balance(train_ds.meta, "train HUMAN-only")
    print_activity_balance(val_ds.meta, "val HUMAN-only")
    print(
        f"DataLoader workers: {args.num_workers}\n"
        f"Position-aware = explicit posture/distance baseline; "
        f"position-normalized = intended micro-motion test.\n"
    )

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(
            "Empty train/val dataset after HUMAN/alignment filtering. "
            "Check manifest, branch, and binary directory."
        )

    log = []
    best = None

    for hp_base in HYPERPARAM_GRID:
        hp = {**hp_base, "num_workers": args.num_workers}
        print(f"\nTraining CNN candidate: {hp}")
        model_cls = CNN1D if args.architecture == "cnn1d" else CNN2D
        model = model_cls(
            base_channels=hp["base_channels"],
            num_blocks=hp["num_blocks"],
            dropout=hp["dropout"],
        )
        model, history, best_val_loss = train_cnn(
            model, train_ds, val_ds,
            epochs=hp["epochs"],
            batch_size=hp["batch_size"],
            lr=hp["lr"],
            patience=hp["patience"],
            num_workers=args.num_workers,
        )
        val_pred, val_prob = predict_cnn(
            model, val_ds,
            batch_size=hp["batch_size"],
            num_workers=args.num_workers,
        )
        y_val = val_ds.meta["y"].to_numpy()
        metrics = compute_metrics_group(y_val, val_pred, val_prob)
        metrics["best_val_loss"] = best_val_loss

        log.append({"hp": hp, "metrics": metrics, "history": history})
        print(
            f"  -> val_accuracy={metrics['accuracy']:.4f} "
            f"val_f1={metrics['f1']:.4f} "
            f"val_roc_auc={metrics['roc_auc']}"
        )

        if best is None or metrics["f1"] > best["metrics"]["f1"]:
            best = {
                "hp": hp,
                "model": model,
                "metrics": metrics,
                "history": history,
            }

    frozen_path = frozen_dir / f"{args.architecture}.pt"
    model_hp = {
        k: best["hp"][k] for k in ("base_channels", "num_blocks", "dropout")
    }
    train_hp = {
        k: best["hp"][k]
        for k in ("epochs", "batch_size", "lr", "patience", "num_workers")
    }
    save_cnn_checkpoint(
        frozen_path, args.architecture, model_hp, train_hp,
        best["model"].state_dict(), best["metrics"]["best_val_loss"],
    )

    with open(out_dir / "rq6_hyperparameter_search_log.json", "w") as f:
        json.dump(log, f, indent=2, default=str)

    pd.DataFrame([{
        "target": "Standing_vs_Reading",
        "branch": args.branch,
        "architecture": args.architecture,
        "amplitude_normalize": bool(args.amplitude_normalize),
        "best_hp": json.dumps(best["hp"]),
        **best["metrics"],
    }]).to_csv(out_dir / "rq6_val_model_selection_summary.csv", index=False)

    print(
        f"\nFROZEN model saved: {frozen_path}\n"
        f"Best Env1-val F1: {best['metrics']['f1']:.4f}\n"
        f"RQ6 Env1 development complete: {out_dir}"
    )


def run_cnn_test(args):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for CNN RQ6.")
    configure_torch()
    print_device_info()

    out_dir = cnn_branch_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    frozen_dir = Path(args.frozen_dir) if args.frozen_dir else out_dir / "frozen_models"
    frozen_path = frozen_dir / f"{args.architecture}.pt"
    if not frozen_path.exists():
        raise FileNotFoundError(f"No frozen model at {frozen_path}. Run develop first.")

    df = validate_manifest_for_cnn(cnn_manifest(args))
    test_meta = df[(df["environment"] == "Env2") & (df["split"] == "test")].copy()

    scaler = Env1TrainOnlyScaler.load(args.scaler_path)
    array_suffix = "full" if args.branch == "position_aware" else "aligned"
    dataset_cls = RQ6WaveformDataset if args.architecture == "cnn1d" else RQ6SpectrogramDataset
    test_ds = dataset_cls(
        test_meta, args.binary_dir, scaler,
        array_suffix=array_suffix,
        amplitude_normalize=args.amplitude_normalize,
    )

    print(
        f"\nRQ6 CNN Env2 test\n"
        f"Branch={args.branch}  Architecture={args.architecture}  "
        f"amplitude_normalize={args.amplitude_normalize}"
    )
    print_activity_balance(test_ds.meta, "Env2 test HUMAN-only")

    if len(test_ds) == 0:
        raise RuntimeError("Empty Env2 test dataset after HUMAN/alignment filtering.")

    model, checkpoint = load_cnn_checkpoint(frozen_path)
    batch_size = int(checkpoint["train_hp"].get("batch_size", 64))
    y_pred, y_prob = predict_cnn(
        model, test_ds, batch_size=batch_size, num_workers=args.num_workers
    )
    y_true = test_ds.meta["y"].to_numpy()
    metrics = compute_metrics_group(y_true, y_pred, y_prob)

    model_name = f"{args.architecture}_{args.branch}"
    predictions = build_predictions_table(
        test_ds.meta, y_true, y_pred, y_prob, model_name
    )
    predictions.to_csv(out_dir / "rq6_env2_test_predictions.csv", index=False)

    pd.DataFrame([{
        "target": "Standing_vs_Reading",
        "branch": args.branch,
        "architecture": args.architecture,
        "amplitude_normalize": bool(args.amplitude_normalize),
        **metrics,
    }]).to_csv(out_dir / "rq6_env2_test_summary.csv", index=False)

    for group_cols, filename in [
        (["model"], "rq6_agg_model.csv"),
        (["model", "subject"], "rq6_agg_model_x_subject.csv"),
        (["model", "activity"], "rq6_agg_model_x_activity.csv"),
        (["model", "condition"], "rq6_agg_model_x_condition.csv"),
    ]:
        aggregate_table(predictions, group_cols).to_csv(out_dir / filename, index=False)

    print("\nEnv2 TEST metrics (frozen Env1 model):")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"\nSaved predictions and aggregates to: {out_dir}")
    print(
        "\nInterpretation guardrail: position_aware measures separability that "
        "may be explained by posture/distance; position_normalized is the "
        "RQ6 micro-motion test."
    )


# ---------------------------------------------------------------------------
# Classical develop/test
# ---------------------------------------------------------------------------

def run_classical_develop(args):
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn is required for classical RQ6.")

    out_dir = classical_dir(args)
    backup_and_clear_output_dir(out_dir)  # only this classical folder
    frozen_dir = out_dir / "frozen_models"
    frozen_dir.mkdir(parents=True, exist_ok=True)

    manifest = classical_manifest(args)
    feature_cols = FEATURE_SETS[args.feature_set]
    train = load_classical_arrays(manifest, feature_cols, "train")
    val = load_classical_arrays(manifest, feature_cols, "val")

    print(
        f"\nRQ6 classical development\n"
        f"Feature set={args.feature_set}\n"
        f"Manifest={manifest}\n"
        f"Features={feature_cols}"
    )
    print_activity_balance(train, "train HUMAN-only")
    print_activity_balance(val, "val HUMAN-only")

    if len(train) == 0 or len(val) == 0:
        raise RuntimeError("Empty classical train/val dataset.")

    x_train_raw = np.ascontiguousarray(train[feature_cols].to_numpy(dtype=np.float32))
    y_train = np.ascontiguousarray(train["y"].to_numpy(dtype=np.int32))
    x_val_raw = np.ascontiguousarray(val[feature_cols].to_numpy(dtype=np.float32))
    y_val = np.ascontiguousarray(val["y"].to_numpy(dtype=np.int32))

    model_keys = ["random_forest", "linear"]
    if XGBOOST_AVAILABLE:
        model_keys.append("xgboost")
    else:
        print("NOTE: xgboost not installed; skipping XGBoost.")

    summary_rows = []
    all_logs = {}

    for model_key in model_keys:
        print(f"\nTuning {model_key}...")
        candidates = CLASSICAL_HYPERPARAM_GRIDS[model_key]
        logs = []
        best = None

        for hp in candidates:
            if model_key == "linear":
                xtr, xv, feature_scaler = standardize_classical_for_linear(
                    x_train_raw, x_val_raw
                )
                estimator = fit_classical(model_key, hp, xtr, y_train, xv, y_val)
                model_bundle = {
                    "estimator": estimator,
                    "_scaler": feature_scaler,
                    "model_key": model_key,
                    "hp": hp,
                }
                val_pred, val_prob = classical_predict(
                    model_key, model_bundle, x_val_raw
                )
            else:
                estimator = fit_classical(
                    model_key, hp, x_train_raw, y_train, x_val_raw, y_val
                )
                model_bundle = estimator
                val_pred = estimator.predict(x_val_raw)
                val_prob = estimator.predict_proba(x_val_raw)[:, 1]

            metrics = compute_metrics_group(y_val, val_pred, val_prob)
            logs.append({"hp": hp, **metrics})
            print(
                f"  {model_key} hp={hp} -> "
                f"val_accuracy={metrics['accuracy']:.4f} "
                f"val_f1={metrics['f1']:.4f}"
            )
            if best is None or metrics["f1"] > best["metrics"]["f1"]:
                best = {"hp": hp, "model": model_bundle, "metrics": metrics}

        frozen_path = frozen_dir / f"{model_key}.joblib"
        save_classical_model(model_key, best["model"], frozen_path)
        summary_rows.append({
            "target": "Standing_vs_Reading",
            "feature_set": args.feature_set,
            "model": model_key,
            "best_hp": json.dumps(best["hp"]),
            **best["metrics"],
        })
        all_logs[model_key] = logs
        print(
            f"  FROZEN {model_key}: val_f1={best['metrics']['f1']:.4f} "
            f"-> {frozen_path}"
        )

    pd.DataFrame(summary_rows).to_csv(
        out_dir / "rq6_val_model_selection_summary.csv", index=False
    )
    with open(out_dir / "rq6_hyperparameter_search_log.json", "w") as f:
        json.dump(all_logs, f, indent=2, default=str)

    print(f"\nRQ6 classical Env1 development complete: {out_dir}")


def run_classical_test(args):
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn is required for classical RQ6.")

    out_dir = ensure_out_dir(classical_dir(args))
    frozen_dir = Path(args.frozen_dir) if args.frozen_dir else out_dir / "frozen_models"
    feature_cols = FEATURE_SETS[args.feature_set]

    df = pd.read_csv(classical_manifest(args), low_memory=False)
    required = {
        "environment", "split", "class_label", "activity",
        "source_file", "row_index", "recording_id", *feature_cols,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    test = df[
        (df["environment"] == "Env2") &
        (df["split"] == "test") &
        (df["class_label"].astype(str).str.upper() == "HUMAN")
    ].copy()
    test["y"] = test["activity"].map(ACTIVITY_TO_INT)
    test = test[test["y"].notna()].copy()

    bad = test[feature_cols].isna().any(axis=1)
    if bad.any():
        print(f"WARNING: dropping {int(bad.sum())} Env2 rows with missing features")
        test = test[~bad]

    print("\nRQ6 classical Env2 test")
    print(f"Feature set={args.feature_set}  Output={out_dir}")
    print_activity_balance(test, "Env2 test HUMAN-only")

    if len(test) == 0:
        raise RuntimeError("Empty Env2 test dataset.")

    x_test = np.ascontiguousarray(test[feature_cols].to_numpy(dtype=np.float32))
    y_test = np.ascontiguousarray(test["y"].to_numpy(dtype=np.int32))
    all_predictions = []
    summary = []

    for model_key in ["random_forest", "linear", "xgboost"]:
        frozen_path = frozen_dir / f"{model_key}.joblib"
        if not frozen_path.exists():
            continue
        try:
            model = load_classical_model(frozen_path)
        except Exception as exc:
            print(f"SKIPPING {model_key}: {exc}")
            continue

        if model_key == "linear":
            y_pred, y_prob = classical_predict(model_key, model, x_test)
        else:
            y_pred = model.predict(x_test)
            y_prob = model.predict_proba(x_test)[:, 1]

        metrics = compute_metrics_group(y_test, y_pred, y_prob)
        summary.append({
            "target": "Standing_vs_Reading",
            "feature_set": args.feature_set,
            "model": model_key,
            **metrics,
        })

        all_predictions.append(
            build_predictions_table(test, y_test, y_pred, y_prob, model_key)
        )
        print(
            f"  {model_key}: accuracy={metrics['accuracy']:.4f} "
            f"precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f} "
            f"f1={metrics['f1']:.4f} "
            f"roc_auc={metrics['roc_auc']}"
        )

    if not all_predictions:
        raise RuntimeError(
            f"No frozen classical models found in {frozen_dir}. Run develop first."
        )

    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_csv(out_dir / "rq6_env2_test_predictions.csv", index=False)
    pd.DataFrame(summary).to_csv(out_dir / "rq6_env2_test_summary.csv", index=False)

    for group_cols, filename in [
        (["model"], "rq6_agg_model.csv"),
        (["model", "subject"], "rq6_agg_model_x_subject.csv"),
        (["model", "activity"], "rq6_agg_model_x_activity.csv"),
        (["model", "condition"], "rq6_agg_model_x_condition.csv"),
    ]:
        aggregate_table(predictions, group_cols).to_csv(out_dir / filename, index=False)

    print(f"\nRQ6 classical Env2 test complete: {out_dir}")


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

def run_smoke():
    print("RQ6 standalone smoke tests")
    print(f"scikit-learn available: {SKLEARN_AVAILABLE}")
    print(f"scipy available: {SCIPY_AVAILABLE}")
    print(f"xgboost available: {XGBOOST_AVAILABLE}")
    print(f"torch available: {TORCH_AVAILABLE}")

    if TORCH_AVAILABLE:
        configure_torch()
        print_device_info()

        for length in (25000, 4900):
            m = CNN1D()
            y = m(torch.randn(2, 1, length))
            assert y.shape == (2,), f"CNN1D shape failed: {y.shape}"
        print("PASS: CNN1D 25,000/4,900-sample shapes")

        for time_bins in (783, 155):
            m = CNN2D()
            y = m(torch.randn(2, 1, 5, time_bins))
            assert y.shape == (2,), f"CNN2D shape failed: {y.shape}"
        print("PASS: CNN2D 5x783/5x155 shapes")

        class TinyDataset(Dataset):
            def __init__(self, n=64):
                rng = np.random.default_rng(42)
                x0 = rng.normal(-1, 0.4, size=(n // 2, 1, 64)).astype(np.float32)
                x1 = rng.normal(1, 0.4, size=(n - n // 2, 1, 64)).astype(np.float32)
                self.x = np.concatenate([x0, x1])
                self.y = np.concatenate([
                    np.zeros(len(x0), dtype=np.int64),
                    np.ones(len(x1), dtype=np.int64),
                ])

            def __len__(self):
                return len(self.y)

            def __getitem__(self, i):
                return torch.from_numpy(self.x[i]), torch.tensor(self.y[i])

        ds = TinyDataset(64)
        tr = torch.utils.data.Subset(ds, range(48))
        va = torch.utils.data.Subset(ds, range(48, 64))
        m = CNN1D(base_channels=4, num_blocks=2, dropout=0.1)
        m, hist, best_loss = train_cnn(
            m, tr, va, epochs=4, batch_size=16, lr=1e-3,
            patience=3, num_workers=0, verbose=False,
        )
        assert hist[-1]["val_loss"] < hist[0]["val_loss"] or hist[-1]["val_acc"] >= 0.8
        print("PASS: CNN training smoke test")

    if SKLEARN_AVAILABLE:
        rng = np.random.default_rng(42)
        x = rng.normal(size=(120, 4))
        y = (x[:, 0] + x[:, 1] > 0).astype(int)
        rf = RandomForestClassifier(n_estimators=20, random_state=42)
        rf.fit(x[:90], y[:90])
        assert len(rf.predict(x[90:])) == 30
        print("PASS: classical Random Forest smoke test")

    print("ALL AVAILABLE SMOKE TESTS PASSED")


# ---------------------------------------------------------------------------
# Shortcuts: CNN1 / CNN2 / CLASSICAL
# ---------------------------------------------------------------------------

def build_shortcut_jobs(args):
    """
    Expand a shortcut name into a list of fully-specified per-run argument sets.
      CNN1      -> cnn1d on position_normalized, position_aware
      CNN2      -> cnn2d on position_normalized, position_aware
      CLASSICAL -> position_normalized (shape_invariant, phase4 manifest)
                   position_aware      (full, phase3 manifest)
    --branch restricts the run to a single branch.
    """
    branches = [args.branch] if args.branch else list(SHORTCUT_BRANCH_ORDER)
    root = args.out_dir or DEFAULT_OUT_ROOT
    jobs = []

    for branch in branches:
        job = copy.copy(args)
        job.branch = branch
        job.frozen_dir = None  # never share one frozen dir across several runs

        if args.model in SHORTCUT_ARCHITECTURE:
            job.model_family = "cnn"
            job.architecture = SHORTCUT_ARCHITECTURE[args.model]
            job.out_dir = root
            job.manifest = args.manifest or DEFAULT_MANIFEST
        else:  # CLASSICAL
            job.model_family = "classical"
            job.feature_set = CLASSICAL_BRANCH_FEATURE_SET[branch]
            job.out_dir = str(Path(root) / f"{branch}_classical")
            job.manifest = (
                args.shape_manifest if job.feature_set == "shape_invariant"
                else DEFAULT_MANIFEST
            )
        jobs.append(job)

    return jobs


def run_shortcut(args, stage):
    jobs = build_shortcut_jobs(args)
    print("=" * 70)
    print(f"RQ6 {args.model} {stage.upper()}  ({len(jobs)} run(s))")
    print("=" * 70)

    for i, job in enumerate(jobs, start=1):
        label = (
            f"{job.architecture} / {job.branch}" if job.model_family == "cnn"
            else f"classical / {job.branch} ({job.feature_set})"
        )
        print("\n" + "=" * 70)
        print(f"{stage.upper()} {i}/{len(jobs)}: {label}")
        print("=" * 70)
        dispatch_stage(job, stage)

    print("\n" + "=" * 70)
    print(f"RQ6 {args.model} {stage.upper()} COMPLETE")
    print("=" * 70)


def dispatch_stage(args, stage):
    if stage == "develop":
        if args.model_family in ("cnn", "both"):
            run_cnn_develop(args)
        if args.model_family in ("classical", "both"):
            run_classical_develop(args)
    else:
        if args.model_family in ("cnn", "both"):
            run_cnn_test(args)
        if args.model_family in ("classical", "both"):
            run_classical_test(args)


def run_stage(args, stage):
    if args.model in SHORTCUTS:
        run_shortcut(args, stage)
        return
    if args.branch is None:
        args.branch = "position_normalized"
    dispatch_stage(args, stage)


def run_develop(args):
    # NOTE: backup/clear now happens per output folder inside the develop
    # functions, so unrelated results under artifacts/rq6 are never deleted.
    run_stage(args, "develop")


def run_test(args):
    run_stage(args, "test")


def run_all(args):
    run_stage(args, "develop")
    run_stage(args, "test")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_common_args(ap):
    ap.add_argument(
        "model", nargs="?", choices=list(SHORTCUTS), default=None,
        help="Shortcut: CNN1 (cnn1d), CNN2 (cnn2d) or CLASSICAL; runs both "
             "branches unless --branch is given.",
    )
    ap.add_argument(
        "--model-family", choices=["cnn", "classical", "both"], default="cnn",
        help="Model family (ignored when a shortcut is used).",
    )
    ap.add_argument(
        "--manifest", type=str, default=None,
        help=f"Default: {DEFAULT_MANIFEST} (classical shape_invariant sets "
             f"use {DEFAULT_SHAPE_MANIFEST}).",
    )
    ap.add_argument("--shape-manifest", type=str, default=DEFAULT_SHAPE_MANIFEST)
    ap.add_argument("--binary-dir", type=str, default=DEFAULT_BINARY_DIR)
    ap.add_argument(
        "--branch", choices=["position_aware", "position_normalized"],
        default=None,
        help="Default: position_normalized for the long form; both branches "
             "for shortcuts.",
    )
    ap.add_argument(
        "--architecture", choices=["cnn1d", "cnn2d"], default="cnn1d",
        help="CNN architecture. Ignored for classical and for CNN1/CNN2 shortcuts.",
    )
    ap.add_argument(
        "--feature-set", choices=list(FEATURE_SETS.keys()),
        default="shape_invariant", help="Classical feature set.",
    )
    ap.add_argument("--scaler-path", default=DEFAULT_SCALER, type=str)
    ap.add_argument(
        "--out-dir", default=None, type=str,
        help=f"Default: {DEFAULT_OUT_ROOT}",
    )
    ap.add_argument("--frozen-dir", default=None, type=str)
    ap.add_argument("--amplitude-normalize", action="store_true")
    ap.add_argument(
        "--num-workers", type=int, default=2,
        help="PyTorch DataLoader workers. Use 0 if macOS multiprocessing is unstable.",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Standalone RQ6 Standing-vs-Reading runner."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    smoke = sub.add_parser("smoke", help="Run local architecture/pipeline smoke tests.")
    smoke.set_defaults(func=lambda a: run_smoke())

    develop = sub.add_parser("develop", help="Env1 train/validation development.")
    add_common_args(develop)
    develop.set_defaults(func=run_develop)

    test = sub.add_parser("test", help="Frozen Env1 model -> Env2 test only.")
    add_common_args(test)
    test.set_defaults(func=run_test)

    all_cmd = sub.add_parser("all", help="Run develop, then test.")
    add_common_args(all_cmd)
    all_cmd.set_defaults(func=run_all)

    args = parser.parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()