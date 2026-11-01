#!/usr/bin/env python3
"""
Phase 5 standalone runner (single file).

Examples:
    python phase5_deep_learning.py smoke
    python phase5_deep_learning.py develop CNN1
    python phase5_deep_learning.py test CNN1
    python phase5_deep_learning.py develop CNN2
    python phase5_deep_learning.py test CNN2
    python phase5_deep_learning.py develop --branch position_aware --architecture cnn1d
    python phase5_deep_learning.py all --branch position_aware --architecture cnn1d
"""

import argparse
import copy
import json
import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import stft
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

SCRIPT_NAME = "phase5_deep_learning.py"

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")


# ============================================================
# BACKUP / CLEANUP
# ============================================================

def backup_existing_phase5(out_dir: Path):
    """
    Back up `out_dir` to a timestamped tar(.gz) beside it, and delete the
    original ONLY after the backup succeeds.

    FIX: this is now called only on the specific branch/architecture folder
    (e.g. artifacts/phase5/position_aware_cnn2d), never on the whole
    artifacts/phase5 directory, so CNN2 runs no longer wipe CNN1 results.
    """
    out_dir = Path(out_dir)

    if not out_dir.exists():
        return None

    if not out_dir.is_dir():
        raise RuntimeError(f"Expected a directory, but found: {out_dir}")

    if not any(out_dir.iterdir()):
        shutil.rmtree(out_dir)
        print(f"Removed empty output directory: {out_dir}")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pigz = shutil.which("pigz")
    extension = ".tar.gz" if pigz else ".tar"

    backup_path = out_dir.parent / f"{out_dir.name}_backup_{timestamp}{extension}"
    counter = 1
    while backup_path.exists():
        backup_path = out_dir.parent / (
            f"{out_dir.name}_backup_{timestamp}_{counter}{extension}"
        )
        counter += 1

    print()
    print("=" * 70)
    print("Existing output detected")
    print("=" * 70)
    print(f"Existing directory : {out_dir}")
    print(f"Backup destination : {backup_path}")

    parent = out_dir.parent

    if pigz:
        print("pigz detected.")
        print("Creating compressed backup with pigz -1...")

        tar_proc = subprocess.Popen(
            ["tar", "-cf", "-", "-C", str(parent), out_dir.name],
            stdout=subprocess.PIPE,
        )
        try:
            with backup_path.open("wb") as output_file:
                pigz_proc = subprocess.Popen(
                    [pigz, "-1"], stdin=tar_proc.stdout, stdout=output_file
                )
                tar_proc.stdout.close()
                pigz_return = pigz_proc.wait()
                tar_return = tar_proc.wait()

            if pigz_return != 0 or tar_return != 0:
                raise RuntimeError(
                    f"Backup failed (tar={tar_return}, pigz={pigz_return})"
                )
        except Exception:
            backup_path.unlink(missing_ok=True)
            if tar_proc.poll() is None:
                tar_proc.kill()
                tar_proc.wait()
            raise
    else:
        print("pigz not found.")
        print("Creating uncompressed TAR backup...")

        result = subprocess.run(
            ["tar", "-cf", str(backup_path), "-C", str(parent), out_dir.name],
            check=False,
        )
        if result.returncode != 0:
            backup_path.unlink(missing_ok=True)
            raise RuntimeError(f"Backup failed (tar={result.returncode})")

    print(f"Backup complete: {backup_path}")

    # Delete the old directory ONLY after the backup completed successfully.
    try:
        shutil.rmtree(out_dir)
    except Exception as exc:
        raise RuntimeError(
            f"Backup succeeded at {backup_path}, but the old directory "
            f"could not be removed: {out_dir}\nReason: {exc}"
        ) from exc

    print(f"Removed old output directory: {out_dir}")
    print("=" * 70)
    print()

    return backup_path


def prepare_output_dir(out_dir: Path) -> Path:
    """Back up + remove previous output, then create a clean directory."""
    out_dir = Path(out_dir)
    backup_existing_phase5(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Fresh output directory ready: {out_dir}")
    return out_dir


# ============================================================
# SCALING
# ============================================================

class Env1TrainOnlyScaler:
    """Amplitude scaler. MUST be fitted only on Env1 training data."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self._fitted_on = None

    def fit(self, samples_2d: np.ndarray, source_tag: str):
        if source_tag != "Env1_train":
            raise RuntimeError(
                f"Refusing to fit scaler on '{source_tag}'. "
                "Per Section 15, the scaler must be fit ONLY on "
                "Env1 train data. Use .transform() for Env1 val "
                "and all of Env2."
            )
        self.mean_ = samples_2d.mean()
        self.std_ = samples_2d.std()
        self._fitted_on = source_tag
        return self

    def transform(self, samples_2d: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Scaler has not been fit yet.")
        return (samples_2d - self.mean_) / (self.std_ if self.std_ > 0 else 1.0)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(
                {
                    "mean": float(self.mean_),
                    "std": float(self.std_),
                    "fitted_on": self._fitted_on,
                },
                f,
            )

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            d = json.load(f)
        obj = cls()
        obj.mean_ = d["mean"]
        obj.std_ = d["std"]
        obj._fitted_on = d["fitted_on"]
        return obj


# ============================================================
# SPECTROGRAM
# ============================================================

FS_HZ = 1_953_125.0
FREQ_BAND_HZ = (20_000.0, 60_000.0)
NPERSEG = 256
NOVERLAP = 224


def compute_band_limited_spectrogram(
    samples: np.ndarray,
    fs_hz: float = FS_HZ,
    band_hz=FREQ_BAND_HZ,
    nperseg: int = NPERSEG,
    noverlap: int = NOVERLAP,
):
    x = samples.astype(np.float64)
    x = x - x.mean()

    freqs, times, Zxx = stft(x, fs=fs_hz, nperseg=nperseg, noverlap=noverlap)
    magnitude = np.abs(Zxx)

    band_mask = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    band_magnitude = magnitude[band_mask, :]
    band_freqs = freqs[band_mask]

    return band_magnitude.astype(np.float32), band_freqs, times


# ============================================================
# MODEL BASE
# ============================================================

class BaseModel(ABC):
    @abstractmethod
    def fit(self, X_train, y_train=None, X_val=None, y_val=None, **hp): ...

    @abstractmethod
    def predict(self, X): ...

    @abstractmethod
    def predict_proba(self, X): ...

    @abstractmethod
    def save(self, path): ...

    @abstractmethod
    def load(self, path): ...

    @abstractmethod
    def get_params(self): ...


# ============================================================
# CNN 1D / CNN 2D
# ============================================================

class CNN1D(nn.Module):
    def __init__(self, base_channels: int = 16, num_blocks: int = 3, dropout: float = 0.3):
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
        x = self.conv(x)
        x = self.pool(x)
        return self.head(x).squeeze(-1)


class CNN2D(nn.Module):
    def __init__(self, base_channels: int = 16, num_blocks: int = 3, dropout: float = 0.3):
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
        x = self.conv(x)
        x = self.pool(x)
        return self.head(x).squeeze(-1)


# ============================================================
# TRAINING
# ============================================================

def get_device():
    """Select the fastest supported PyTorch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _make_loader(dataset, batch_size, shuffle, num_workers=0):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    return DataLoader(dataset, **kwargs)


def train_model(
    model,
    train_dataset,
    val_dataset,
    epochs=30,
    batch_size=64,
    lr=1e-3,
    patience=5,
    device=None,
    verbose=True,
    num_workers=0,
):
    device = device or get_device()
    model = model.to(device)

    train_loader = _make_loader(train_dataset, batch_size, True, num_workers)
    val_loader = _make_loader(val_dataset, batch_size, False, num_workers)

    if verbose:
        print(f"    Device: {device}")
        if device.type == "mps":
            print("    Apple Silicon MPS acceleration: ENABLED")
        print(f"    DataLoader workers: {num_workers}")
        print(
            f"    {len(train_dataset)} train / {len(val_dataset)} val recordings, "
            f"batch_size={batch_size} -> {len(train_loader)} batches/epoch"
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    history = []

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        n_batches = len(train_loader)
        print_every = max(1, n_batches // 20)

        for batch_idx, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).float()

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * xb.size(0)
            train_n += xb.size(0)

            if verbose and (batch_idx % print_every == 0 or batch_idx == n_batches - 1):
                running_loss = train_loss_sum / train_n
                # FIX: trailing spaces + \r prevent leftover characters from
                # earlier, longer lines (the "ENABLEDers: 2" garbling).
                print(
                    f"    epoch {epoch + 1}/{epochs} "
                    f"batch {batch_idx + 1}/{n_batches} "
                    f"running_train_loss={running_loss:.4f}" + " " * 20,
                    end="\r",
                    flush=True,
                )

        if verbose:
            print()

        train_loss = train_loss_sum / train_n

        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        val_correct = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True).float()

                logits = model(xb)
                loss = criterion(logits, yb)

                val_loss_sum += loss.item() * xb.size(0)
                val_n += xb.size(0)

                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == yb).sum().item()

        val_loss = val_loss_sum / val_n
        val_acc = val_correct / val_n

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        if verbose:
            print(
                f"    epoch {epoch + 1}/{epochs} "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} "
                f"val_acc={val_acc:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            if verbose:
                print(
                    f"    early stopping at epoch {epoch + 1} "
                    f"(no val improvement for {patience} epochs)"
                )
            break

    model.load_state_dict(best_state)
    return model, history, best_val_loss


def predict_with_model(model, dataset, batch_size=64, device=None, num_workers=0):
    device = device or get_device()
    model = model.to(device)
    model.eval()

    loader = _make_loader(dataset, batch_size, False, num_workers)
    all_probs = []

    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

    probs = np.concatenate(all_probs)
    preds = (probs > 0.5).astype(int)
    return preds, probs


# ============================================================
# DATASETS
# ============================================================

CLASS_TO_INT = {"NON_HUMAN": 0, "HUMAN": 1}


class _BaseRecordingDataset(Dataset):
    def __init__(
        self,
        metadata_df,
        binary_dir: Path,
        scaler: Env1TrainOnlyScaler,
        array_suffix: str,
        amplitude_normalize: bool = False,
    ):
        self.meta = metadata_df.reset_index(drop=True)
        self.binary_dir = Path(binary_dir)
        self.scaler = scaler
        self.array_suffix = array_suffix
        self.amplitude_normalize = amplitude_normalize
        self._memmap_cache = {}

        self._source_files = self.meta["source_file"].to_numpy(copy=False)
        self._row_indices = self.meta["row_index"].to_numpy(copy=False)
        self._labels = (
            self.meta["class_label"].map(CLASS_TO_INT).to_numpy(dtype=np.int64, copy=False)
        )

    def __len__(self):
        return len(self._row_indices)

    def _get_memmap(self, source_file: str):
        if source_file not in self._memmap_cache:
            stem = Path(source_file).stem
            path = self.binary_dir / f"{stem}_{self.array_suffix}.npy"
            self._memmap_cache[source_file] = np.load(path, mmap_mode="r")
        return self._memmap_cache[source_file]

    def _get_waveform(self, idx):
        source_file = self._source_files[idx]
        row_index = int(self._row_indices[idx])

        mm = self._get_memmap(source_file)
        waveform = np.asarray(mm[row_index], dtype=np.float32)

        if self.amplitude_normalize:
            peak = np.max(np.abs(waveform))
            waveform = waveform / (peak + 1e-9)
        else:
            waveform = self.scaler.transform(waveform)

        return np.asarray(waveform, dtype=np.float32)

    def _get_label(self, idx):
        return int(self._labels[idx])


class _WaveformItemMixin:
    def __getitem__(self, idx):
        waveform = self._get_waveform(idx)
        label = self._get_label(idx)
        return (
            torch.from_numpy(waveform).unsqueeze(0),
            torch.tensor(label, dtype=torch.long),
        )


class PositionAwareWaveformDataset(_WaveformItemMixin, _BaseRecordingDataset):
    def __init__(self, metadata_df, binary_dir, scaler, amplitude_normalize=False):
        super().__init__(
            metadata_df, binary_dir, scaler,
            array_suffix="full", amplitude_normalize=amplitude_normalize,
        )


class PositionNormalizedWaveformDataset(_WaveformItemMixin, _BaseRecordingDataset):
    def __init__(self, metadata_df, binary_dir, scaler, amplitude_normalize=False):
        valid = metadata_df[metadata_df["alignment_ok"] == True].copy()  # noqa: E712
        super().__init__(
            valid, binary_dir, scaler,
            array_suffix="aligned", amplitude_normalize=amplitude_normalize,
        )


class _BaseSpectrogramDataset(_BaseRecordingDataset):
    def __getitem__(self, idx):
        waveform = self._get_waveform(idx)
        spec, _, _ = compute_band_limited_spectrogram(waveform)
        label = self._get_label(idx)
        return (
            torch.from_numpy(np.ascontiguousarray(spec)).unsqueeze(0),
            torch.tensor(label, dtype=torch.long),
        )


class PositionAwareSpectrogramDataset(_BaseSpectrogramDataset):
    def __init__(self, metadata_df, binary_dir, scaler, amplitude_normalize=False):
        super().__init__(
            metadata_df, binary_dir, scaler,
            array_suffix="full", amplitude_normalize=amplitude_normalize,
        )


class PositionNormalizedSpectrogramDataset(_BaseSpectrogramDataset):
    def __init__(self, metadata_df, binary_dir, scaler, amplitude_normalize=False):
        valid = metadata_df[metadata_df["alignment_ok"] == True].copy()  # noqa: E712
        super().__init__(
            valid, binary_dir, scaler,
            array_suffix="aligned", amplitude_normalize=amplitude_normalize,
        )


# ============================================================
# CNN WRAPPER
# ============================================================

ARCHITECTURES = {"cnn1d": CNN1D, "cnn2d": CNN2D}
_MODEL_HP_KEYS = ("base_channels", "num_blocks", "dropout")
_TRAIN_HP_KEYS = ("epochs", "batch_size", "lr", "patience", "num_workers")


class CNNModel(BaseModel):
    def __init__(self, architecture: str = "cnn1d", **hp):
        assert architecture in ARCHITECTURES, f"Unknown architecture '{architecture}'"
        self.architecture = architecture
        self.hp = hp
        self.model_hp = {k: v for k, v in hp.items() if k in _MODEL_HP_KEYS}
        self.train_hp = {k: v for k, v in hp.items() if k in _TRAIN_HP_KEYS}
        self.model = ARCHITECTURES[architecture](**self.model_hp)
        self.history = None
        self.best_val_loss = None

    def fit(self, X_train, y_train=None, X_val=None, y_val=None, **hp):
        if hp:
            self.hp.update(hp)
            self.model_hp = {k: v for k, v in self.hp.items() if k in _MODEL_HP_KEYS}
            self.train_hp = {k: v for k, v in self.hp.items() if k in _TRAIN_HP_KEYS}
            self.model = ARCHITECTURES[self.architecture](**self.model_hp)

        self.model, self.history, self.best_val_loss = train_model(
            self.model, X_train, X_val, **self.train_hp
        )
        return self

    def _predict_kwargs(self):
        return {
            "batch_size": int(self.train_hp.get("batch_size", 64)),
            "num_workers": int(self.train_hp.get("num_workers", 0)),
        }

    def predict(self, X):
        preds, _ = predict_with_model(self.model, X, **self._predict_kwargs())
        return preds

    def predict_proba(self, X):
        _, probs = predict_with_model(self.model, X, **self._predict_kwargs())
        return probs

    def save(self, path):
        torch.save(
            {
                "architecture": self.architecture,
                "model_hp": self.model_hp,
                "train_hp": self.train_hp,
                "state_dict": self.model.state_dict(),
                "best_val_loss": self.best_val_loss,
            },
            path,
        )

    def load(self, path):
        d = torch.load(path, map_location="cpu")
        self.architecture = d["architecture"]
        self.model_hp = d["model_hp"]
        self.train_hp = d["train_hp"]
        self.hp = {**self.model_hp, **self.train_hp}
        self.model = ARCHITECTURES[self.architecture](**self.model_hp)
        self.model.load_state_dict(d["state_dict"])
        self.best_val_loss = d.get("best_val_loss")
        return self

    def get_params(self):
        return {"architecture": self.architecture, **self.hp}


# ============================================================
# EVALUATION
# ============================================================

def compute_metrics_group(y_true, y_pred, y_prob=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    else:
        metrics["roc_auc"] = None

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (None, None, None, None)

    metrics.update({"tp": tp, "tn": tn, "fp": fp, "fn": fn})
    return metrics


# ============================================================
# RESULTS STORE
# ============================================================

def build_predictions_table(meta_df, y_true, y_pred, y_prob, model_name: str):
    out = meta_df[
        [
            "environment", "condition", "class_label", "subject",
            "activity", "source_file", "row_index", "recording_id",
        ]
    ].copy()
    out["model"] = model_name
    out["y_true"] = y_true
    out["y_pred"] = y_pred
    out["y_prob"] = y_prob
    return out


def append_predictions(predictions_path, new_predictions: pd.DataFrame):
    if os.path.exists(predictions_path):
        existing = pd.read_csv(predictions_path)
        combined = pd.concat([existing, new_predictions], ignore_index=True)
    else:
        combined = new_predictions
    combined.to_csv(predictions_path, index=False)
    return combined


def aggregate_table(predictions_df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    rows = []
    for keys, group in predictions_df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)

        metrics = compute_metrics_group(group["y_true"], group["y_pred"], group["y_prob"])
        row = dict(zip(group_cols, keys))
        row.update(metrics)
        row["n"] = len(group)
        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# CONFIG
# ============================================================

DATASET_CLASSES = {
    ("position_aware", "cnn1d"): PositionAwareWaveformDataset,
    ("position_normalized", "cnn1d"): PositionNormalizedWaveformDataset,
    ("position_aware", "cnn2d"): PositionAwareSpectrogramDataset,
    ("position_normalized", "cnn2d"): PositionNormalizedSpectrogramDataset,
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

SHORTCUT_ARCHITECTURE = {"CNN1": "cnn1d", "CNN2": "cnn2d"}


def _branch_out_dir(args) -> Path:
    return Path(args.out_dir) / f"{args.branch}_{args.architecture}"


def _frozen_path(args) -> Path:
    frozen_dir = (
        Path(args.frozen_dir)
        if getattr(args, "frozen_dir", None)
        else _branch_out_dir(args) / "frozen_models"
    )
    return frozen_dir / f"{args.architecture}.pt"


# ============================================================
# STAGE: DEVELOP
# ============================================================

def run_develop(args):
    out_dir = _branch_out_dir(args)

    # FIX: only this branch/architecture folder is backed up and cleared,
    # so other architectures' results (e.g. CNN1D) are never touched.
    prepare_output_dir(out_dir)

    frozen_dir = out_dir / "frozen_models"
    frozen_dir.mkdir(exist_ok=True)

    df = pd.read_csv(args.manifest, low_memory=False)
    env1 = df[df["environment"] == "Env1"].copy()
    train_meta = env1[env1["split"] == "train"]
    val_meta = env1[env1["split"] == "val"]

    scaler = Env1TrainOnlyScaler.load(args.scaler_path)
    DatasetClass = DATASET_CLASSES[(args.branch, args.architecture)]

    train_ds = DatasetClass(
        train_meta, args.binary_dir, scaler,
        amplitude_normalize=args.amplitude_normalize,
    )
    val_ds = DatasetClass(
        val_meta, args.binary_dir, scaler,
        amplitude_normalize=args.amplitude_normalize,
    )

    print(
        f"Branch={args.branch}  Architecture={args.architecture}  "
        f"amplitude_normalize={args.amplitude_normalize}"
    )
    print(f"PyTorch device: {get_device()}")
    if get_device().type == "mps":
        print("Apple Silicon MPS acceleration: ENABLED")
    print(f"train={len(train_ds)}  val={len(val_ds)}\n")

    log = []
    best = None

    for hp in HYPERPARAM_GRID:
        hp = {**hp, "num_workers": args.num_workers}
        print(f"Training with hp={hp}")

        model = CNNModel(architecture=args.architecture, **hp)
        model.fit(train_ds, X_val=val_ds)

        val_pred, val_prob = predict_with_model(
            model.model, val_ds,
            batch_size=hp["batch_size"], num_workers=args.num_workers,
        )
        y_val = val_ds.meta["class_label"].map(CLASS_TO_INT).values

        metrics = compute_metrics_group(y_val, val_pred, val_prob)
        metrics["best_val_loss"] = model.best_val_loss

        log.append({"hp": hp, "metrics": metrics, "history": model.history})

        print(f"  -> val_f1={metrics['f1']:.4f}  val_loss={model.best_val_loss:.4f}\n")

        if best is None or metrics["f1"] > best["metrics"]["f1"]:
            best = {"hp": hp, "model": model, "metrics": metrics}

    frozen_path = frozen_dir / f"{args.architecture}.pt"
    best["model"].save(frozen_path)

    print(
        f"FROZEN best {args.architecture}: hp={best['hp']}  "
        f"val_f1={best['metrics']['f1']:.4f}  saved to {frozen_path}"
    )

    with open(out_dir / "hyperparameter_search_log.json", "w") as f:
        json.dump(log, f, indent=2, default=str)

    pd.DataFrame(
        [
            {
                "architecture": args.architecture,
                "branch": args.branch,
                "best_hp": json.dumps(best["hp"]),
                **best["metrics"],
            }
        ]
    ).to_csv(out_dir / "env1_val_model_selection_summary.csv", index=False)

    print(
        f"\nSaved: {out_dir}/env1_val_model_selection_summary.csv, "
        f"hyperparameter_search_log.json"
    )
    print(
        "Phase 5 (Env1 development) complete for this branch/architecture. "
        "Model is FROZEN."
    )


# ============================================================
# STAGE: TEST
# ============================================================

def run_test(args):
    out_dir = _branch_out_dir(args)
    frozen_path = _frozen_path(args)

    if not frozen_path.exists():
        print(
            f"No frozen model found at {frozen_path} -- run "
            f"'python {SCRIPT_NAME} develop ...' first for this "
            f"branch/architecture."
        )
        return

    df = pd.read_csv(args.manifest, low_memory=False)
    test_meta = df[(df["environment"] == "Env2") & (df["split"] == "test")]

    scaler = Env1TrainOnlyScaler.load(args.scaler_path)
    DatasetClass = DATASET_CLASSES[(args.branch, args.architecture)]

    test_ds = DatasetClass(
        test_meta, args.binary_dir, scaler,
        amplitude_normalize=args.amplitude_normalize,
    )

    print(
        f"Branch={args.branch}  Architecture={args.architecture}  "
        f"test set: {len(test_ds)} recordings"
    )
    print(f"PyTorch device: {get_device()}")

    model = CNNModel(architecture=args.architecture)
    model.load(frozen_path)

    y_pred, y_prob = predict_with_model(
        model.model, test_ds,
        batch_size=int(model.train_hp.get("batch_size", 64)),
        num_workers=args.num_workers,
    )
    y_true = test_ds.meta["class_label"].map(CLASS_TO_INT).values

    model_name = f"{args.architecture}_{args.branch}"

    predictions = build_predictions_table(
        test_ds.meta, y_true, y_pred, y_prob, model_name=model_name
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out_dir / "env2_test_predictions.csv", index=False)

    print(f"\nSaved: {out_dir / 'env2_test_predictions.csv'}")

    for group_cols, fname in [
        (["model", "environment"], "agg_model_x_environment.csv"),
        (["model", "condition"], "agg_model_x_condition.csv"),
        (["model", "subject"], "agg_model_x_subject.csv"),
        (["model", "activity"], "agg_model_x_activity.csv"),
    ]:
        agg = aggregate_table(predictions, group_cols)
        agg.to_csv(out_dir / fname, index=False)
        print(f"\n{' x '.join(group_cols)}:")
        print(agg.to_string(index=False))

    print(f"\nPhase 5 (Env2 final test) complete for {model_name}.")


# ============================================================
# STAGE: SMOKE
# ============================================================

def run_smoke(args):
    class _SyntheticWaveformDataset(Dataset):
        def __init__(self, n, length, seed=0):
            rng = np.random.default_rng(seed)
            self.X = rng.normal(0, 1, size=(n, length)).astype(np.float32)
            self.y = rng.integers(0, 2, size=n)

            bump_start = length // 2 - 50
            bump_end = length // 2 + 50

            for i in range(n):
                if self.y[i] == 1:
                    self.X[i] *= 1.5
                    self.X[i, bump_start:bump_end] += 3.0

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return (
                torch.tensor(self.X[idx]).unsqueeze(0),
                torch.tensor(self.y[idx], dtype=torch.long),
            )

    def test_cnn1d_shapes():
        print("Test 1: CNN1D output shape for both branch lengths...")
        model = CNN1D(base_channels=8, num_blocks=2)
        for length, label in [(25000, "full"), (4900, "aligned")]:
            x = torch.randn(4, 1, length)
            out = model(x)
            assert out.shape == (4,), (
                f"CNN1D ({label}, length={length}): expected (4,), got {out.shape}"
            )
            print(f"  OK: length={length} ({label}) -> output shape {tuple(out.shape)}")

    def test_cnn2d_shapes():
        print("\nTest 2: CNN2D output shape for both branch spectrogram sizes...")
        model = CNN2D(base_channels=8, num_blocks=2)
        for freq, time, label in [
            (5, 783, "position-aware"),
            (5, 155, "position-normalized"),
        ]:
            x = torch.randn(4, 1, freq, time)
            out = model(x)
            assert out.shape == (4,), (
                f"CNN2D ({label}): expected (4,), got {out.shape}"
            )
            print(f"  OK: shape=({freq},{time}) ({label}) -> output shape {tuple(out.shape)}")

    def test_training_reduces_loss():
        print("\nTest 3: training loop reduces loss on separable synthetic data...")
        train_ds = _SyntheticWaveformDataset(400, 2000, seed=1)
        val_ds = _SyntheticWaveformDataset(100, 2000, seed=2)
        model = CNN1D(base_channels=8, num_blocks=2)

        _, history, _ = train_model(
            model, train_ds, val_ds,
            epochs=25, batch_size=32, lr=2e-3, patience=8, verbose=False,
        )

        first_loss = history[0]["val_loss"]
        last_loss = min(h["val_loss"] for h in history)
        print(f"  first-epoch val_loss={first_loss:.4f}  best val_loss={last_loss:.4f}")
        assert last_loss < first_loss, "Training did not reduce val loss"

        final_val_acc = max(h["val_acc"] for h in history)
        print(f"  best val_acc={final_val_acc:.4f} (expect > 0.85)")
        assert final_val_acc > 0.85, "Val accuracy too low on separable data"
        print("  PASSED")

    def test_save_load_roundtrip():
        print("\nTest 4: CNNModel save/load round-trip...")
        train_ds = _SyntheticWaveformDataset(100, 1000, seed=3)
        val_ds = _SyntheticWaveformDataset(40, 1000, seed=4)

        model = CNNModel(
            architecture="cnn1d", base_channels=8, num_blocks=2,
            epochs=5, batch_size=16, lr=1e-3, patience=3,
        )
        model.fit(train_ds, X_val=val_ds)
        probs_before = model.predict_proba(val_ds)

        tmp_path = "/tmp/cnn_smoke_test.pt"
        model.save(tmp_path)

        model2 = CNNModel(architecture="cnn1d")
        model2.load(tmp_path)
        probs_after = model2.predict_proba(val_ds)

        assert np.allclose(probs_before, probs_after, atol=1e-5), (
            "save/load round-trip produced different predictions"
        )
        print("  PASSED: identical predictions before/after save+load")

    test_cnn1d_shapes()
    test_cnn2d_shapes()
    test_training_reduces_loss()
    test_save_load_roundtrip()
    print("\nAll Phase 5 smoke tests PASSED.")


# ============================================================
# STAGE: ALL
# ============================================================

def run_all(args):
    run_develop(args)
    print("\n" + "=" * 70 + "\n")
    run_test(args)


# ============================================================
# SHORTCUTS: CNN1 / CNN2
# ============================================================

def _run_shortcut(args, stage: str):
    """
    FIX: the architecture is derived from the shortcut name (CNN1 -> cnn1d,
    CNN2 -> cnn2d) instead of the --architecture default ("cnn1d"), which
    previously made `develop CNN2` silently train the 1D CNN.

    FIX: no whole-Phase-5 backup/wipe here. Each branch/architecture folder
    is backed up and cleared individually inside run_develop(), so CNN1D and
    CNN2D results coexist.
    """
    architecture = SHORTCUT_ARCHITECTURE[args.model]
    label = args.model

    if args.architecture != architecture and args.architecture != "cnn1d":
        print(
            f"NOTE: --architecture={args.architecture} ignored; "
            f"{label} always uses {architecture}."
        )

    if stage == "develop":
        branches = ["position_aware", "position_normalized"]
    else:
        branches = ["position_normalized", "position_aware"]

    print("=" * 70)
    print(f"PHASE 5 {label} {stage.upper()}")
    print("=" * 70)
    print()

    for i, branch in enumerate(branches, start=1):
        print("=" * 70)
        print(f"{architecture.upper()} {stage.upper()} {i}/{len(branches)}: {branch}")
        print("=" * 70)
        print()

        branch_args = argparse.Namespace(
            manifest=args.manifest,
            binary_dir=args.binary_dir,
            branch=branch,
            architecture=architecture,
            scaler_path=args.scaler_path,
            out_dir=args.out_dir,
            amplitude_normalize=args.amplitude_normalize,
            num_workers=args.num_workers,
            frozen_dir=getattr(args, "frozen_dir", None),
        )

        if stage == "develop":
            run_develop(branch_args)
        else:
            run_test(branch_args)

        print()

    print("=" * 70)
    print(f"PHASE 5 {label} {stage.upper()} COMPLETE")
    print("=" * 70)
    print()


# ============================================================
# CLI
# ============================================================

def _add_common_data_args(p):
    p.add_argument("--manifest", type=str, default="artifacts/phase3/master_manifest.csv")
    p.add_argument("--binary-dir", type=str, default="artifacts/phase3/binary")
    p.add_argument(
        "--branch",
        choices=["position_aware", "position_normalized"],
        required=False,
        default=None,
    )
    p.add_argument("--architecture", choices=["cnn1d", "cnn2d"], default="cnn1d")
    p.add_argument("--scaler-path", type=str, default="artifacts/phase3/scaler.json")
    p.add_argument("--out-dir", type=str, default="artifacts/phase5")
    p.add_argument(
        "--amplitude-normalize",
        action="store_true",
        help=(
            "Tests whether a CNN handles amplitude-scale distance-leakage "
            "better than hand-crafted features."
        ),
    )


def build_parser():
    ap = argparse.ArgumentParser(
        prog=SCRIPT_NAME,
        description="Phase 5 CNN pipeline -- subcommand selects the stage.",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p_smoke = sub.add_parser("smoke", help="Run synthetic-data smoke tests.")
    p_smoke.set_defaults(func=run_smoke)

    p_develop = sub.add_parser("develop", help="Env1 hyperparameter search + freeze best model.")
    _add_common_data_args(p_develop)
    p_develop.add_argument("model", nargs="?", choices=["CNN1", "CNN2"], default=None)
    p_develop.add_argument("--num-workers", type=int, default=2,
                           help="DataLoader worker processes. Try 2-4 if training is slow.")
    p_develop.set_defaults(func=run_develop)

    p_test = sub.add_parser("test", help="Env2 final test of a frozen model (no retraining).")
    _add_common_data_args(p_test)
    p_test.add_argument("model", nargs="?", choices=["CNN1", "CNN2"], default=None)
    p_test.add_argument("--num-workers", type=int, default=2,
                        help="DataLoader worker processes.")
    p_test.add_argument("--frozen-dir", type=str, default=None)
    p_test.set_defaults(func=run_test)

    p_all = sub.add_parser("all", help="Run 'develop' immediately followed by 'test'.")
    _add_common_data_args(p_all)
    p_all.add_argument("--num-workers", type=int, default=2)
    p_all.add_argument("--frozen-dir", type=str, default=None)
    p_all.set_defaults(func=run_all)

    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()

    if args.command in {"develop", "test"} and args.model in SHORTCUT_ARCHITECTURE:
        _run_shortcut(args, args.command)
        return

    # FIX: "all" also requires --branch (it previously crashed later).
    if args.command in {"develop", "test", "all"} and args.branch is None:
        ap.error(f"{args.command} requires --branch unless a CNN1/CNN2 shortcut is used.")

    args.func(args)


if __name__ == "__main__":
    main()