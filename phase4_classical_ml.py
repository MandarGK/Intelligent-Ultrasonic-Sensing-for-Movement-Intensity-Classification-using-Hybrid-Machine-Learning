"""
Phase 4 — Classical ML (single-file consolidated version).

Everything previously split across core/{evaluation,feature_config,
results_store,shape_features,signal_features}.py, core/models/{base,
linear_model,random_forest,xgboost_model}.py, and the four experiment
scripts now lives in this one file.

Phase 4 has four distinct stages:

  1. build-aligned
       Build the position-normalized feature table from Phase 3's
       aligned/onset-cropped waveforms.

  2. build-shape
       Build the position- AND amplitude-normalized feature table.

  3. train
       Env1 development: tune RF/XGB/Linear, select via Env1 val,
       freeze the winners.

  4. test
       Env2 final test: run frozen models once, no retraining/retuning,
       save predictions + Section 13 aggregate tables.

Backup behavior
---------------

The first stage, build-aligned, supports:

    --clean

When --clean is supplied and artifacts/phase4 already exists:

    1. A timestamped backup is created.
    2. The old phase4 directory is deleted ONLY after the backup succeeds.
    3. A fresh phase4 directory is then created.

pigz is used when available:

    artifacts/phase4_backup_YYYYMMDD_HHMMSS.tar.gz

Otherwise:

    artifacts/phase4_backup_YYYYMMDD_HHMMSS.tar

IMPORTANT:
Do NOT use --clean on train/test after you've created the feature tables,
because those stages need to preserve the files and frozen models created
by previous stages.

Typical workflow
----------------

1. Start a completely fresh Phase 4 run:

    python phase4_classical_ml.py build-aligned \
        --manifest artifacts/phase3/master_manifest.csv \
        --binary-dir artifacts/phase3/binary \
        --out artifacts/phase4/aligned_feature_manifest.csv \
        --clean

2. Build shape-normalized features:

    python phase4_classical_ml.py build-shape \
        --manifest artifacts/phase3/master_manifest.csv \
        --binary-dir artifacts/phase3/binary \
        --out artifacts/phase4/shape_feature_manifest.csv

3. Train:

    python phase4_classical_ml.py train \
        --manifest artifacts/phase4/aligned_feature_manifest.csv \
        --feature-set aligned \
        --out-dir artifacts/phase4

4. Test:

    python phase4_classical_ml.py test \
        --manifest artifacts/phase4/aligned_feature_manifest.csv \
        --feature-set aligned \
        --frozen-dir artifacts/phase4/frozen_models \
        --out-dir artifacts/phase4

Feature-set choices
-------------------

    full
        Position-aware features from the original master manifest.

    aligned
        Position-normalized features computed from the Phase 3
        onset-aligned waveform.

    shape_invariant
        Position- AND amplitude-normalized features.

    shape_invariant_v2
        Same shape-normalized table, dropping band_energy_fraction.

Requires
--------

    numpy
    pandas
    scikit-learn
    joblib

xgboost is optional. If it is not installed, the XGBoost model is skipped.
"""

import argparse
import json
import shutil
import subprocess
from datetime import datetime
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# =============================================================================
# OPTIONAL XGBOOST
# =============================================================================

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True

except ImportError:
    XGBOOST_AVAILABLE = False


# =============================================================================
# CONSTANTS
# =============================================================================

FS_HZ = 1_953_125.0

CLASS_TO_INT = {
    "NON_HUMAN": 0,
    "HUMAN": 1,
}

CARRIER_BAND_HZ = (
    20_000.0,
    60_000.0,
)

MIN_PEAK_AMPLITUDE = 1e-6


# =============================================================================
# SECTION 0 — PHASE 4 BACKUP LOGIC
# =============================================================================

def backup_existing_phase4(out_dir: Path) -> Path | None:
    """
    Back up the existing Phase 4 output directory.

    Behavior:

      - If phase4 does not exist:
          do nothing.

      - If phase4 exists but is empty:
          remove it.

      - If phase4 contains files:
          1. Create a timestamped backup.
          2. ONLY after the backup succeeds, delete phase4.

      - Uses pigz -1 if available.

      - Falls back to an uncompressed TAR if pigz is unavailable.

    The original phase4 directory is NEVER deleted before the backup
    succeeds.
    """

    # ------------------------------------------------------------------
    # Nothing exists.
    # ------------------------------------------------------------------

    if not out_dir.exists():
        return None

    # ------------------------------------------------------------------
    # Make sure the path is a directory.
    # ------------------------------------------------------------------

    if not out_dir.is_dir():
        raise RuntimeError(
            f"Expected Phase 4 output path to be a directory: {out_dir}"
        )

    # ------------------------------------------------------------------
    # Check whether the directory contains anything.
    # ------------------------------------------------------------------

    has_content = any(out_dir.iterdir())

    # ------------------------------------------------------------------
    # Empty directory.
    #
    # There is nothing to back up, so simply remove it.
    # ------------------------------------------------------------------

    if not has_content:
        shutil.rmtree(out_dir)

        print(
            f"Removed empty Phase 4 output directory: {out_dir}"
        )

        return None

    # ------------------------------------------------------------------
    # Existing Phase 4 directory contains files.
    # Create timestamped backup.
    # ------------------------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    # Check whether pigz is installed.
    pigz = shutil.which("pigz")

    if pigz:
        extension = ".tar.gz"
    else:
        extension = ".tar"

    backup_path = (
        out_dir.parent
        / f"{out_dir.name}_backup_{timestamp}{extension}"
    )

    # ------------------------------------------------------------------
    # Avoid overwriting a backup created in the same second.
    # ------------------------------------------------------------------

    counter = 1

    while backup_path.exists():

        backup_path = (
            out_dir.parent
            / f"{out_dir.name}_backup_{timestamp}_{counter}{extension}"
        )

        counter += 1

    print()
    print("=" * 70)
    print("PHASE 4 BACKUP")
    print("=" * 70)

    print(
        f"Existing Phase 4 output found:\n"
        f"  {out_dir}"
    )

    # ------------------------------------------------------------------
    # Backup using pigz.
    # ------------------------------------------------------------------

    if pigz:

        print(
            f"pigz found: {pigz}"
        )

        print(
            f"Creating compressed backup:\n"
            f"  {backup_path}"
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

                # Parent process no longer needs its copy
                # of the pipe.
                tar_proc.stdout.close()

                pigz_return = pigz_proc.wait()
                tar_return = tar_proc.wait()

            # Both processes must succeed.
            if pigz_return != 0 or tar_return != 0:

                raise RuntimeError(
                    "Phase 4 backup failed "
                    f"(tar={tar_return}, pigz={pigz_return})"
                )

        except Exception:

            # Remove incomplete backup.
            backup_path.unlink(
                missing_ok=True
            )

            # Make sure tar is not left running.
            if tar_proc.poll() is None:

                tar_proc.kill()
                tar_proc.wait()

            raise

    # ------------------------------------------------------------------
    # Backup without pigz.
    # ------------------------------------------------------------------

    else:

        print(
            "pigz not found."
        )

        print(
            "Creating uncompressed TAR backup:"
        )

        print(
            f"  {backup_path}"
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

        if result.returncode != 0:

            # Remove incomplete backup.
            backup_path.unlink(
                missing_ok=True
            )

            raise RuntimeError(
                "Phase 4 backup failed "
                f"(tar={result.returncode})"
            )

    # ------------------------------------------------------------------
    # Backup succeeded.
    # ------------------------------------------------------------------

    print()
    print(
        f"Phase 4 backup complete:"
    )

    print(
        f"  {backup_path}"
    )

    # ------------------------------------------------------------------
    # IMPORTANT:
    #
    # Delete the old Phase 4 directory ONLY after the backup succeeds.
    # ------------------------------------------------------------------

    try:

        shutil.rmtree(out_dir)

    except Exception as exc:

        raise RuntimeError(
            f"Backup succeeded at {backup_path}, but failed to remove "
            f"old Phase 4 directory {out_dir}: {exc}"
        ) from exc

    print(
        f"Removed old Phase 4 output directory:"
    )

    print(
        f"  {out_dir}"
    )

    print("=" * 70)
    print()

    return backup_path


# =============================================================================
# SECTION 1 — signal_features.py
# =============================================================================

def compute_time_domain_features(
    samples: np.ndarray,
) -> dict:

    x = samples.astype(np.float64)

    mean = x.mean()
    std = x.std()
    rms = np.sqrt(np.mean(x ** 2))
    peak = np.max(np.abs(x))
    peak_to_peak = x.max() - x.min()

    crest_factor = (
        peak / rms
        if rms > 0
        else 0.0
    )

    return {
        "mean": mean,
        "std": std,
        "rms": rms,
        "peak": peak,
        "peak_to_peak": peak_to_peak,
        "crest_factor": crest_factor,
    }


def compute_frequency_domain_features(
    samples: np.ndarray,
    fs_hz: float = FS_HZ,
    band_hz=CARRIER_BAND_HZ,
) -> dict:

    x = samples.astype(np.float64)

    x = x - x.mean()

    n = len(x)

    spectrum = np.abs(
        np.fft.rfft(x)
    )

    freqs = np.fft.rfftfreq(
        n,
        d=1.0 / fs_hz,
    )

    band_mask = (
        (freqs >= band_hz[0])
        &
        (freqs <= band_hz[1])
    )

    if not band_mask.any():

        return {
            "dominant_frequency_hz": 0.0,
            "band_energy": 0.0,
            "total_energy": float(
                np.sum(spectrum ** 2)
            ),
            "band_energy_fraction": 0.0,
        }

    band_spectrum = spectrum[
        band_mask
    ]

    band_freqs = freqs[
        band_mask
    ]

    dominant_idx = np.argmax(
        band_spectrum
    )

    dominant_frequency_hz = (
        band_freqs[dominant_idx]
    )

    band_energy = np.sum(
        band_spectrum ** 2
    )

    total_energy = np.sum(
        spectrum ** 2
    )

    return {
        "dominant_frequency_hz": float(
            dominant_frequency_hz
        ),
        "band_energy": float(
            band_energy
        ),
        "total_energy": float(
            total_energy
        ),
        "band_energy_fraction": float(
            band_energy / total_energy
        )
        if total_energy > 0
        else 0.0,
    }


def compute_all_features(
    samples: np.ndarray,
    fs_hz: float = FS_HZ,
) -> dict:

    feats = {}

    feats.update(
        compute_time_domain_features(samples)
    )

    feats.update(
        compute_frequency_domain_features(
            samples,
            fs_hz,
        )
    )

    return feats


# =============================================================================
# SECTION 2 — shape_features.py
# =============================================================================

def compute_scale_invariant_features(
    samples: np.ndarray,
    fs_hz: float = FS_HZ,
) -> dict:

    x = samples.astype(np.float64)

    x = x - x.mean()

    peak = np.max(
        np.abs(x)
    )

    # Noise-floor recording.
    if peak < MIN_PEAK_AMPLITUDE:

        return {
            "shape_features_valid": False,
            "scale_factor_peak": float(peak),
        }

    std_over_peak = (
        x.std() / peak
    )

    peak_to_peak_over_peak = (
        (x.max() - x.min()) / peak
    )

    rms = np.sqrt(
        np.mean(x ** 2)
    )

    crest_factor = (
        peak / rms
        if rms > 0
        else 0.0
    )

    freq_feats = compute_frequency_domain_features(
        x,
        fs_hz=fs_hz,
        band_hz=CARRIER_BAND_HZ,
    )

    return {
        "shape_features_valid": True,
        "scale_factor_peak": float(peak),
        "std_over_peak": float(
            std_over_peak
        ),
        "peak_to_peak_over_peak": float(
            peak_to_peak_over_peak
        ),
        "crest_factor": float(
            crest_factor
        ),
        "dominant_frequency_hz": float(
            freq_feats["dominant_frequency_hz"]
        ),
        "band_energy_fraction": float(
            freq_feats.get(
                "band_energy_fraction",
                0.0,
            )
        ),
    }


# =============================================================================
# SECTION 3 — feature_config.py
# =============================================================================

FEATURE_COLUMNS = [
    "mean",
    "std",
    "rms",
    "peak",
    "peak_to_peak",
    "crest_factor",
    "dominant_frequency_hz",
    "band_energy",
    "band_energy_fraction",
]

SHAPE_FEATURE_COLUMNS = [
    "std_over_peak",
    "peak_to_peak_over_peak",
    "crest_factor",
    "dominant_frequency_hz",
    "band_energy_fraction",
]

SHAPE_FEATURE_COLUMNS_V2 = [
    "std_over_peak",
    "peak_to_peak_over_peak",
    "crest_factor",
    "dominant_frequency_hz",
]

FEATURE_SETS = {
    "full": FEATURE_COLUMNS,

    "aligned": FEATURE_COLUMNS,

    "shape_invariant": SHAPE_FEATURE_COLUMNS,

    "shape_invariant_v2": SHAPE_FEATURE_COLUMNS_V2,
}


# =============================================================================
# SECTION 4 — MODELS
# =============================================================================

class BaseModel(ABC):

    @abstractmethod
    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        **hp,
    ):
        ...

    @abstractmethod
    def predict(self, X):
        ...

    @abstractmethod
    def predict_proba(self, X):
        ...

    @abstractmethod
    def save(self, path):
        ...

    @abstractmethod
    def load(self, path):
        ...

    @abstractmethod
    def get_params(self):
        ...


class LinearModel(BaseModel):
    """
    Logistic Regression model.
    """

    def __init__(self, **hp):

        self.hp = hp

        self.model = LogisticRegression(
            max_iter=2000,
            random_state=42,
            **hp,
        )

        self.feature_scaler = StandardScaler()

    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        **hp,
    ):

        if hp:
            self.hp.update(hp)

        self.model = LogisticRegression(
            max_iter=2000,
            random_state=42,
            **self.hp,
        )

        X_train_scaled = (
            self.feature_scaler.fit_transform(
                X_train
            )
        )

        self.model.fit(
            X_train_scaled,
            y_train,
        )

        return self

    def predict(self, X):

        return self.model.predict(
            self.feature_scaler.transform(X)
        )

    def predict_proba(self, X):

        return self.model.predict_proba(
            self.feature_scaler.transform(X)
        )[:, 1]

    def save(self, path):

        joblib.dump(
            {
                "model": self.model,
                "hp": self.hp,
                "feature_scaler": self.feature_scaler,
            },
            path,
        )

    def load(self, path):

        d = joblib.load(path)

        self.model = d["model"]
        self.hp = d["hp"]
        self.feature_scaler = d[
            "feature_scaler"
        ]

        return self

    def get_params(self):

        return dict(self.hp)


class RandomForestModel(BaseModel):

    def __init__(self, **hp):

        self.hp = hp

        self.model = RandomForestClassifier(
            random_state=42,
            **hp,
        )

    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        **hp,
    ):

        if hp:
            self.hp.update(hp)

        self.model = RandomForestClassifier(
            random_state=42,
            **self.hp,
        )

        self.model.fit(
            X_train,
            y_train,
        )

        return self

    def predict(self, X):

        return self.model.predict(X)

    def predict_proba(self, X):

        return self.model.predict_proba(X)[:, 1]

    def save(self, path):

        joblib.dump(
            {
                "model": self.model,
                "hp": self.hp,
            },
            path,
        )

    def load(self, path):

        d = joblib.load(path)

        self.model = d["model"]
        self.hp = d["hp"]

        return self

    def get_params(self):

        return dict(self.hp)


if XGBOOST_AVAILABLE:

    class XGBoostModel(BaseModel):

        def __init__(self, **hp):

            self.hp = hp

            self.model = XGBClassifier(
                random_state=42,
                eval_metric="logloss",
                **hp,
            )

        def fit(
            self,
            X_train,
            y_train,
            X_val=None,
            y_val=None,
            **hp,
        ):

            if hp:
                self.hp.update(hp)

            self.model = XGBClassifier(
                random_state=42,
                eval_metric="logloss",
                **self.hp,
            )

            if (
                X_val is not None
                and y_val is not None
            ):

                self.model.fit(
                    X_train,
                    y_train,
                    eval_set=[
                        (X_val, y_val)
                    ],
                    verbose=False,
                )

            else:

                self.model.fit(
                    X_train,
                    y_train,
                )

            return self

        def predict(self, X):

            return self.model.predict(X)

        def predict_proba(self, X):

            return self.model.predict_proba(
                X
            )[:, 1]

        def save(self, path):

            joblib.dump(
                {
                    "model": self.model,
                    "hp": self.hp,
                },
                path,
            )

        def load(self, path):

            d = joblib.load(path)

            self.model = d["model"]
            self.hp = d["hp"]

            return self

        def get_params(self):

            return dict(self.hp)


MODEL_CLASSES = {
    "random_forest": RandomForestModel,
    "linear": LinearModel,
}

if XGBOOST_AVAILABLE:

    MODEL_CLASSES[
        "xgboost"
    ] = XGBoostModel


HYPERPARAM_GRIDS = {

    "random_forest": [
        {
            "n_estimators": 100,
            "max_depth": None,
        },
        {
            "n_estimators": 200,
            "max_depth": 10,
        },
        {
            "n_estimators": 200,
            "max_depth": None,
        },
        {
            "n_estimators": 300,
            "max_depth": 20,
        },
    ],

    "xgboost": [
        {
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.1,
        },
        {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.05,
        },
        {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.1,
        },
    ],

    "linear": [
        {"C": 0.1},
        {"C": 1.0},
        {"C": 10.0},
    ],
}


# =============================================================================
# SECTION 5 — EVALUATION + RESULTS
# =============================================================================

def compute_metrics_group(
    y_true,
    y_pred,
    y_prob=None,
):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
    }

    if (
        y_prob is not None
        and len(np.unique(y_true)) > 1
    ):

        metrics["roc_auc"] = roc_auc_score(
            y_true,
            y_prob,
        )

    else:

        metrics["roc_auc"] = None

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    if cm.size == 4:

        tn, fp, fn, tp = cm.ravel()

    else:

        tn, fp, fn, tp = (
            None,
            None,
            None,
            None,
        )

    metrics.update(
        {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        }
    )

    return metrics


def build_predictions_table(
    meta_df: pd.DataFrame,
    y_true,
    y_pred,
    y_prob,
    model_name: str,
) -> pd.DataFrame:

    out = meta_df[
        [
            "environment",
            "condition",
            "class_label",
            "subject",
            "activity",
            "source_file",
            "row_index",
            "recording_id",
        ]
    ].copy()

    out["model"] = model_name
    out["y_true"] = y_true
    out["y_pred"] = y_pred
    out["y_prob"] = y_prob

    return out


def aggregate_table(
    predictions_df: pd.DataFrame,
    group_cols: list,
) -> pd.DataFrame:

    """
    Computes TP/TN/FP/FN and standard metrics for any slice.
    """

    rows = []

    for keys, group in predictions_df.groupby(
        group_cols
    ):

        if not isinstance(keys, tuple):
            keys = (keys,)

        metrics = compute_metrics_group(
            group["y_true"],
            group["y_pred"],
            group["y_prob"],
        )

        row = dict(
            zip(
                group_cols,
                keys,
            )
        )

        row.update(metrics)

        row["n"] = len(group)

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# SECTION 6 — BUILD ALIGNED FEATURE TABLE
# =============================================================================

ALIGNED_METADATA_COLUMNS = [
    "source_file",
    "row_index",
    "environment",
    "condition",
    "class_label",
    "subject",
    "activity",
    "subject_group",
    "recording_id",
    "block_id",
    "split",
]


def build_aligned_feature_table(
    master_manifest_path,
    binary_dir,
    verbose=True,
):

    manifest = pd.read_csv(
        master_manifest_path,
        low_memory=False,
    )

    valid = manifest[
        manifest["alignment_ok"] == True
    ].copy()

    if verbose:

        n_total = len(manifest)
        n_valid = len(valid)

        print(
            f"Using {n_valid}/{n_total} recordings "
            f"({n_valid / n_total:.1%}) with valid alignment."
        )

    binary_dir = Path(binary_dir)

    rows = []

    memmap_cache = {}

    for i, (_, row) in enumerate(
        valid.iterrows()
    ):

        stem = Path(
            row["source_file"]
        ).stem

        if stem not in memmap_cache:

            memmap_cache[stem] = np.load(
                binary_dir
                / f"{stem}_aligned.npy",
                mmap_mode="r",
            )

        mm = memmap_cache[stem]

        waveform = np.array(
            mm[int(row["row_index"])]
        )

        if np.isnan(waveform).any():

            raise ValueError(
                f"Row {row['recording_id']} is marked "
                "alignment_ok=True but its aligned waveform "
                "contains NaN -- inconsistent Phase 3 output."
            )

        feats = compute_all_features(
            waveform,
            fs_hz=FS_HZ,
        )

        out_row = {
            col: row[col]
            for col in ALIGNED_METADATA_COLUMNS
            if col in row
        }

        out_row.update(feats)

        rows.append(out_row)

        if (
            verbose
            and (i + 1) % 5000 == 0
        ):

            print(
                f"  ... {i + 1}/{len(valid)} processed",
                end="\r",
                flush=True,
            )

    if verbose:

        print(
            f"  ... {len(rows)}/{len(valid)} processed"
            + " " * 10
        )

    return pd.DataFrame(rows)


# =============================================================================
# SECTION 7 — BUILD SHAPE-NORMALIZED FEATURE TABLE
# =============================================================================

def build_shape_feature_table(
    master_manifest_path,
    binary_dir,
    verbose=True,
):

    manifest = pd.read_csv(
        master_manifest_path,
        low_memory=False,
    )

    valid = manifest[
        manifest["alignment_ok"] == True
    ].copy()

    if verbose:

        print(
            f"Alignment-valid: "
            f"{len(valid)}/{len(manifest)} recordings."
        )

    binary_dir = Path(binary_dir)

    rows = []

    memmap_cache = {}

    n_shape_invalid = 0

    for i, (_, row) in enumerate(
        valid.iterrows()
    ):

        stem = Path(
            row["source_file"]
        ).stem

        if stem not in memmap_cache:

            memmap_cache[stem] = np.load(
                binary_dir
                / f"{stem}_aligned.npy",
                mmap_mode="r",
            )

        mm = memmap_cache[stem]

        waveform = np.array(
            mm[int(row["row_index"])]
        )

        if np.isnan(waveform).any():

            raise ValueError(
                f"Row {row['recording_id']} is marked "
                "alignment_ok=True but its aligned waveform "
                "contains NaN -- inconsistent Phase 3 output."
            )

        feats = compute_scale_invariant_features(
            waveform,
            fs_hz=FS_HZ,
        )

        if not feats[
            "shape_features_valid"
        ]:

            n_shape_invalid += 1

            continue

        out_row = {
            col: row[col]
            for col in ALIGNED_METADATA_COLUMNS
            if col in row
        }

        out_row.update(feats)

        rows.append(out_row)

        if (
            verbose
            and (i + 1) % 5000 == 0
        ):

            print(
                f"  ... {i + 1}/{len(valid)} processed",
                end="\r",
                flush=True,
            )

    if verbose:

        print(
            f"  ... {len(valid)}/{len(valid)} processed"
            + " " * 10
        )

    if n_shape_invalid:

        print(
            f"NOTE: {n_shape_invalid} rows excluded -- "
            f"peak amplitude at noise floor "
            f"(< {MIN_PEAK_AMPLITUDE})"
        )

    return pd.DataFrame(rows)


# =============================================================================
# SECTION 8 — ENV1 DEVELOPMENT
# =============================================================================

def load_split(
    manifest_path,
    feature_columns,
):

    df = pd.read_csv(
        manifest_path,
        low_memory=False,
    )

    df = df[
        df["environment"] == "Env1"
    ].copy()

    df["y"] = df[
        "class_label"
    ].map(CLASS_TO_INT)

    missing_feat = df[
        feature_columns
    ].isna().any(axis=1)

    if missing_feat.any():

        print(
            f"WARNING: dropping "
            f"{missing_feat.sum()} Env1 rows "
            "with missing feature values"
        )

        df = df[
            ~missing_feat
        ]

    train = df[
        df["split"] == "train"
    ]

    val = df[
        df["split"] == "val"
    ]

    return train, val


def tune_model(
    model_key,
    train,
    val,
    feature_columns,
):

    X_train = train[
        feature_columns
    ].values

    y_train = train[
        "y"
    ].values

    X_val = val[
        feature_columns
    ].values

    y_val = val[
        "y"
    ].values

    candidates = HYPERPARAM_GRIDS[
        model_key
    ]

    log = []

    best = None

    for hp in candidates:

        model = MODEL_CLASSES[
            model_key
        ](**hp)

        model.fit(
            X_train,
            y_train,
            X_val,
            y_val,
        )

        val_pred = model.predict(
            X_val
        )

        val_prob = model.predict_proba(
            X_val
        )

        metrics = compute_metrics_group(
            y_val,
            val_pred,
            val_prob,
        )

        log.append(
            {
                "hp": hp,
                **metrics,
            }
        )

        print(
            f"    {model_key} "
            f"hp={hp} -> "
            f"val_f1={metrics['f1']:.4f} "
            f"val_acc={metrics['accuracy']:.4f}"
        )

        if (
            best is None
            or metrics["f1"]
            > best["metrics"]["f1"]
        ):

            best = {
                "hp": hp,
                "model": model,
                "metrics": metrics,
            }

    return best, log


def cmd_train(args):

    feature_columns = FEATURE_SETS[
        args.feature_set
    ]

    print(
        f"Feature set: '{args.feature_set}' "
        f"-> columns: {feature_columns}\n"
    )

    out_dir = Path(
        args.out_dir
    )

    # ------------------------------------------------------------------
    # IMPORTANT:
    #
    # We intentionally DO NOT clean/back up phase4 here.
    #
    # The phase4 directory already contains the feature manifests
    # created by build-aligned/build-shape, and deleting it here would
    # destroy those files.
    # ------------------------------------------------------------------

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    frozen_dir = (
        out_dir
        / "frozen_models"
    )

    frozen_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not XGBOOST_AVAILABLE:

        print(
            "NOTE: xgboost not installed -- "
            "skipping the XGBoost model.\n"
        )

        print(
            "Install it with:"
        )

        print(
            "  pip install xgboost\n"
        )

    print(
        "Loading Env1 train/val split..."
    )

    train, val = load_split(
        args.manifest,
        feature_columns,
    )

    print(
        f"  train={len(train)} "
        f" val={len(val)}"
    )

    print(
        "  train class balance: "
        f"{train['y'].value_counts().to_dict()}"
    )

    print(
        "  val class balance:   "
        f"{val['y'].value_counts().to_dict()}\n"
    )

    all_logs = {}

    summary_rows = []

    for model_key in MODEL_CLASSES:

        print(
            f"Tuning {model_key}..."
        )

        best, log = tune_model(
            model_key,
            train,
            val,
            feature_columns,
        )

        all_logs[
            model_key
        ] = log

        frozen_path = (
            frozen_dir
            / f"{model_key}.joblib"
        )

        best["model"].save(
            frozen_path
        )

        print(
            f"  -> FROZEN best {model_key}: "
            f"hp={best['hp']} "
            f"val_f1={best['metrics']['f1']:.4f} "
            f"saved to {frozen_path}\n"
        )

        summary_rows.append(
            {
                "model": model_key,
                "feature_set": args.feature_set,
                "best_hp": json.dumps(
                    best["hp"]
                ),
                **{
                    k: v
                    for k, v
                    in best["metrics"].items()
                },
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        out_dir
        / "env1_val_model_selection_summary.csv",
        index=False,
    )

    with open(
        out_dir
        / "hyperparameter_search_log.json",
        "w",
    ) as f:

        json.dump(
            all_logs,
            f,
            indent=2,
            default=str,
        )

    print(
        "Model selection summary:"
    )

    print(
        summary[
            [
                "model",
                "best_hp",
                "accuracy",
                "f1",
                "roc_auc",
            ]
        ]
    )

    print(
        f"\nSaved: "
        f"{out_dir / 'env1_val_model_selection_summary.csv'}"
    )

    print(
        "Saved: "
        f"{out_dir / 'hyperparameter_search_log.json'}"
    )

    print(
        f"Frozen models in: {frozen_dir}/"
    )

    print(
        "\nPhase 4 (Env1 development) complete. "
        "Models are FROZEN -- ready for Env2 test."
    )


# =============================================================================
# SECTION 9 — ENV2 FINAL TEST
# =============================================================================

def cmd_test(args):

    feature_columns = FEATURE_SETS[
        args.feature_set
    ]

    print(
        f"Feature set: '{args.feature_set}' "
        f"-> columns: {feature_columns}\n"
    )

    out_dir = Path(
        args.out_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    frozen_dir = Path(
        args.frozen_dir
    )

    df = pd.read_csv(
        args.manifest,
        low_memory=False,
    )

    test = df[
        (df["environment"] == "Env2")
        &
        (df["split"] == "test")
    ].copy()

    test["y"] = test[
        "class_label"
    ].map(CLASS_TO_INT)

    missing_feat = test[
        feature_columns
    ].isna().any(axis=1)

    if missing_feat.any():

        print(
            f"WARNING: dropping "
            f"{missing_feat.sum()} Env2 rows "
            "with missing feature values"
        )

        test = test[
            ~missing_feat
        ]

    print(
        f"Env2 test set: {len(test)} recordings\n"
    )

    X_test = test[
        feature_columns
    ].values

    y_test = test[
        "y"
    ].values

    all_predictions = []

    for model_key in MODEL_CLASSES:

        frozen_path = (
            frozen_dir
            / f"{model_key}.joblib"
        )

        if not frozen_path.exists():

            print(
                f"SKIPPING {model_key}: "
                f"no frozen model found at "
                f"{frozen_path} "
                "(run train first)"
            )

            continue

        print(
            f"Running frozen {model_key} "
            "on Env2 test "
            "(once, no retraining)..."
        )

        model = MODEL_CLASSES[
            model_key
        ]()

        model.load(
            frozen_path
        )

        y_pred = model.predict(
            X_test
        )

        y_prob = model.predict_proba(
            X_test
        )

        preds_table = build_predictions_table(
            test,
            y_test,
            y_pred,
            y_prob,
            model_name=model_key,
        )

        all_predictions.append(
            preds_table
        )

    if not all_predictions:

        print(
            "\nNo frozen models found -- "
            "nothing to evaluate. "
            "Run train first."
        )

        return

    predictions = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    predictions.to_csv(
        out_dir
        / "env2_test_predictions.csv",
        index=False,
    )

    print(
        f"\nSaved raw predictions: "
        f"{out_dir / 'env2_test_predictions.csv'}"
    )

    # ------------------------------------------------------------------
    # Section 13 aggregate tables.
    # ------------------------------------------------------------------

    aggregate_specs = [
        (
            ["model", "environment"],
            "agg_model_x_environment.csv",
        ),
        (
            ["model", "condition"],
            "agg_model_x_condition.csv",
        ),
        (
            ["model", "subject"],
            "agg_model_x_subject.csv",
        ),
        (
            ["model", "activity"],
            "agg_model_x_activity.csv",
        ),
    ]

    for group_cols, fname in aggregate_specs:

        agg = aggregate_table(
            predictions,
            group_cols,
        )

        agg.to_csv(
            out_dir / fname,
            index=False,
        )

        print(
            f"\n{' x '.join(group_cols)}:"
        )

        print(
            agg.to_string(
                index=False
            )
        )

    print(
        f"\nAll aggregate tables saved to "
        f"{out_dir}/"
    )

    print(
        "\nPhase 4 (Env2 final test) complete."
    )


# =============================================================================
# SECTION 10 — BUILD-ALIGNED CLI
# =============================================================================

def cmd_build_aligned(args):

    out_path = Path(
        args.out
    )

    # ---------------------------------------------------------------
    # --clean is intended for starting a completely new Phase 4 run.
    #
    # This must happen BEFORE creating the new output directory/file.
    # ---------------------------------------------------------------

    if args.clean:

        # The clean operation applies to the Phase 4 root directory,
        # not merely to the aligned CSV file.
        #
        # Example:
        #
        #   --out artifacts/phase4/aligned_feature_manifest.csv
        #
        # gives:
        #
        #   artifacts/phase4
        #
        phase4_root = out_path.parent

        print(
            "Starting a CLEAN Phase 4 run."
        )

        backup_existing_phase4(
            phase4_root
        )

    # Create the output directory.
    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    table = build_aligned_feature_table(
        args.manifest,
        args.binary_dir,
    )

    table.to_csv(
        out_path,
        index=False,
    )

    print(
        f"\nSaved: {out_path} "
        f"({len(table)} rows)"
    )

    print(
        "\nSplit summary:"
    )

    print(
        table.groupby(
            [
                "environment",
                "condition",
                "split",
            ],
            dropna=False,
        ).size()
    )


# =============================================================================
# SECTION 11 — BUILD-SHAPE CLI
# =============================================================================

def cmd_build_shape(args):

    out_path = Path(
        args.out
    )

    # No automatic clean here.
    #
    # Phase 4 may already contain:
    #
    #   aligned_feature_manifest.csv
    #
    # and we don't want to delete it when creating:
    #
    #   shape_feature_manifest.csv

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    table = build_shape_feature_table(
        args.manifest,
        args.binary_dir,
    )

    table.to_csv(
        out_path,
        index=False,
    )

    print(
        f"\nSaved: {out_path} "
        f"({len(table)} rows)"
    )

    print(
        "\nSplit summary:"
    )

    print(
        table.groupby(
            [
                "environment",
                "condition",
                "split",
            ],
            dropna=False,
        ).size()
    )


# =============================================================================
# SECTION 11.5 — BUILD / ALL SHORTCUT COMMANDS
# =============================================================================

def cmd_build(args):
    """
    Shortcut for:

        build-aligned
        build-shape

    Default paths:

        manifest:
            artifacts/phase3/master_manifest.csv

        binary-dir:
            artifacts/phase3/binary

        aligned output:
            artifacts/phase4/aligned_feature_manifest.csv

        shape output:
            artifacts/phase4/shape_feature_manifest.csv

    If arguments are supplied, those values are used instead.
    """

    print("=" * 70)
    print("PHASE 4 BUILD")
    print("=" * 70)
    print()
    print("Step 1/2: Building aligned feature table...")
    print()

    aligned_args = argparse.Namespace(
        manifest=args.manifest,
        binary_dir=args.binary_dir,
        out=args.aligned_out,
        clean=args.clean,
    )

    cmd_build_aligned(aligned_args)

    print()
    print("=" * 70)
    print("Step 2/2: Building shape-normalized feature table...")
    print("=" * 70)
    print()

    shape_args = argparse.Namespace(
        manifest=args.manifest,
        binary_dir=args.binary_dir,
        out=args.shape_out,
    )

    cmd_build_shape(shape_args)

    print()
    print("=" * 70)
    print("PHASE 4 BUILD COMPLETE")
    print("=" * 70)
    print()
    print(f"Aligned feature manifest:")
    print(f"  {args.aligned_out}")
    print()
    print(f"Shape feature manifest:")
    print(f"  {args.shape_out}")
    print()


def cmd_all(args):
    """
    Shortcut for:

        train
        test

    Default paths:

        manifest:
            artifacts/phase3/master_manifest.csv

        feature-set:
            full

        out-dir:
            artifacts/phase4/full

        frozen-dir:
            artifacts/phase4/full/frozen_models

    If arguments are supplied, those values are used instead.
    """

    print("=" * 70)
    print("PHASE 4 ALL")
    print("=" * 70)
    print()
    print("Step 1/2: Training models...")
    print()

    train_args = argparse.Namespace(
        manifest=args.manifest,
        feature_set=args.feature_set,
        out_dir=args.out_dir,
    )

    cmd_train(train_args)

    print()
    print("=" * 70)
    print("Step 2/2: Testing frozen models...")
    print("=" * 70)
    print()

    test_args = argparse.Namespace(
        manifest=args.manifest,
        feature_set=args.feature_set,
        frozen_dir=args.frozen_dir,
        out_dir=args.out_dir,
    )

    cmd_test(test_args)

    print()
    print("=" * 70)
    print("PHASE 4 ALL COMPLETE")
    print("=" * 70)
    print()



# =============================================================================
# SECTION 12 — MAIN / CLI
# =============================================================================




def main():

    ap = argparse.ArgumentParser(
        description=(
            "Phase 4 classical ML "
            "(single-file, subcommands)"
        )
    )

    sub = ap.add_subparsers(
        dest="command",
        required=True,
    )
    
    
    # ------------------------------------------------------------------
    # BUILD
    # ------------------------------------------------------------------
    p_build = sub.add_parser(
        "build",
        help=(
            "Build both aligned and shape-normalized "
            "feature tables"
        ),
    )

    p_build.add_argument(
        "--manifest",
        type=str,
        default="artifacts/phase3/master_manifest.csv",
        help=(
            "Path to Phase 3's master_manifest.csv"
        ),
    )

    p_build.add_argument(
        "--binary-dir",
        type=str,
        default="artifacts/phase3/binary",
        help=(
            "Path to Phase 3's binary/ directory"
        ),
    )

    p_build.add_argument(
        "--aligned-out",
        type=str,
        default=(
            "artifacts/phase4/"
            "aligned_feature_manifest.csv"
        ),
        help=(
            "Output path for the aligned feature manifest"
        ),
    )

    p_build.add_argument(
        "--shape-out",
        type=str,
        default=(
            "artifacts/phase4/"
            "shape_feature_manifest.csv"
        ),
        help=(
            "Output path for the shape feature manifest"
        ),
    )

    p_build.add_argument(
        "--clean",
        action="store_true",
        help=(
            "Backup the existing Phase 4 directory and "
            "delete it before starting the build."
        ),
    )

    p_build.set_defaults(
        func=cmd_build
    )
    
    

    # ------------------------------------------------------------------
    # BUILD ALIGNED
    # ------------------------------------------------------------------

    p_ba = sub.add_parser(
        "build-aligned",
        help=(
            "Build the position-normalized "
            "feature table"
        ),
    )

    p_ba.add_argument(
        "--manifest",
        type=str,
        required=True,
        help=(
            "Path to Phase 3's "
            "master_manifest.csv"
        ),
    )

    p_ba.add_argument(
        "--binary-dir",
        type=str,
        required=True,
        help=(
            "Path to Phase 3's binary/ directory"
        ),
    )

    p_ba.add_argument(
        "--out",
        type=str,
        default=(
            "artifacts/phase4/"
            "aligned_feature_manifest.csv"
        ),
        help=(
            "Output feature manifest path"
        ),
    )

    p_ba.add_argument(
        "--clean",
        action="store_true",
        help=(
            "Backup the existing Phase 4 directory "
            "and delete it before building a fresh "
            "aligned feature table. Use this only "
            "when starting a new Phase 4 run."
        ),
    )

    p_ba.set_defaults(
        func=cmd_build_aligned
    )

    # ------------------------------------------------------------------
    # BUILD SHAPE
    # ------------------------------------------------------------------

    p_bs = sub.add_parser(
        "build-shape",
        help=(
            "Build the position + amplitude-normalized "
            "(shape_invariant) feature table"
        ),
    )

    p_bs.add_argument(
        "--manifest",
        type=str,
        required=True,
        help=(
            "Path to Phase 3's "
            "master_manifest.csv"
        ),
    )

    p_bs.add_argument(
        "--binary-dir",
        type=str,
        required=True,
        help=(
            "Path to Phase 3's binary/ directory"
        ),
    )

    p_bs.add_argument(
        "--out",
        type=str,
        default=(
            "artifacts/phase4/"
            "shape_feature_manifest.csv"
        ),
        help=(
            "Output feature manifest path"
        ),
    )

    p_bs.set_defaults(
        func=cmd_build_shape
    )

    # ------------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------------

    p_tr = sub.add_parser(
        "train",
        help=(
            "Env1 development: "
            "tune + freeze RF/XGB/Linear"
        ),
    )

    p_tr.add_argument(
        "--manifest",
        type=str,
        required=True,
    )

    p_tr.add_argument(
        "--out-dir",
        type=str,
        default="artifacts/phase4",
    )

    p_tr.add_argument(
        "--feature-set",
        type=str,
        default="full",
        choices=list(
            FEATURE_SETS.keys()
        ),
        help=(
            "'full'/'aligned' = "
            "position-aware / position-normalized; "
            "'shape_invariant'/'shape_invariant_v2' = "
            "position- AND amplitude-normalized"
        ),
    )

    p_tr.set_defaults(
        func=cmd_train
    )

        # ------------------------------------------------------------------
    # ALL
    # ------------------------------------------------------------------
    p_all = sub.add_parser(
        "all",
        help=(
            "Train and then test all configured classical ML models"
        ),
    )

    p_all.add_argument(
        "--manifest",
        type=str,
        default="artifacts/phase3/master_manifest.csv",
        help=(
            "Path to the feature manifest / master manifest"
        ),
    )

    p_all.add_argument(
        "--feature-set",
        type=str,
        default="full",
        choices=list(FEATURE_SETS.keys()),
        help=(
            "'full'/'aligned' = position-aware / "
            "position-normalized; "
            "'shape_invariant'/'shape_invariant_v2' = "
            "position- and amplitude-normalized"
        ),
    )

    p_all.add_argument(
        "--out-dir",
        type=str,
        default="artifacts/phase4/full",
        help=(
            "Output directory for training/test results"
        ),
    )

    p_all.add_argument(
        "--frozen-dir",
        type=str,
        default=(
            "artifacts/phase4/"
            "full/frozen_models"
        ),
        help=(
            "Directory containing frozen models"
        ),
    )

    p_all.set_defaults(
        func=cmd_all
    )

    # ------------------------------------------------------------------
    # TEST
    # ------------------------------------------------------------------

    p_te = sub.add_parser(
        "test",
        help=(
            "Env2 final test: "
            "run frozen models once, "
            "no retuning"
        ),
    )

    p_te.add_argument(
        "--manifest",
        type=str,
        required=True,
    )

    p_te.add_argument(
        "--frozen-dir",
        type=str,
        default=(
            "artifacts/phase4/"
            "frozen_models"
        ),
    )

    p_te.add_argument(
        "--out-dir",
        type=str,
        default="artifacts/phase4",
    )

    p_te.add_argument(
        "--feature-set",
        type=str,
        default="full",
        choices=list(
            FEATURE_SETS.keys()
        ),
    )

    p_te.set_defaults(
        func=cmd_test
    )

    # ------------------------------------------------------------------
    # Parse arguments and dispatch.
    # ------------------------------------------------------------------

    args = ap.parse_args()

    args.func(args)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()