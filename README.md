# Intelligent_Ultrasonic_Sensing_for_Movement_Intensity_Classification_using_Hybrid_Machine_Learning — Chapter 7 Code

## Project overview

This folder contains the runnable code for the ultrasonic human-presence/activity
classification experiments.

The pipeline studies how **signal representation, position normalization, model
family, environment, and subject** affect cross-environment generalization.

Two main research tasks are implemented:

1. **Human Presence (RQ1):** HUMAN vs. EmptyRoom / ChairTable.
2. **Standing vs. Reading (RQ6 / Phase 6):** distinguishes two human activities while
   comparing position-aware and position-normalized inputs.

The implementation uses:
- Classical ML: Linear/Logistic Regression, Random Forest, XGBoost
- CNN1D: raw ultrasonic waveform
- CNN2D: band-limited spectrogram
- Position-aware and position-normalized signal branches

## Contents

- `phase3_preprocessing.py` — converts raw CSV recordings into reusable binary
  waveforms, aligned crops, features, manifests, and the Env1 train/validation split.
- `phase4_classical_ml.py` — builds classical feature sets, trains/freezes
  classical models, and evaluates them on Env2.
- `phase5_deep_learning.py` — trains/freezes/tests CNN1D and CNN2D models.
- `rq6_standing_vs_reading_single.py` — runs the Standing-vs-Reading experiment
  (RQ6 / Phase 6).
- `consolidate_results.py` — combines completed experiment outputs into tagged
  result tables.

> **Important:** The ZIP contains the Python source only. The raw ultrasonic CSV
> dataset is not included and must be supplied separately.

## Requirements

Python 3.10+ is recommended.

Install the main dependencies:

```bash
pip install numpy pandas scipy scikit-learn joblib xgboost torch
```

If XGBoost is unavailable, the classical pipeline can still run but XGBoost
results will be skipped.

# Quick commands

Run these from inside the `Chap7` folder.

### Phase 3 — preprocessing

```bash
python phase3_preprocessing.py --data-dir /path/to/Data
```

### Phase 4 — Classical ML

**Build both aligned and shape-normalized feature tables:**

```bash
python phase4_classical_ml.py build
```

**Build only the aligned / position-normalized table:**

```bash
python phase4_classical_ml.py build-aligned --manifest artifacts/phase3/master_manifest.csv --binary-dir artifacts/phase3/binary --out artifacts/phase4/aligned_feature_manifest.csv --clean
```

**Build only the shape-invariant table:**

```bash
python phase4_classical_ml.py build-shape --manifest artifacts/phase3/master_manifest.csv --binary-dir artifacts/phase3/binary --out artifacts/phase4/shape_feature_manifest.csv
```

**Run the full classical pipeline:**

```bash
python phase4_classical_ml.py all
```

**Run the aligned / position-normalized branch:**

```bash
python phase4_classical_ml.py all --manifest artifacts/phase4/aligned_feature_manifest.csv --feature-set aligned --out-dir artifacts/phase4/aligned
```

**Run the shape-invariant branch:**

```bash
python phase4_classical_ml.py all --manifest artifacts/phase4/shape_feature_manifest.csv --feature-set shape_invariant --out-dir artifacts/phase4/shape
```

### Phase 5 — CNNs

**CNN1D, position-normalized:**

```bash
python phase5_deep_learning.py all --architecture cnn1d --branch position_normalized --out-dir artifacts/phase5/position_normalized_cnn1d
```

**CNN1D, position-aware:**

```bash
python phase5_deep_learning.py all --architecture cnn1d --branch position_aware --out-dir artifacts/phase5/position_aware_cnn1d
```

**CNN2D, position-normalized:**

```bash
python phase5_deep_learning.py all --architecture cnn2d --branch position_normalized --out-dir artifacts/phase5/position_normalized_cnn2d
```

**CNN2D, position-aware:**

```bash
python phase5_deep_learning.py all --architecture cnn2d --branch position_aware --out-dir artifacts/phase5/position_aware_cnn2d
```

**Quick smoke test:**

```bash
python phase5_deep_learning.py smoke
```

### Phase 6 — Standing vs. Reading (RQ6)

**Run CNN1D for both position branches:**

```bash
python rq6_standing_vs_reading_single.py all CNN1
```

**Run CNN2D for both position branches:**

```bash
python rq6_standing_vs_reading_single.py all CNN2
```

**Run classical models for both position branches:**

```bash
python rq6_standing_vs_reading_single.py all CLASSICAL
```

### One-page shortcut

The shortest useful sequence is:

```text
Phase 3
python phase3_preprocessing.py --data-dir /path/to/Data

Phase 4
python phase4_classical_ml.py build
python phase4_classical_ml.py all

Phase 5
python phase5_deep_learning.py all --architecture cnn1d --branch position_normalized --out-dir artifacts/phase5/position_normalized_cnn1d
python phase5_deep_learning.py all --architecture cnn1d --branch position_aware --out-dir artifacts/phase5/position_aware_cnn1d
python phase5_deep_learning.py all --architecture cnn2d --branch position_normalized --out-dir artifacts/phase5/position_normalized_cnn2d
python phase5_deep_learning.py all --architecture cnn2d --branch position_aware --out-dir artifacts/phase5/position_aware_cnn2d

Phase 6 / RQ6
python rq6_standing_vs_reading_single.py all CNN1
python rq6_standing_vs_reading_single.py all CNN2
python rq6_standing_vs_reading_single.py all CLASSICAL
```

> **Note:** Phase 5 and Phase 6 `all` commands perform development/model selection
> and then evaluate the frozen model on the Env2 test set. These are full
> experiments and can take substantially longer than the Phase 5 smoke test.

## 1. Preprocess the raw data

The raw-data folder should contain the experiment CSV files using the expected
filename convention, for example:

```text
Room1_EMPTY_5000.csv
Room1_with_Chair_5000.csv
ENV1_HUMAN_P1_5000.csv
ENV1_HUMAN_P2_5000.csv
ENV1_HUMAN_SITTING_P1_5000.csv
ENV1_HUMAN_SITTING_P2_5000.csv
ENV2_HUMAN_P3_5000.csv
ENV2_HUMAN_P4_5000.csv
ENV2_HUMAN_SITTING_P3_5000.csv
ENV2_HUMAN_SITTING_P4_5000.csv
...
```

Run:

```bash
python phase3_preprocessing.py --data-dir /path/to/Data
```

Main outputs are written to:

```text
artifacts/phase3/
├── binary/
├── master_manifest.csv
├── bonus_pd_manifest.csv
├── env1_block_split_log.csv
└── scaler.json
```

## 2. Run classical ML

Build the feature tables:

```bash
python phase4_classical_ml.py build
```

This is equivalent to building both the aligned and shape-normalized feature
tables in one command.

You can also build them separately:

```bash
python phase4_classical_ml.py build-aligned   --manifest artifacts/phase3/master_manifest.csv   --binary-dir artifacts/phase3/binary   --out artifacts/phase4/aligned_feature_manifest.csv   --clean

python phase4_classical_ml.py build-shape   --manifest artifacts/phase3/master_manifest.csv   --binary-dir artifacts/phase3/binary   --out artifacts/phase4/shape_feature_manifest.csv
```

Then train and test. For example:

```bash
python phase4_classical_ml.py all   --manifest artifacts/phase3/master_manifest.csv   --feature-set full   --out-dir artifacts/phase4/full
```

For the aligned / position-normalized branch:

```bash
python phase4_classical_ml.py all   --manifest artifacts/phase4/aligned_feature_manifest.csv   --feature-set aligned   --out-dir artifacts/phase4/aligned
```

For the shape-invariant branch:

```bash
python phase4_classical_ml.py all   --manifest artifacts/phase4/shape_feature_manifest.csv   --feature-set shape_invariant   --out-dir artifacts/phase4/shape
```

The classical pipeline performs model selection on Env1 and evaluates the frozen
models on Env2.

## 3. Run CNN1D / CNN2D

CNN1D, position-normalized:

```bash
python phase5_deep_learning.py all   --architecture cnn1d   --branch position_normalized   --out-dir artifacts/phase5/position_normalized_cnn1d
```

CNN1D, position-aware:

```bash
python phase5_deep_learning.py all   --architecture cnn1d   --branch position_aware   --out-dir artifacts/phase5/position_aware_cnn1d
```

CNN2D, position-normalized:

```bash
python phase5_deep_learning.py all   --architecture cnn2d   --branch position_normalized   --out-dir artifacts/phase5/position_normalized_cnn2d
```

CNN2D, position-aware:

```bash
python phase5_deep_learning.py all   --architecture cnn2d   --branch position_aware   --out-dir artifacts/phase5/position_aware_cnn2d
```

Each `all` command performs Env1 development/model selection followed by the
frozen Env2 test.

For a quick installation/pipeline check:

```bash
python phase5_deep_learning.py smoke
```

## 4. Run Standing vs. Reading (RQ6 / Phase 6)

After Phase 3 is available (and Phase 4 feature tables are available for the
classical branch), the RQ6 runner can execute both branches.

CNN1D:

```bash
python rq6_standing_vs_reading_single.py all CNN1
```

CNN2D:

```bash
python rq6_standing_vs_reading_single.py all CNN2
```

Classical models:

```bash
python rq6_standing_vs_reading_single.py all CLASSICAL
```

These commands use the default output root:

```text
artifacts/rq6/
├── position_normalized_cnn1d/
├── position_aware_cnn1d/
├── position_normalized_cnn2d/
├── position_aware_cnn2d/
├── position_normalized_classical/
└── position_aware_classical/
```

## 5. Consolidate results

Before consolidation, check `RUNS_CONFIG` inside
`consolidate_results.py` and make sure every completed run points to the
correct output directory.

Then run:

```bash
python consolidate_results.py --out-dir artifacts/phase9
```

This produces tagged, combined result tables so that results from different
representations, branches, and model families remain distinguishable.

## Recommended execution order

```text
Raw CSV data
    ↓
phase3_preprocessing.py
    ↓
artifacts/phase3
    ├──→ phase4_classical_ml.py
    │       ├── build
    │       └── all
    │
    ├──→ phase5_deep_learning.py
    │       └── all (CNN1D/CNN2D × two branches)
    │
    └──→ rq6_standing_vs_reading_single.py
            └── all (CNN1/CNN2/CLASSICAL)
                    ↓
            consolidate_results.py
                    ↓
              artifacts/phase9
```

The Google Drive link to Download the Data - https://drive.google.com/file/d/1h9AyIfVoALVj2eov_cVnlCvbAM4UdKfx/view?usp=drive_link
