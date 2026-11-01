# Intelligent Ultrasonic Sensing for Movement Intensity Classification using Hybrid Machine Learning

Individual Project, Master of Engineering Information Technology
Author: Mandar Gokul Kale
Supervisor: Dr. Andreas Pech

This repository contains the full experimental pipeline for classifying ultrasonic echoes
recorded with an SRF02 ultrasonic sensor and a Red Pitaya STEMlab 125-14 board: data
acquisition notes, dataset validation, exploratory analysis, preprocessing, classical
machine learning, 1D/2D CNNs, and results consolidation. The accompanying paper is an
IEEE-format technical report.

---

## 1. What this project asks

Two binary classification tasks are evaluated under a strict **cross-environment protocol**:
every model is developed (trained, tuned, selected) on Environment 1 only, frozen, and
then tested once on Environment 2 with no retraining, refitting, or threshold changes.

| Task | Question | Classes |
|---|---|---|
| **RQ1** | Can ultrasonic echoes separate human presence from non-human conditions? | `HUMAN` vs `NON_HUMAN` (EmptyRoom, ChairTable) |
| **RQ6** | Can Standing be separated from Reading / page-turning? | `Standing` vs `Reading` (human recordings only) |

The study varies the **signal representation** and **model family** independently, so that
the effect of the representation can be separated from the effect of the classifier:

| Representation | Model family | What it retains |
|---|---|---|
| `full` | Logistic regression, Random Forest, XGBoost | 9 statistics on the full 25,000-sample window (absolute timing and amplitude intact) |
| `aligned` | Same three classical models | The same 9 statistics on the onset-aligned window (timing removed, amplitude kept) |
| `shape_invariant` | Same three classical models | 5 ratio features on the onset-aligned, peak-normalized window (timing and amplitude removed) |
| CNN1D, position-aware / position-normalized | 1D CNN | Raw waveform: full window, or onset-aligned 4,900-sample crop |
| CNN2D, position-aware / position-normalized | 2D CNN | Band-limited (20-60 kHz) STFT spectrogram of the same two inputs |

---

## 2. Headline results

All numbers are Env2 test accuracy (%) from frozen models. For the three classical
representations the value is the best of the three classical models per condition.

**RQ1: human-presence detection, by condition**

| Condition | Full | Aligned | Shape | 2D-A | 2D-N | 1D-A | 1D-N |
|---|---|---|---|---|---|---|---|
| ChairTable | 0.00 | 69.91 | 55.14 | 22.48 | 33.53 | 93.02 | **97.72** |
| EmptyRoom | 96.42 | 95.69 | 97.21 | 96.12 | 93.65 | **98.00** | 93.71 |
| HumanReading | 99.51 | 92.73 | 90.89 | **99.93** | 96.36 | 98.59 | 86.06 |
| HumanStanding | 99.97 | 97.34 | 94.08 | **99.98** | 99.74 | 99.97 | 99.91 |

`2D-A` / `2D-N` = CNN2D position-aware / position-normalized; `1D-A` / `1D-N` = CNN1D
position-aware / position-normalized.

**RQ6: Standing vs Reading**

| Model | Position-aware | Position-normalized |
|---|---|---|
| Linear | 81.68 | 79.57 |
| Random Forest | 87.43 | 88.95 |
| XGBoost | 88.73 | 88.76 |
| CNN1D | 91.58 | **95.30** |
| CNN2D | **97.13** | 96.86 |

**Main findings**

1. **ChairTable is the failure mode, and aggregate accuracy hides it.** Under `full`, every
   classical model labels 100% of ChairTable recordings as HUMAN (0.00% accuracy) while
   scoring 99.93% on Env1 validation, the signature of an environment-specific shortcut.
2. **Representation matters as much as architecture.** CNN1D (raw waveform) reaches
   97.72% on ChairTable; CNN2D (band-limited spectrogram), also a deep model, reaches only
   33.53% in the same position-normalized setting.
3. **Normalization is a trade-off.** Representations that fix ChairTable lower HumanReading
   accuracy (e.g. CNN1D 98.59% to 86.06%).
4. **More flexible classical models are more shortcut-prone.** Under `aligned`, Random
   Forest and XGBoost reach 99.3-99.4% validation accuracy but only about 20% on Env2
   ChairTable, while logistic regression (95.5% validation) reaches 69.91%.
5. **The "trivial baseline" for RQ6 was not trivial.** The position-aware branch did not
   dominate; CNN1D is better position-normalized (95.30%) than position-aware (91.58%).
6. **Subject effects are task-specific.** For Standing vs Reading, one Env2 subject is
   classified better than the other under every representation; for human-vs-non-human the
   direction flips between classical and CNN models.

These conclusions are scoped to two environments and two main subjects per environment.

---

## 3. Hardware and acquisition

| Parameter | Value |
|---|---|
| Sensor | SRF02 ultrasonic ranging sensor (I2C address `0x70`, ranging command `0x52`) |
| Acquisition board | Red Pitaya STEMlab 125-14 |
| Master clock / decimation | 125 MHz / 64 |
| Effective sampling rate | 1,953,125 Hz (512 ns per sample) |
| ADC resolution | 14-bit |
| Samples per recording | 25,000 (about 12.8 ms window) |
| Inter-recording interval | about 66.4 ms (about 15 Hz) |
| Carrier frequency | about 40 kHz (confirmed empirically from FFT) |
| Mounting | Tripod, sensor about 2.1 m above the floor, pointing at the measurement area |
| Transport | C program `dma_with_udp_faster2` on the Red Pitaya streams frames over UDP (port 61231, link-local `169.254.x.x`); SSH is used for control |

The SRF02 produces the ultrasonic ping through its own I2C-triggered driver. The Red Pitaya
only performs triggered, DMA-based acquisition of the returning echo on one ADC channel;
its analog outputs are not used.

**Recording format.** Each CSV row is one recording: 17 header floats followed by 25,000 raw
ADC samples (25,017 columns, no header row). The header `Class` field is unused by the
firmware, and the header `Distance` field is constant across the whole dataset, so neither
is used as a label or feature. All labels come from the filename.

**Acquisition caveat.** The PyQt6 recording GUI writes files in append mode. Re-running a
save to the same path stacks a second session on top of the first. One file in this dataset
contained an interrupted first capture (798 rows followed by a 56 s gap); it was trimmed to
the final contiguous 5,000 rows. `inspect_file_sessions` detects this pattern.

---

## 4. Dataset

The raw data is **not included** in this repository (about 700 MB per file, roughly 5 GB in
total, and recordings of real people). Contact the author if you need access.

| | Env1 (development) | Env2 (test) |
|---|---|---|
| Space | Laboratory room | Corridor |
| Non-human | EmptyRoom, ChairTable | EmptyRoom, ChairTable |
| Human | 2 subjects x {Standing, Reading} | 2 main subjects x {Standing, Reading} |
| Held out | none | 1 additional subject x {Standing, Reading} (never used in any reported result) |
| Recordings | 30,000 | 40,000 (30,000 main + 10,000 held-out) |

70,000 recordings in total, 5,000 per condition per subject. In the main pipeline, 50,000
are `HUMAN` and 20,000 are `NON_HUMAN`.

**Filename convention** (labels are parsed from filenames; unparseable names stop the
pipeline rather than being guessed):

```
Room1_EMPTY_5000.csv              Room2_EMPTY_5000.csv
Room1_with_Chair_5000.csv         Room2_with_Chair_5000.csv
ENV1_HUMAN_<SUBJECT>_5000.csv     ENV2_HUMAN_<SUBJECT>_5000.csv
ENV1_HUMAN_SITTING_<SUBJECT>_5000.csv   ENV2_HUMAN_SITTING_<SUBJECT>_5000.csv
```

`SITTING` means seated reading and page-turning. Which subject tokens belong to the main
experiment and which one is held out is set in `ultrasonic_project/configs/data.yaml`.
The paper refers to subjects only as Person 1 to Person 5.

---

## 5. Repository layout

```
.
├── ultrasonic_project/              Phases 1-2: dataset discovery, validation, EDA
│   ├── configs/data.yaml            data_dir, expected constants, subject groups
│   └── src/
│       ├── core/                    manifest, io, signal processing, EDA helpers
│       └── experiments/
│           ├── phase1_discovery.py
│           ├── inspect_file_sessions.py
│           └── phase2_eda.py
├── phase3_preprocessing.py          Phase 3: onset detection, alignment, split, scaler
├── phase4_classical_ml.py           Phase 4: feature tables, RF / XGBoost / linear
├── phase5_deep_learning.py          Phase 5: CNN1D / CNN2D
├── rq6_standing_vs_reading.py       RQ6: Standing vs Reading (classical + CNN)
├── consolidate_results.py           Phase 9: consolidate every run into report tables
├── requirements.txt
└── artifacts/                       generated outputs (do not commit)
```

Phases 3-5, RQ6 and Phase 9 are each a **single self-contained script**; the stage is chosen
with a subcommand.

---

## 6. Installation

Tested with Python 3.13.7 on an Apple MacBook Pro (M3 Pro, 18 GB). CNNs train on the PyTorch
Metal (MPS) backend; classical models run on CPU.

```bash
python -m venv venv
source venv/bin/activate
pip install numpy pandas scipy scikit-learn xgboost joblib pyyaml matplotlib torch
```

Versions used for the reported results: NumPy 2.5.3, pandas 3.0.6, SciPy 1.18.1,
scikit-learn 1.9.1, XGBoost 3.4.1, PyTorch 2.14.0.

The recording GUI additionally needs `PyQt6`, `pyqtgraph`, and `paramiko`.
`xgboost` is optional: if it is missing, the XGBoost model is skipped with a printed note.

---

## 7. Running the pipeline

Run every command from the repository root unless noted. Paths below are examples.

### Phase 1: dataset discovery and validation

Set `data_dir` in `ultrasonic_project/configs/data.yaml`, then:

```bash
cd ultrasonic_project
python -m src.experiments.phase1_discovery
python -m src.experiments.inspect_file_sessions <file.csv>   # only if a file fails the row count
```

Checks column and row counts, NaNs, and the sampling-frequency and ADC-resolution header
fields on every row. It stops on the first parse failure or failed file.

### Phase 2: exploratory analysis

```bash
python -m src.experiments.phase2_eda
```

Writes per-recording statistics, summaries by condition/environment/subject/activity, and
waveform / FFT / spectrogram / RMS plots under `artifacts/phase2_eda/`.

### Phase 3: preprocessing

```bash
python phase3_preprocessing.py --data-dir /path/to/Data --out-dir artifacts/phase3 \
    --chunksize 500 --val-fraction 0.2
```

Streams each CSV once and writes, under `artifacts/phase3/`:

| Output | Content |
|---|---|
| `binary/*_full.npy`, `binary/*_aligned.npy` | memory-mapped waveforms: full window and onset-aligned crop |
| `master_manifest.csv` | one row per recording: metadata, features, onset info, Env1 split |
| `bonus_pd_manifest.csv` | the held-out subject, kept separate |
| `env1_block_split_log.csv` | which time-blocks went to train vs validation |
| `scaler.json` | amplitude scaler fit on Env1-train only |

### Phase 4: classical machine learning

```bash
# build the two normalized feature tables
python phase4_classical_ml.py build-aligned --manifest artifacts/phase3/master_manifest.csv \
    --binary-dir artifacts/phase3/binary --out artifacts/phase4/aligned_feature_manifest.csv
python phase4_classical_ml.py build-shape --manifest artifacts/phase3/master_manifest.csv \
    --binary-dir artifacts/phase3/binary --out artifacts/phase4/shape_feature_manifest.csv

# Env1 development: tune and freeze (feature-set -> manifest)
#   full            -> artifacts/phase3/master_manifest.csv
#   aligned         -> artifacts/phase4/aligned_feature_manifest.csv
#   shape_invariant -> artifacts/phase4/shape_feature_manifest.csv
python phase4_classical_ml.py train --manifest artifacts/phase3/master_manifest.csv \
    --feature-set full --out-dir artifacts/phase4/full

# Env2 final test: frozen models, run once
python phase4_classical_ml.py test --manifest artifacts/phase3/master_manifest.csv \
    --feature-set full --frozen-dir artifacts/phase4/full/frozen_models \
    --out-dir artifacts/phase4/full
```

Use a **separate `--out-dir` per feature set**; outputs are overwritten otherwise.

### Phase 5: CNNs

```bash
python phase5_deep_learning.py smoke-test        # run first: sanity-checks the CNN code on synthetic data

python phase5_deep_learning.py train --manifest artifacts/phase3/master_manifest.csv \
    --binary-dir artifacts/phase3/binary --scaler-path artifacts/phase3/scaler.json \
    --branch position_normalized --architecture cnn1d --out-dir artifacts/phase5

python phase5_deep_learning.py test  --manifest artifacts/phase3/master_manifest.csv \
    --binary-dir artifacts/phase3/binary --scaler-path artifacts/phase3/scaler.json \
    --branch position_normalized --architecture cnn1d --out-dir artifacts/phase5
```

Repeat `train` and `test` for the four combinations of `--branch {position_aware,
position_normalized}` and `--architecture {cnn1d,cnn2d}`. Results land in
`artifacts/phase5/<branch>_<architecture>/`. Optional flags: `--amplitude-normalize`
(divide each waveform by its peak) and `--num-workers N` (if data loading is slow).

### RQ6: Standing vs Reading

```bash
# classical, position-aware baseline
python rq6_standing_vs_reading.py develop-classical --manifest artifacts/phase3/master_manifest.csv \
    --feature-set full --out-dir artifacts/rq6/position_aware_classical
python rq6_standing_vs_reading.py test-classical --manifest artifacts/phase3/master_manifest.csv \
    --feature-set full --frozen-dir artifacts/rq6/position_aware_classical/frozen_models \
    --out-dir artifacts/rq6/position_aware_classical

# classical, position-normalized
python rq6_standing_vs_reading.py develop-classical --manifest artifacts/phase4/shape_feature_manifest.csv \
    --feature-set shape_invariant --out-dir artifacts/rq6/position_normalized_classical

# CNNs
python rq6_standing_vs_reading.py develop-cnn --manifest artifacts/phase3/master_manifest.csv \
    --binary-dir artifacts/phase3/binary --scaler-path artifacts/phase3/scaler.json \
    --branch position_normalized --architecture cnn1d --out-dir artifacts/rq6
python rq6_standing_vs_reading.py test-cnn --manifest artifacts/phase3/master_manifest.csv \
    --binary-dir artifacts/phase3/binary --scaler-path artifacts/phase3/scaler.json \
    --branch position_normalized --architecture cnn1d --out-dir artifacts/rq6
```

### Phase 9: consolidate results

Edit `RUNS_CONFIG` at the top of `consolidate_results.py` so each `dir` points at a finished
run, then:

```bash
python consolidate_results.py --base-dir . --out-dir artifacts/phase9
```

Produces `all_results_by_{condition,environment,subject,activity}.csv`,
`rq1_condition_pivot.csv`, `rq6_overall_pivot.csv`, and RQ6 summary tables. Missing files
are reported as warnings rather than skipped silently.

---

## 8. Method summary

**Leakage prevention** (the central design constraint)

- **Environment hold-out.** Env2 is never used for training, tuning, feature selection,
  threshold choice, scaling, or model selection.
- **Block-level split.** Consecutive recordings are only about 66 ms apart and are
  correlated, so Env1 is split in blocks of 50 consecutive rows (about 3.3 s), stratified by
  condition. No block is divided between train and validation.
- **Scaler guard.** The amplitude scaler can only be fit on `Env1_train`; any other source
  raises an error. It is applied, never refit, to validation and Env2.
- **Frozen models.** Weights, hyperparameters, and scaler parameters are saved before any
  Env2 evaluation (`.joblib` for classical models, `.pt` for CNNs).

**Onset detection and alignment.** The onset of the echo burst is the first sustained
crossing of a per-recording threshold (median + k x MAD of the Hilbert envelope, with a
minimum-duration filter) rather than the global peak, because the strongest reflection is
not always the nearest. The position-normalized input is a fixed 4,900-sample window starting
at that onset. Recordings where no valid onset exists (1,001 of 60,000 main-pipeline
recordings) are excluded from that branch and never zero-padded.

**Features.** Frequency features search only 20-60 kHz around the confirmed carrier; an
unrestricted search returned a dominant frequency of 0.0 Hz for 97.4% of recordings.

| Set | Features |
|---|---|
| `full` (9) | mean, std, RMS, peak, peak-to-peak, crest factor, dominant frequency, band energy, band-energy fraction |
| `shape_invariant` (5) | std/peak, peak-to-peak/peak, crest factor, dominant frequency, band-energy fraction |

**Models and training**

| Model | Detail |
|---|---|
| Logistic regression | `C` in {0.1, 1, 10}; features standardized inside the model |
| Random Forest | (trees, depth) in {(100, none), (200, 10), (200, none), (300, 20)} |
| XGBoost | (trees, depth, learning rate) in {(100, 3, 0.1), (200, 5, 0.05), (300, 4, 0.1)} |
| CNN1D | 3 conv blocks (conv, BatchNorm, ReLU, max-pool), first kernel 25 samples (about one carrier cycle), adaptive average pooling, FC head with dropout |
| CNN2D | same block structure, pooling on the time axis only; input is a 20-60 kHz STFT (256-sample window, 224-sample overlap) |
| CNN training | Adam, binary cross-entropy, early stopping on validation loss (patience 5), best-validation weights restored; 2-point grid over width, dropout, and learning rate |

Model selection uses validation F1 on Env1 only.

---

## 9. Reproducibility notes

- Random seeds are fixed (`random_state=42` for classical models, seeded block split).
- Every stage writes inspectable artifacts: split assignments, scaler parameters, frozen
  models, hyperparameter search logs, and per-recording prediction tables with full metadata.
  All aggregate tables are groupby operations on the prediction tables.
- Under the `full` representation, every point in the hyperparameter grid scores about
  99.92-99.93% on Env1 validation, so the frozen configuration is decided by tie-breaking on
  grid order. Re-running can therefore shift a few Env2 numbers in the second decimal place
  (observed: at most 0.02 percentage points). The conclusions do not change.
- Results marked `TBD` or absent in a table mean the experiment was not run; nothing is
  estimated.

---

## 10. Limitations

- Two environments only, and two main subjects per environment; claims about environment or
  subject generalization hold for these spaces and people, not in general.
- One chair placement defines ChairTable; other static objects were not tested.
- Single sensor geometry (fixed height and orientation).
- The held-out subject is not yet used in any reported evaluation.
- The header `Distance` field is constant and uninformative, so range cannot be cross-checked
  against the sensor's own estimate.

---

## 11. Citation

If you use this work, please cite the accompanying paper:

> M. G. Kale, "Intelligent Ultrasonic Sensing for Movement Intensity Classification using
> Hybrid Machine Learning," Individual Project, Master of Engineering Information Technology.

## 12. License

No license has been chosen yet. Add one (for example MIT for the code) before making the
repository public.

## 13. Acknowledgment

Thanks to Dr. Andreas Pech for guidance and supervision throughout this project.