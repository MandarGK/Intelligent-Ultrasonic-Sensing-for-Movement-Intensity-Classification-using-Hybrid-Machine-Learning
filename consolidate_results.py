"""
Phase 9 — Results consolidation.

Problem this solves: every run (RQ1 classical/CNN, RQ6 classical/CNN,
each feature-set and branch) saves its own agg_model_x_*.csv files in its
own output folder, with no column saying which run/representation/branch
produced it. Concatenating them blindly (as all_project_results.csv did)
loses exactly that information and makes two different runs
indistinguishable -- which is what happened with RQ6's two unlabeled
classical triplets.

Fix: RUNS_CONFIG below is the single source of truth for what each run
IS. You edit this list once (one entry per completed run, pointing at
its output folder), the script reads each run's 4 agg files, tags every
row with that run's metadata, and concatenates into one long-format
table per grouping dimension. Nothing is guessed from folder names or
file order.

Usage
-----
    python consolidate_results.py --out-dir artifacts/phase9

Run this after filling in RUNS_CONFIG below with the real output folder
for each run you've completed. Runs you haven't completed yet should
stay commented out -- missing runs are reported as "not yet included",
never silently skipped without you knowing.
"""

import argparse
from pathlib import Path

import pandas as pd

# =============================================================================
# EDIT THIS — one entry per completed run.
# =============================================================================
#
# dir            : the run's --out-dir (the folder directly containing
#                   agg_model_x_condition.csv etc, and for CNN runs this
#                   is the {branch}_{architecture} subfolder, e.g.
#                   artifacts/phase5/position_normalized_cnn1d)
# research_question : "RQ1" (human vs non-human) or "RQ6" (standing vs reading)
# model_family   : "classical" or "cnn"
# representation : what the model actually saw --
#                   classical: "full" | "shape_invariant" | "shape_invariant_v2" | "aligned"
#                   cnn: "cnn1d" | "cnn2d"
# branch         : "position_aware" | "position_normalized" | "na" (classical
#                   full/shape runs that don't have an explicit branch concept
#                   -- full IS position_aware, shape_invariant IS
#                   position_normalized+amplitude_normalized; set accordingly)
# notes          : free text, carried into the output for your own reference

RUNS_CONFIG = [
    # --- RQ1, classical, full feature set (CONFIRMED from your upload) ---
    {
        "name": "rq1_classical_full",
        "dir": "artifacts/phase4/full",  # EDIT if your actual folder differs
        "research_question": "RQ1", "model_family": "classical",
        "representation": "full", "branch": "position_aware",
        "notes": "Catastrophic ChairTable failure: 0.0% accuracy, all 3 models",
    },
    # --- RQ1, classical, shape_invariant (CONFIRMED -- ran after discovering
    # it was never executed; best model RF: ChairTable 55.1%, a real jump
    # off full's 0.0% floor but well short of CNN1D's 93-98%) ---
    {
        "name": "rq1_classical_shape_invariant",
        "dir": "artifacts/phase4/shape",  # EDIT if your actual folder differs
        "research_question": "RQ1", "model_family": "classical",
        "representation": "shape_invariant", "branch": "position_normalized",
        "notes": "ChairTable: linear 35.4%, RF 55.1%, xgb 50.8%. HumanReading drops "
                 "to ~89% vs full's 99.5% -- same pattern seen in RQ6",
    },

    # --- RQ1, CNN1D, both branches (CONFIRMED) ---
    {
        "name": "rq1_cnn1d_position_aware",
        "dir": "artifacts/phase5/position_aware_cnn1d",
        "research_question": "RQ1", "model_family": "cnn",
        "representation": "cnn1d", "branch": "position_aware",
        "notes": "Best position-aware result: 93.0% ChairTable, far above classical/CNN2D",
    },
    {
        "name": "rq1_cnn1d_position_normalized",
        "dir": "artifacts/phase5/position_normalized_cnn1d",
        "research_question": "RQ1", "model_family": "cnn",
        "representation": "cnn1d", "branch": "position_normalized",
        "notes": "Strongest RQ1 result overall: 97.7% ChairTable with distance removed",
    },

    # --- RQ1, CNN2D, both branches (CONFIRMED) ---
    {
        "name": "rq1_cnn2d_position_aware",
        "dir": "artifacts/phase5/position_aware_cnn2d",
        "research_question": "RQ1", "model_family": "cnn",
        "representation": "cnn2d", "branch": "position_aware",
        "notes": "ChairTable 22.5% -- far worse than CNN1D on the same branch",
    },
    {
        "name": "rq1_cnn2d_position_normalized",
        "dir": "artifacts/phase5/position_normalized_cnn2d",
        "research_question": "RQ1", "model_family": "cnn",
        "representation": "cnn2d", "branch": "position_normalized",
        "notes": "ChairTable 33.5% -- improved but still the weakest of the 4 CNN runs",
    },

    # --- RQ6, classical, shape_invariant (CONFIRMED: dir + feature_set both
    # verified directly from your rglob scan -- RF F1=0.8846 best) ---
    {
        "name": "rq6_classical_shape_invariant",
        "dir": "artifacts/rq6/position_normalized_classical",
        "research_question": "RQ6", "model_family": "classical",
        "representation": "shape_invariant", "branch": "position_normalized",
        "file_prefix": "rq6_", "summary_file": "rq6_env2_test_summary.csv",
        "notes": "RF F1=0.8846 (best of linear/RF/xgboost)",
    },
    # --- RQ6, classical, full (CONFIRMED: this was the mystery second
    # triplet -- it's 'full', not a v2 variant -- XGBoost F1=0.8822 best) ---
    {
        "name": "rq6_classical_full",
        "dir": "artifacts/rq6/position_aware_classical",
        "research_question": "RQ6", "model_family": "classical",
        "representation": "full", "branch": "position_aware",
        "file_prefix": "rq6_", "summary_file": "rq6_env2_test_summary.csv",
        "notes": "trivial baseline (posture/distance shortcut); XGBoost F1=0.8822 best",
    },

    # --- RQ6, CNN1D and CNN2D, both branches (dirs CONFIRMED via rglob scan.
    # Per-activity/per-subject breakdown files not yet uploaded for these --
    # the consolidator will warn and skip those dims gracefully if absent,
    # rather than fail; overall numbers alone are already confirmed:
    # cnn1d position_aware F1=0.9135, position_normalized F1=0.9516,
    # cnn2d position_aware F1=0.9712, position_normalized F1=0.9672 ---
    {
        "name": "rq6_cnn1d_position_aware",
        "dir": "artifacts/rq6/position_aware_cnn1d",
        "research_question": "RQ6", "model_family": "cnn",
        "representation": "cnn1d", "branch": "position_aware",
        "file_prefix": "rq6_", "summary_file": "rq6_env2_test_summary.csv",
        "notes": "trivial baseline; overall F1=0.9135 (confirmed, per-activity breakdown not yet uploaded)",
    },
    {
        "name": "rq6_cnn1d_position_normalized",
        "dir": "artifacts/rq6/position_normalized_cnn1d",
        "research_question": "RQ6", "model_family": "cnn",
        "representation": "cnn1d", "branch": "position_normalized",
        "file_prefix": "rq6_", "summary_file": "rq6_env2_test_summary.csv",
        "notes": "real test; overall F1=0.9516 (confirmed, per-activity breakdown not yet uploaded)",
    },
    {
        "name": "rq6_cnn2d_position_aware",
        "dir": "artifacts/rq6/position_aware_cnn2d",
        "research_question": "RQ6", "model_family": "cnn",
        "representation": "cnn2d", "branch": "position_aware",
        "file_prefix": "rq6_", "summary_file": "rq6_env2_test_summary.csv",
        "notes": "trivial baseline; overall F1=0.9712 (confirmed, per-activity breakdown not yet uploaded)",
    },
    {
        "name": "rq6_cnn2d_position_normalized",
        "dir": "artifacts/rq6/position_normalized_cnn2d",
        "research_question": "RQ6", "model_family": "cnn",
        "representation": "cnn2d", "branch": "position_normalized",
        "file_prefix": "rq6_", "summary_file": "rq6_env2_test_summary.csv",
        "notes": "real test; overall F1=0.9672 (confirmed, per-activity breakdown not yet uploaded)",
    },
]

AGG_FILES = {
    "condition": "agg_model_x_condition.csv",
    "environment": "agg_model_x_environment.csv",
    "subject": "agg_model_x_subject.csv",
    "activity": "agg_model_x_activity.csv",
}

TAG_COLUMNS = ["research_question", "model_family", "representation", "branch", "name", "notes"]


def load_all_runs(runs_config, base_dir: Path):
    """Returns a dict: {dimension: concatenated DataFrame}, each row tagged
    with its run's metadata. Missing files are reported, not silently
    dropped. 'file_prefix' (default "") handles scripts that save their
    agg files as "{prefix}agg_model_x_{dim}.csv" -- e.g. RQ6's "rq6_" prefix."""
    tables = {dim: [] for dim in AGG_FILES}
    missing = []

    for run in runs_config:
        run_dir = base_dir / run["dir"] if not Path(run["dir"]).is_absolute() else Path(run["dir"])
        prefix = run.get("file_prefix", "")
        for dim, fname in AGG_FILES.items():
            fpath = run_dir / f"{prefix}{fname}"
            if not fpath.exists():
                missing.append(f"{run['name']}: {fpath} not found")
                continue
            df = pd.read_csv(fpath)
            for col in TAG_COLUMNS:
                df[col] = run.get(col, "")
            tables[dim].append(df)

    concatenated = {}
    for dim, dfs in tables.items():
        concatenated[dim] = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    return concatenated, missing


def load_overall_summaries(runs_config, base_dir: Path):
    """Loads each run's 'summary_file' (a simpler, one-row-per-model table,
    e.g. rq6_env2_test_summary.csv) -- a different shape than AGG_FILES'
    per-dimension breakdowns, used when only the overall result is
    available yet. Tags every row with the run's metadata the same way."""
    rows = []
    missing = []

    for run in runs_config:
        summary_file = run.get("summary_file")
        if not summary_file:
            continue
        run_dir = base_dir / run["dir"] if not Path(run["dir"]).is_absolute() else Path(run["dir"])
        fpath = run_dir / summary_file
        if not fpath.exists():
            missing.append(f"{run['name']}: {fpath} not found")
            continue
        df = pd.read_csv(fpath)
        # CNN summary files use 'architecture' (cnn1d/cnn2d), not 'model' --
        # normalize so classical and CNN rows both land in the same 'model'
        # column instead of CNN rows silently vanishing from any pivot/groupby
        # keyed on 'model'.
        if "model" not in df.columns and "architecture" in df.columns:
            df["model"] = df["architecture"]
        for col in TAG_COLUMNS:
            df[col] = run.get(col, "")
        rows.append(df)

    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), missing


def build_rq6_overall_pivot(overall_summary: pd.DataFrame) -> pd.DataFrame:
    """representation/branch x model -> accuracy, the RQ6 equivalent of
    the RQ1 condition pivot: lets you see the full-vs-shape-invariant-vs-
    CNN1D-vs-CNN2D, position-aware-vs-normalized picture in one table."""
    rq6 = overall_summary[overall_summary["research_question"] == "RQ6"].copy()
    if rq6.empty:
        return rq6
    rq6["run_label"] = rq6["representation"] + " / " + rq6["branch"]
    pivot = rq6.pivot_table(index="model", columns="run_label", values="accuracy", aggfunc="first")
    return pivot.round(4)


def build_rq1_condition_pivot(by_condition: pd.DataFrame) -> pd.DataFrame:
    """The table that actually answers RQ1/RQ3/RQ4/RQ5 together: accuracy
    per condition, for every representation/branch combination, side by
    side. This is the single most useful table in the whole report.

    Uses the BEST model within each representation/branch (aggfunc='max'),
    not pandas' arbitrary row order ('first') -- a classical run has 3
    models (linear/RF/xgboost) while a CNN run has exactly 1, so picking
    'first' would silently compare an arbitrary (often worst) classical
    model against the CNN's only model. This makes it a fair best-of-
    family comparison instead. The full per-model breakdown (including
    how much models within one representation disagree -- e.g. linear vs
    RF on shape_invariant ChairTable) is preserved in
    all_results_by_condition.csv."""
    rq1 = by_condition[by_condition["research_question"] == "RQ1"].copy()
    if rq1.empty:
        return rq1
    rq1["run_label"] = rq1["representation"] + " / " + rq1["branch"]
    pivot = rq1.pivot_table(index="condition", columns="run_label", values="accuracy", aggfunc="max")
    return pivot.round(4)


def build_rq6_summary(by_activity: pd.DataFrame, by_subject: pd.DataFrame) -> dict:
    rq6_act = by_activity[by_activity["research_question"] == "RQ6"].copy()
    rq6_subj = by_subject[by_subject["research_question"] == "RQ6"].copy()
    return {"by_activity": rq6_act, "by_subject": rq6_subj}


def main():
    ap = argparse.ArgumentParser(description="Phase 9: consolidate results across all runs")
    ap.add_argument("--base-dir", type=str, default=".",
                     help="Base directory the 'dir' paths in RUNS_CONFIG are relative to")
    ap.add_argument("--out-dir", type=str, default="artifacts/phase9")
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {len(RUNS_CONFIG)} configured run(s)...\n")
    tables, missing = load_all_runs(RUNS_CONFIG, base_dir)

    if missing:
        print("WARNING -- these expected files were not found (check RUNS_CONFIG 'dir' paths):")
        for m in missing:
            print(f"  - {m}")
        print()

    for dim, df in tables.items():
        if df.empty:
            continue
        out_path = out_dir / f"all_results_by_{dim}.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved: {out_path}  ({len(df)} rows, {df['name'].nunique()} runs)")

    print()
    rq1_pivot = build_rq1_condition_pivot(tables["condition"])
    if not rq1_pivot.empty:
        rq1_path = out_dir / "rq1_condition_pivot.csv"
        rq1_pivot.to_csv(rq1_path)
        print(f"\nSaved: {rq1_path}")
        print("\nRQ1 condition x representation/branch (accuracy):")
        print(rq1_pivot.to_string())
    else:
        print("\nNo RQ1 runs configured yet -- rq1_condition_pivot.csv not produced.")

    rq6 = build_rq6_summary(tables["activity"], tables["subject"])
    if not rq6["by_activity"].empty:
        rq6["by_activity"].to_csv(out_dir / "rq6_by_activity.csv", index=False)
        rq6["by_subject"].to_csv(out_dir / "rq6_by_subject.csv", index=False)
        print(f"\nSaved: {out_dir/'rq6_by_activity.csv'}, rq6_by_subject.csv")
    else:
        print("\nNo RQ6 runs configured yet (or labeling still unresolved) -- "
              "rq6_by_activity.csv / rq6_by_subject.csv not produced.")

    overall_summary, missing_summaries = load_overall_summaries(RUNS_CONFIG, base_dir)
    if missing_summaries:
        print("WARNING -- these summary files were not found:")
        for m in missing_summaries:
            print(f"  - {m}")
        print()
    if not overall_summary.empty:
        overall_path = out_dir / "rq6_overall_summary.csv"
        overall_summary.to_csv(overall_path, index=False)
        print(f"Saved: {overall_path}")

        rq6_pivot = build_rq6_overall_pivot(overall_summary)
        if not rq6_pivot.empty:
            rq6_pivot_path = out_dir / "rq6_overall_pivot.csv"
            rq6_pivot.to_csv(rq6_pivot_path)
            print(f"Saved: {rq6_pivot_path}")
            print("\nRQ6 model x representation/branch (accuracy):")
            print(rq6_pivot.to_string())

    configured_names = {r["name"] for r in RUNS_CONFIG}
    print(f"\n{len(configured_names)} run(s) included: {sorted(configured_names)}")
    print("\nPhase 9 consolidation complete. Review the pivot tables above before Phase 10 (Analysis).")


if __name__ == "__main__":
    main()
