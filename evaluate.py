"""
Evaluation script for the MLLM prototype.

Loads all Final_Runs CSVs, deduplicates multi-model batch files by filtering
on the `mllm` column, then computes:
  1. Accuracy per model / run / category (mean ± std)
  2. Failed cases export for manual reasoning review
  3. Hallucination rate (model reports element present when ground truth says absent)
  4. Precision, Recall, F1 (binary per-screenshot element detection)
  5. Corrected metrics using manual `corrected_classification` annotations
  6. Stability (inter-run consistency: same classification across all 3 runs?)
  7. Performance metrics (latency, tokens, tokens/sec)
  8. Category-specific accuracy breakdown

Results are written to evaluation_results/ and printed to the console.
"""

import os
import re
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FINAL_RUNS_DIR = Path(__file__).parent / "data" / "runs" / "Final_Runs"
OUTPUT_DIR = Path(__file__).parent / "evaluation_results"

# Map each model folder name to a substring that must appear in the `mllm`
# column so that shared batch CSVs are filtered to the correct model's rows.
MODEL_FILTER: dict[str, str] = {
    "GPT_5.2": "gpt-5.2",
    "Claude_Sonnet_4.6": "claude-sonnet-4-6",
    "Gemini_3.1 Pro": "gemini-3.1-pro-preview",
    "LLAVA": "llava",
    "Qwen3-VL": "qwen3-vl",
}

# Friendly display names for console output
MODEL_DISPLAY: dict[str, str] = {
    "GPT_5.2": "GPT-5.2",
    "Claude_Sonnet_4.6": "Claude Sonnet 4.6",
    "Gemini_3.1 Pro": "Gemini 3.1 Pro",
    "LLAVA": "LLaVA (local)",
    "Qwen3-VL": "Qwen3-VL 32b",
}

# Ground-truth values that mean the element is truly absent from each screenshot
ABSENT_FROM_A = {"only_b", "neither"}
ABSENT_FROM_B = {"only_a", "neither"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _parse_run_number(path: Path) -> int | None:
    """Extract run number from a path segment like 'Run_2'."""
    for part in path.parts:
        m = re.fullmatch(r"Run_(\d+)", part, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def _parse_category(filename: str) -> int | None:
    """Extract category number from filenames like 'results_categorie_3.csv'."""
    m = re.search(r"categorie_(\d+)", filename)
    if m:
        return int(m.group(1))
    return None


def load_data() -> pd.DataFrame:
    """
    Load all CSVs from Final_Runs, tagging each row with model_folder, run,
    and category.  Qwen Run_4 (timestamp-named files) is excluded; a warning
    is printed.

    Multi-model batch files are deduplicated by keeping only rows whose `mllm`
    value contains the expected substring for the folder being loaded.
    """
    frames: list[pd.DataFrame] = []
    skipped_files: list[str] = []

    for model_folder, mllm_filter in MODEL_FILTER.items():
        folder_path = FINAL_RUNS_DIR / model_folder
        if not folder_path.exists():
            print(f"[WARN] Folder not found, skipping: {folder_path}")
            continue

        csv_files = sorted(folder_path.rglob("*.csv"))
        for csv_path in csv_files:
            run_num = _parse_run_number(csv_path)
            category = _parse_category(csv_path.name)

            # Skip Qwen Run_4 (timestamp filenames, incomplete data)
            if model_folder == "Qwen3-VL" and run_num == 4:
                skipped_files.append(str(csv_path.relative_to(FINAL_RUNS_DIR)))
                continue

            # Skip files that don't match the expected naming pattern
            if run_num is None or category is None:
                skipped_files.append(str(csv_path.relative_to(FINAL_RUNS_DIR)))
                continue

            try:
                df = pd.read_csv(csv_path, dtype=str)
            except Exception as exc:
                print(f"[WARN] Could not read {csv_path}: {exc}")
                continue

            # Filter to only the rows belonging to this model
            mllm_col = df["mllm"].str.lower()
            df = df[mllm_col.str.contains(mllm_filter.lower(), na=False)].copy()

            if df.empty:
                continue

            df["model_folder"] = model_folder
            df["model_display"] = MODEL_DISPLAY[model_folder]
            df["run"] = run_num
            df["category"] = category

            frames.append(df)

    if skipped_files:
        print("[INFO] Skipped files (Qwen Run_4 or unrecognised naming):")
        for f in skipped_files:
            print(f"       {f}")
        print()

    if not frames:
        sys.exit("[ERROR] No data loaded — check FINAL_RUNS_DIR path.")

    combined = pd.concat(frames, ignore_index=True)

    # Normalise boolean-like columns
    for col in ("correct", "presence_a", "presence_b"):
        combined[col] = combined[col].str.strip().str.lower().map(
            {"true": True, "false": False}
        )

    # Normalise numeric columns
    for col in ("mllm_avg_latency_ms", "mllm_avg_tokens", "mllm_tokens_per_sec",
                "mllm_accuracy", "mllm_correct_count", "mllm_evaluated_count"):
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

    return combined


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _fmt(value: float, decimals: int = 4) -> str:
    return f"{value:.{decimals}f}"


def _accuracy_table(df: pd.DataFrame, groupby: list[str]) -> pd.DataFrame:
    """Return accuracy (0–1) and sample counts for an arbitrary grouping."""
    agg = (
        df.groupby(groupby)["correct"]
        .agg(correct_count="sum", total="count")
        .reset_index()
    )
    agg["accuracy"] = agg["correct_count"] / agg["total"]
    return agg


# ---------------------------------------------------------------------------
# 1. Accuracy analysis
# ---------------------------------------------------------------------------

def compute_accuracy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      per_run_cat  — accuracy for every (model, run, category) cell
      summary      — per-model mean accuracy ± std across runs, plus totals
    """
    per_run_cat = _accuracy_table(df, ["model_display", "run", "category"])

    # Per-model summary: mean and std of per-(run,category) accuracies
    summary = (
        per_run_cat.groupby("model_display")["accuracy"]
        .agg(
            accuracy_mean="mean",
            accuracy_std="std",
            n_cells="count",
        )
        .reset_index()
    )
    # Also add raw totals for reference
    totals = _accuracy_table(df, ["model_display"]).rename(
        columns={"correct_count": "total_correct", "total": "total_evaluated"}
    )
    summary = summary.merge(totals, on="model_display")
    summary["accuracy_overall"] = summary["total_correct"] / summary["total_evaluated"]
    return per_run_cat, summary


# ---------------------------------------------------------------------------
# 2. Reasoning export (failed cases for manual review)
# ---------------------------------------------------------------------------

def export_failed_cases(df: pd.DataFrame) -> pd.DataFrame:
    failed = df[df["correct"] == False].copy()  # noqa: E712
    cols = [
        "model_display", "run", "category", "pair_id",
        "element", "ground_truth", "model_classification",
        "presence_a", "presence_b", "reasoning",
    ]
    return failed[cols].sort_values(["model_display", "run", "category", "pair_id"])


# ---------------------------------------------------------------------------
# 3. Hallucination analysis
# ---------------------------------------------------------------------------

def compute_hallucinations(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Hallucination (broad): model reports an element as present in a screenshot
    where the ground truth says it is absent.

      Screenshot A hallucination: ground_truth ∈ {only_b, neither} AND presence_a is True
      Screenshot B hallucination: ground_truth ∈ {only_a, neither} AND presence_b is True

    Hallucination (strict): ground_truth == 'neither' AND model_classification != 'neither'
    (the element does not exist in either image but the model claims it appears somewhere)
    """
    d = df.copy()
    d["gt_norm"] = d["ground_truth"].str.strip().str.lower()
    d["mc_norm"] = d["model_classification"].str.strip().str.lower()

    d["halluc_a"] = d["gt_norm"].isin(ABSENT_FROM_A) & (d["presence_a"] == True)  # noqa: E712
    d["halluc_b"] = d["gt_norm"].isin(ABSENT_FROM_B) & (d["presence_b"] == True)  # noqa: E712
    d["halluc_any"] = d["halluc_a"] | d["halluc_b"]
    d["halluc_strict"] = (d["gt_norm"] == "neither") & (d["mc_norm"] != "neither")

    detail_cols = [
        "model_display", "run", "category", "pair_id", "element",
        "ground_truth", "model_classification", "presence_a", "presence_b",
        "halluc_a", "halluc_b", "halluc_any", "halluc_strict",
    ]
    detail = d[detail_cols].copy()

    # Per-model aggregates
    # Broad: each row contributes 2 screenshot assessments (A and B)
    agg = d.groupby("model_display").agg(
        halluc_a_count=("halluc_a", "sum"),
        halluc_b_count=("halluc_b", "sum"),
        halluc_any_count=("halluc_any", "sum"),
        halluc_strict_count=("halluc_strict", "sum"),
        total_pairs=("pair_id", "count"),
    ).reset_index()

    agg["total_screenshot_assessments"] = agg["total_pairs"] * 2
    agg["halluc_broad_rate"] = (
        (agg["halluc_a_count"] + agg["halluc_b_count"]) /
        agg["total_screenshot_assessments"]
    )
    agg["halluc_pair_rate"] = agg["halluc_any_count"] / agg["total_pairs"]
    agg["halluc_strict_rate"] = agg["halluc_strict_count"] / agg["total_pairs"]

    return detail, agg


# ---------------------------------------------------------------------------
# 4. Precision, Recall, F1
# ---------------------------------------------------------------------------

def compute_precision_recall(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary per-screenshot element detection metrics.

    Positive class = "element is present in this screenshot".
    Each pair contributes two assessments (Screenshot A + B), giving 540 data
    points per model across 270 pairs.

      TP: ground truth = present, model says present
      FP: ground truth = absent,  model says present  (= hallucination)
      FN: ground truth = present, model says absent   (= missed detection)

    Precision = TP / (TP + FP)  — trustworthiness of a positive prediction
    Recall    = TP / (TP + FN)  — completeness of detection
    F1        = harmonic mean of Precision and Recall
    """
    d = df.copy()
    d["gt_norm"] = d["ground_truth"].str.strip().str.lower()

    d["true_a"] = d["gt_norm"].isin({"only_a", "both"})
    d["true_b"] = d["gt_norm"].isin({"only_b", "both"})

    d["tp"] = (
        (d["true_a"] & (d["presence_a"] == True)).astype(int)   # noqa: E712
        + (d["true_b"] & (d["presence_b"] == True)).astype(int)  # noqa: E712
    )
    d["fp"] = (
        (~d["true_a"] & (d["presence_a"] == True)).astype(int)   # noqa: E712
        + (~d["true_b"] & (d["presence_b"] == True)).astype(int)  # noqa: E712
    )
    d["fn"] = (
        (d["true_a"] & (d["presence_a"] == False)).astype(int)   # noqa: E712
        + (d["true_b"] & (d["presence_b"] == False)).astype(int)  # noqa: E712
    )

    agg = d.groupby("model_display")[["tp", "fp", "fn"]].sum().reset_index()
    agg["precision"] = agg["tp"] / (agg["tp"] + agg["fp"])
    agg["recall"] = agg["tp"] / (agg["tp"] + agg["fn"])
    agg["f1"] = (
        2 * agg["precision"] * agg["recall"]
        / (agg["precision"] + agg["recall"])
    )
    return agg


# ---------------------------------------------------------------------------
# 5. Corrected metrics (manual annotations in failed_cases.csv)
# ---------------------------------------------------------------------------

def _load_corrected_df(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Merge `corrected_classification` from the annotated failed_cases.csv into df.

    Fill rules:
      - Originally correct rows            → correct_classification = ground_truth
      - Failed rows with annotation        → use the annotated value
      - Failed rows without annotation     → correct_classification = model_classification
        (still genuinely wrong, no change)

    Returns the enriched DataFrame or None if the annotation column is unavailable.
    """
    failed_path = OUTPUT_DIR / "failed_cases.csv"
    if not failed_path.exists():
        return None
    annotated = pd.read_csv(failed_path, dtype=str)
    if "corrected_classification" not in annotated.columns:
        return None

    merge_keys = ["model_display", "run", "category", "pair_id", "element"]
    annotated["run"] = annotated["run"].astype(str)
    annotated["category"] = annotated["category"].astype(str)

    d = df.copy()
    d["run"] = d["run"].astype(str)
    d["category"] = d["category"].astype(str)

    d = d.merge(
        annotated[merge_keys + ["corrected_classification"]].rename(
            columns={"corrected_classification": "correct_classification"}
        ),
        on=merge_keys,
        how="left",
    )

    was_correct = d["correct"] == True  # noqa: E712
    d.loc[was_correct, "correct_classification"] = (
        d.loc[was_correct, "correct_classification"].fillna(d.loc[was_correct, "ground_truth"])
    )
    d.loc[~was_correct, "correct_classification"] = (
        d.loc[~was_correct, "correct_classification"].fillna(d.loc[~was_correct, "model_classification"])
    )
    return d


def compute_corrected_metrics(
    df: pd.DataFrame, original_pr: pd.DataFrame, original_acc: pd.DataFrame
) -> pd.DataFrame | None:
    """
    Read `corrected_classification` from the annotated failed_cases.csv and merge
    it back into the full dataset to compute corrected Accuracy, Precision,
    Recall, and F1.

    For rows that were already correct (not in failed_cases), corrected_classification
    is filled with ground_truth — they are correct by definition.

    Returns a per-model DataFrame with original vs corrected metrics side by side,
    or None if the annotation column is missing.
    """
    d = _load_corrected_df(df)
    if d is None:
        print("[WARN] failed_cases.csv has no 'corrected_classification' column — skipping corrected metrics.")
        return None

    d["cc_norm"] = d["correct_classification"].str.strip().str.lower()
    d["gt_norm"] = d["ground_truth"].str.strip().str.lower()

    # Corrected accuracy flag
    d["corrected_correct"] = d["cc_norm"] == d["gt_norm"]

    # Corrected presence flags (what the model effectively detected, post-correction)
    d["corr_pres_a"] = d["cc_norm"].isin({"only_a", "both"})
    d["corr_pres_b"] = d["cc_norm"].isin({"only_b", "both"})

    # Ground-truth presence flags (same logic as compute_precision_recall)
    d["true_a"] = d["gt_norm"].isin({"only_a", "both"})
    d["true_b"] = d["gt_norm"].isin({"only_b", "both"})

    d["c_tp"] = (d["true_a"] & d["corr_pres_a"]).astype(int) \
              + (d["true_b"] & d["corr_pres_b"]).astype(int)
    d["c_fp"] = (~d["true_a"] & d["corr_pres_a"]).astype(int) \
              + (~d["true_b"] & d["corr_pres_b"]).astype(int)
    d["c_fn"] = (d["true_a"] & ~d["corr_pres_a"]).astype(int) \
              + (d["true_b"] & ~d["corr_pres_b"]).astype(int)

    # How many failed cases had their classification changed
    d["reclassified"] = (
        (d["corrected_correct"] == True)  # noqa: E712
        & (d["correct"] == False)         # noqa: E712
    )

    # Per-model corrected accuracy (overall)
    corr_acc = (
        d.groupby("model_display")
        .agg(
            corr_correct_count=("corrected_correct", "sum"),
            total=("corrected_correct", "count"),
            n_reclassified=("reclassified", "sum"),
        )
        .reset_index()
    )
    corr_acc["corrected_accuracy"] = corr_acc["corr_correct_count"] / corr_acc["total"]

    # Per-model corrected P/R/F1
    corr_pr = d.groupby("model_display")[["c_tp", "c_fp", "c_fn"]].sum().reset_index()
    corr_pr["corrected_precision"] = corr_pr["c_tp"] / (corr_pr["c_tp"] + corr_pr["c_fp"])
    corr_pr["corrected_recall"] = corr_pr["c_tp"] / (corr_pr["c_tp"] + corr_pr["c_fn"])
    corr_pr["corrected_f1"] = (
        2 * corr_pr["corrected_precision"] * corr_pr["corrected_recall"]
        / (corr_pr["corrected_precision"] + corr_pr["corrected_recall"])
    )

    # Merge original metrics in for side-by-side comparison
    orig_acc_lookup = original_acc[["model_display", "accuracy_overall"]].rename(
        columns={"accuracy_overall": "original_accuracy"}
    )
    orig_pr_lookup = original_pr[["model_display", "precision", "recall", "f1"]].rename(
        columns={"precision": "original_precision", "recall": "original_recall", "f1": "original_f1"}
    )

    result = (
        corr_acc
        .merge(corr_pr[["model_display", "corrected_precision", "corrected_recall", "corrected_f1"]],
               on="model_display")
        .merge(orig_acc_lookup, on="model_display")
        .merge(orig_pr_lookup, on="model_display")
    )

    result["accuracy_delta"] = result["corrected_accuracy"] - result["original_accuracy"]
    result["precision_delta"] = result["corrected_precision"] - result["original_precision"]
    result["recall_delta"] = result["corrected_recall"] - result["original_recall"]
    result["f1_delta"] = result["corrected_f1"] - result["original_f1"]

    return result[
        [
            "model_display", "n_reclassified",
            "original_accuracy", "corrected_accuracy", "accuracy_delta",
            "original_precision", "corrected_precision", "precision_delta",
            "original_recall", "corrected_recall", "recall_delta",
            "original_f1", "corrected_f1", "f1_delta",
        ]
    ]


# ---------------------------------------------------------------------------
# 6. Stability (inter-run consistency)
# ---------------------------------------------------------------------------

def compute_stability(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every unique (model, category, pair_id, element) combination the model
    evaluated across 3 runs, check whether the `model_classification` was
    identical in all runs.

    A pair is 'stable' if nunique(model_classification) == 1 across its 3 rows.

    Returns per-model stability rates, also broken down by category.
    """
    grp = (
        df.groupby(["model_display", "category", "pair_id", "element"])["model_classification"]
        .nunique()
        .reset_index(name="n_unique_classifications")
    )
    grp["stable"] = grp["n_unique_classifications"] == 1

    # Overall per model
    overall = (
        grp.groupby("model_display")["stable"]
        .agg(stable_pairs="sum", total_unique_pairs="count")
        .reset_index()
    )
    overall["stability_rate"] = overall["stable_pairs"] / overall["total_unique_pairs"]

    # Per model × category
    by_cat = (
        grp.groupby(["model_display", "category"])["stable"]
        .agg(stable_pairs="sum", total_unique_pairs="count")
        .reset_index()
    )
    by_cat["stability_rate"] = by_cat["stable_pairs"] / by_cat["total_unique_pairs"]

    return overall, by_cat


def compute_corrected_stability(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Same as compute_stability but using `correct_classification` (post annotation).
    Returns None if the annotation column is unavailable.
    """
    d = _load_corrected_df(df)
    if d is None:
        return None, None

    d["cc_norm"] = d["correct_classification"].str.strip().str.lower()

    grp = (
        d.groupby(["model_display", "category", "pair_id", "element"])["cc_norm"]
        .nunique()
        .reset_index(name="n_unique_classifications")
    )
    grp["stable"] = grp["n_unique_classifications"] == 1

    overall = (
        grp.groupby("model_display")["stable"]
        .agg(corrected_stable_pairs="sum", total_unique_pairs="count")
        .reset_index()
    )
    overall["corrected_stability_rate"] = (
        overall["corrected_stable_pairs"] / overall["total_unique_pairs"]
    )

    by_cat = (
        grp.groupby(["model_display", "category"])["stable"]
        .agg(corrected_stable_pairs="sum", total_unique_pairs="count")
        .reset_index()
    )
    by_cat["corrected_stability_rate"] = (
        by_cat["corrected_stable_pairs"] / by_cat["total_unique_pairs"]
    )

    return overall, by_cat


# ---------------------------------------------------------------------------
# 7. Performance metrics
# ---------------------------------------------------------------------------

def compute_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    mllm_avg_* columns are run-level aggregates repeated on every row.
    Deduplicate to one value per (model, run, category) before averaging.
    """
    dedup = (
        df.drop_duplicates(subset=["model_display", "run", "category"])
        [["model_display", "run", "category",
          "mllm_avg_latency_ms", "mllm_avg_tokens", "mllm_tokens_per_sec"]]
    )
    perf = (
        dedup.groupby("model_display")
        .agg(
            avg_latency_ms=("mllm_avg_latency_ms", "mean"),
            std_latency_ms=("mllm_avg_latency_ms", "std"),
            avg_tokens=("mllm_avg_tokens", "mean"),
            std_tokens=("mllm_avg_tokens", "std"),
            avg_tokens_per_sec=("mllm_tokens_per_sec", "mean"),
            std_tokens_per_sec=("mllm_tokens_per_sec", "std"),
            n_run_category_cells=("mllm_avg_latency_ms", "count"),
        )
        .reset_index()
    )
    return perf


# ---------------------------------------------------------------------------
# 5. Category-specific accuracy
# ---------------------------------------------------------------------------

def compute_category_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    cat_acc = _accuracy_table(df, ["model_display", "category"])
    # Pivot for easy reading
    pivot = cat_acc.pivot(
        index="model_display", columns="category", values="accuracy"
    ).reset_index()
    pivot.columns.name = None
    pivot.rename(
        columns={c: f"cat_{c}_accuracy" for c in pivot.columns if isinstance(c, int)},
        inplace=True,
    )
    return pivot


# ---------------------------------------------------------------------------
# Console printing helpers
# ---------------------------------------------------------------------------

SEP = "-" * 90


def _print_section(title: str) -> None:
    print(f"\n{'=' * 90}")
    print(f"  {title}")
    print(f"{'=' * 90}")


def print_accuracy_summary(summary: pd.DataFrame) -> None:
    _print_section("ACCURACY SUMMARY (per model)")
    display = summary[[
        "model_display", "accuracy_mean", "accuracy_std",
        "accuracy_overall", "total_correct", "total_evaluated",
    ]].copy()
    display.columns = [
        "Model", "Mean Acc (over runs×cats)", "Std Dev",
        "Overall Acc", "Correct", "Total",
    ]
    display["Mean Acc (over runs×cats)"] = display["Mean Acc (over runs×cats)"].map(
        lambda x: f"{x:.4f}"
    )
    display["Std Dev"] = display["Std Dev"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
    display["Overall Acc"] = display["Overall Acc"].map(lambda x: f"{x:.4f}")
    print(display.to_string(index=False))


def print_hallucination_summary(agg: pd.DataFrame) -> None:
    _print_section("HALLUCINATION RATES (per model)")
    display = agg[[
        "model_display",
        "halluc_broad_rate", "halluc_pair_rate", "halluc_strict_rate",
        "halluc_any_count", "total_pairs",
    ]].copy()
    display.columns = [
        "Model",
        "Broad Rate (per screenshot)", "Pair Rate (≥1 halluc in pair)",
        "Strict Rate (neither→present)",
        "Pairs w/ hallucination", "Total Pairs",
    ]
    for col in ["Broad Rate (per screenshot)", "Pair Rate (≥1 halluc in pair)",
                "Strict Rate (neither→present)"]:
        display[col] = display[col].map(lambda x: f"{x:.4f}")
    print(display.to_string(index=False))
    print()
    print("  Broad: model marks element present in a screenshot where ground truth says absent")
    print("  Strict: ground_truth == 'neither' (absent everywhere) but model claims presence")


def print_precision_recall(pr: pd.DataFrame) -> None:
    _print_section("PRECISION / RECALL / F1 (binary per-screenshot element detection)")
    display = pr[["model_display", "tp", "fp", "fn", "precision", "recall", "f1"]].copy()
    display.columns = ["Model", "TP", "FP", "FN", "Precision", "Recall", "F1"]
    for col in ("Precision", "Recall", "F1"):
        display[col] = display[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
    print(display.to_string(index=False))
    print()
    print("  Positive class: element is present in a screenshot")
    print("  FP = hallucination (absent but model says present)")
    print("  FN = missed detection (present but model says absent)")


def print_corrected_metrics(result: pd.DataFrame) -> None:
    _print_section("CORRECTED METRICS (after manual reasoning review)")
    print(f"  {'Model':<22} {'Reclassified':>12}  "
          f"{'Orig Acc':>9} {'Corr Acc':>9} {'Δ Acc':>7}  "
          f"{'Orig P':>7} {'Corr P':>7} {'Δ P':>6}  "
          f"{'Orig R':>7} {'Corr R':>7} {'Δ R':>6}  "
          f"{'Orig F1':>8} {'Corr F1':>8} {'Δ F1':>6}")
    print("  " + "-" * 133)
    for _, row in result.sort_values("model_display").iterrows():
        def _d(v: float) -> str:
            return f"{v:+.4f}" if pd.notna(v) else "—"

        def _f(v: float) -> str:
            return f"{v:.4f}" if pd.notna(v) else "—"

        print(
            f"  {row['model_display']:<22} {int(row['n_reclassified']):>12}  "
            f"{_f(row['original_accuracy']):>9} {_f(row['corrected_accuracy']):>9} {_d(row['accuracy_delta']):>7}  "
            f"{_f(row['original_precision']):>7} {_f(row['corrected_precision']):>7} {_d(row['precision_delta']):>6}  "
            f"{_f(row['original_recall']):>7} {_f(row['corrected_recall']):>7} {_d(row['recall_delta']):>6}  "
            f"{_f(row['original_f1']):>8} {_f(row['corrected_f1']):>8} {_d(row['f1_delta']):>6}"
        )
    print()
    print("  Reclassified: failed cases where corrected_classification == ground_truth")
    print("  Δ = corrected − original  (positive = model was better than the raw label suggests)")


def print_stability(
    overall: pd.DataFrame,
    by_cat: pd.DataFrame,
    corr_overall: pd.DataFrame | None = None,
    corr_by_cat: pd.DataFrame | None = None,
) -> None:
    _print_section("STABILITY / INTER-RUN CONSISTENCY")
    has_corrected = corr_overall is not None

    if has_corrected:
        merged = overall.merge(
            corr_overall[["model_display", "corrected_stable_pairs", "corrected_stability_rate"]],
            on="model_display", how="left",
        )
        merged["delta"] = merged["corrected_stability_rate"] - merged["stability_rate"]
        display = merged[["model_display", "stable_pairs", "total_unique_pairs",
                           "stability_rate", "corrected_stable_pairs",
                           "corrected_stability_rate", "delta"]].copy()
        display.columns = ["Model", "Stable (orig)", "Total Pairs",
                           "Rate (orig)", "Stable (corr)", "Rate (corr)", "Δ"]
        display["Rate (orig)"] = display["Rate (orig)"].map(lambda x: f"{x:.4f}")
        display["Rate (corr)"] = display["Rate (corr)"].map(
            lambda x: f"{x:.4f}" if pd.notna(x) else "—"
        )
        display["Δ"] = display["Δ"].map(lambda x: f"{x:+.4f}" if pd.notna(x) else "—")
    else:
        display = overall[["model_display", "stable_pairs", "total_unique_pairs",
                           "stability_rate"]].copy()
        display.columns = ["Model", "Stable Pairs", "Total Pairs", "Stability Rate"]
        display["Stability Rate"] = display["Stability Rate"].map(lambda x: f"{x:.4f}")

    print(display.to_string(index=False))
    print()
    print("  Stable: model gave identical classification in all 3 runs for a given pair")
    print("  Total Pairs = unique (pair_id × element) combinations per model (90)")

    # Category breakdown
    print()
    print("  --- By Category ---")
    if has_corrected:
        by_cat_str = by_cat.copy()
        by_cat_str["category"] = by_cat_str["category"].astype(str)
        corr_by_cat_str = corr_by_cat.copy()
        corr_by_cat_str["category"] = corr_by_cat_str["category"].astype(str)
        cat_merged = by_cat_str.merge(
            corr_by_cat_str[["model_display", "category", "corrected_stability_rate"]],
            on=["model_display", "category"], how="left",
        )
        cat_pivot_orig = by_cat_str.pivot(
            index="model_display", columns="category", values="stability_rate"
        )
        cat_pivot_orig.columns = [f"Cat {c} (orig)" for c in cat_pivot_orig.columns]
        cat_pivot_corr = corr_by_cat_str.pivot(
            index="model_display", columns="category", values="corrected_stability_rate"
        )
        cat_pivot_corr.columns = [f"Cat {c} (corr)" for c in cat_pivot_corr.columns]
        cat_display = cat_pivot_orig.join(cat_pivot_corr).reset_index()
    else:
        cat_pivot = by_cat.pivot(
            index="model_display", columns="category", values="stability_rate"
        ).reset_index()
        cat_pivot.columns = ["Model"] + [f"Cat {c}" for c in cat_pivot.columns[1:]]
        cat_display = cat_pivot

    for col in cat_display.columns[1:]:
        cat_display[col] = cat_display[col].map(
            lambda x: f"{x:.4f}" if pd.notna(x) else "—"
        )
    if "model_display" in cat_display.columns:
        cat_display = cat_display.rename(columns={"model_display": "Model"})
    print(cat_display.to_string(index=False))


def print_performance_summary(perf: pd.DataFrame) -> None:
    _print_section("PERFORMANCE METRICS (per model, mean ± std)")
    display = perf[[
        "model_display",
        "avg_latency_ms", "std_latency_ms",
        "avg_tokens", "std_tokens",
        "avg_tokens_per_sec", "std_tokens_per_sec",
    ]].copy()
    display["Latency (ms)"] = display.apply(
        lambda r: f"{r['avg_latency_ms']:.0f} ± {r['std_latency_ms']:.0f}"
        if pd.notna(r["std_latency_ms"]) else f"{r['avg_latency_ms']:.0f}",
        axis=1,
    )
    display["Tokens / req"] = display.apply(
        lambda r: f"{r['avg_tokens']:.1f} ± {r['std_tokens']:.1f}"
        if pd.notna(r["std_tokens"]) else f"{r['avg_tokens']:.1f}",
        axis=1,
    )
    display["Tokens / sec"] = display.apply(
        lambda r: f"{r['avg_tokens_per_sec']:.1f} ± {r['std_tokens_per_sec']:.1f}"
        if pd.notna(r["std_tokens_per_sec"]) else f"{r['avg_tokens_per_sec']:.1f}",
        axis=1,
    )
    out = display[["model_display", "Latency (ms)", "Tokens / req", "Tokens / sec"]]
    out = out.rename(columns={"model_display": "Model"})
    print(out.to_string(index=False))


def print_category_accuracy(per_run_cat: pd.DataFrame) -> None:
    _print_section("ACCURACY BY CATEGORY (per model, mean ± std across runs)")
    # per_run_cat has one accuracy value per (model_display, run, category)
    pivot = per_run_cat.groupby(["model_display", "category"])["accuracy"].agg(
        ["mean", "std"]
    ).reset_index()
    pivot.columns = ["Model", "Category", "Mean Acc", "Std"]
    pivot["Mean ± Std"] = pivot.apply(
        lambda r: f"{r['Mean Acc']:.4f} ± {r['Std']:.4f}"
        if pd.notna(r["Std"]) else f"{r['Mean Acc']:.4f}",
        axis=1,
    )
    pivoted = pivot.pivot(index="Model", columns="Category", values="Mean ± Std")
    pivoted.columns = [f"Cat {c}" for c in pivoted.columns]
    print(pivoted.to_string())


def print_failed_summary(failed: pd.DataFrame) -> None:
    _print_section("FAILED CASES SUMMARY (for manual reasoning review)")
    counts = failed.groupby("model_display").size().reset_index(name="failed_count")
    totals = failed.groupby("model_display")["pair_id"].count()  # same as above, for reference
    print(counts.rename(columns={"model_display": "Model"}).to_string(index=False))
    print(f"\n  Total failed cases: {len(failed)}")
    print("  Full reasoning text exported to evaluation_results/failed_cases.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Loading data from: {FINAL_RUNS_DIR}")
    df = load_data()
    print(f"Loaded {len(df):,} rows across {df['model_display'].nunique()} models.\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Accuracy
    per_run_cat, acc_summary = compute_accuracy(df)
    print_accuracy_summary(acc_summary)
    acc_summary.to_csv(OUTPUT_DIR / "summary_accuracy.csv", index=False)
    per_run_cat.to_csv(OUTPUT_DIR / "accuracy_by_run_category.csv", index=False)

    # 2. Failed cases / reasoning export
    failed = export_failed_cases(df)
    print_failed_summary(failed)
    failed_path = OUTPUT_DIR / "failed_cases.csv"
    _write_failed = True
    if failed_path.exists():
        try:
            existing_cols = pd.read_csv(failed_path, nrows=0).columns.tolist()
            if "corrected_classification" in existing_cols:
                print("[INFO] failed_cases.csv already has 'corrected_classification' — skipping overwrite to preserve annotations.")
                _write_failed = False
        except Exception:
            pass
    if _write_failed:
        failed.to_csv(failed_path, index=False)

    # 3. Hallucination
    halluc_detail, halluc_agg = compute_hallucinations(df)
    print_hallucination_summary(halluc_agg)
    halluc_agg.to_csv(OUTPUT_DIR / "hallucination_summary.csv", index=False)
    halluc_detail.to_csv(OUTPUT_DIR / "hallucination_detail.csv", index=False)

    # 4. Precision / Recall / F1
    pr = compute_precision_recall(df)
    print_precision_recall(pr)
    pr.to_csv(OUTPUT_DIR / "precision_recall.csv", index=False)

    # 5. Corrected metrics (only when annotation column is present)
    corrected = compute_corrected_metrics(df, pr, acc_summary)
    if corrected is not None:
        print_corrected_metrics(corrected)
        corrected.to_csv(OUTPUT_DIR / "corrected_metrics.csv", index=False)

    # 6. Stability
    stab_overall, stab_by_cat = compute_stability(df)
    corr_stab_overall, corr_stab_by_cat = compute_corrected_stability(df)
    print_stability(stab_overall, stab_by_cat, corr_stab_overall, corr_stab_by_cat)
    stab_out = stab_overall.copy()
    if corr_stab_overall is not None:
        stab_out = stab_out.merge(
            corr_stab_overall[["model_display", "corrected_stable_pairs", "corrected_stability_rate"]],
            on="model_display", how="left",
        )
        stab_out["stability_delta"] = (
            stab_out["corrected_stability_rate"] - stab_out["stability_rate"]
        )
    stab_out.to_csv(OUTPUT_DIR / "stability.csv", index=False)
    stab_by_cat.to_csv(OUTPUT_DIR / "stability_by_category.csv", index=False)

    # 7. Performance
    perf = compute_performance(df)
    print_performance_summary(perf)
    perf.to_csv(OUTPUT_DIR / "performance_metrics.csv", index=False)

    # 7. Category accuracy
    cat_pivot = compute_category_accuracy(df)
    print_category_accuracy(per_run_cat)
    cat_pivot.to_csv(OUTPUT_DIR / "accuracy_by_category.csv", index=False)

    # Combined summary CSV
    combined_summary = (
        acc_summary
        .merge(halluc_agg[["model_display", "halluc_broad_rate",
                            "halluc_pair_rate", "halluc_strict_rate"]],
               on="model_display", how="left")
        .merge(pr[["model_display", "precision", "recall", "f1"]],
               on="model_display", how="left")
        .merge(stab_out[["model_display", "stability_rate"]],
               on="model_display", how="left")
        .merge(perf[["model_display", "avg_latency_ms", "std_latency_ms",
                      "avg_tokens", "avg_tokens_per_sec"]],
               on="model_display", how="left")
    )
    combined_summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)

    print(f"\n{SEP}")
    print(f"  Results written to: {OUTPUT_DIR.resolve()}")
    print(f"  Files:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        print(f"    {f.name}")
    print(SEP)


if __name__ == "__main__":
    main()
