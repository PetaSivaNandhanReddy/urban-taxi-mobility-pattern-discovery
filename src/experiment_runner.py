"""
experiment_runner.py
====================
Automated, reproducible experiment runner for Phase 2, Phase 3, Phase 4, and Phase 5.

Executes:
1. Real-Data Pair-Level Evaluation on 99-Taxi Baseline (Phase 2)
   - Unified ordered-pair evaluation (PrefixSpan, Direct Contiguous, Co-occurrence)
   - Order-agnostic co-occurrence baseline
   - Explicitly bounded universe accuracy
2. Objective Synthetic Planted Benchmark (Phase 3)
   - Realized support computation from generated finite sample
   - Exact support-count conversion (ceil(min_support * n_sequences))
   - Pair-level and pattern-level recovery against realized ground truth
3. Parameter Sensitivity Experiments (Phase 4)
   - Support-threshold sensitivity (1%, 2%, 5%) on 500m grid
   - Grid-resolution sensitivity (300m, 500m, 800m)
   - Jaccard pattern stability analysis across support thresholds
4. Nested T-Drive Sample Scaling (Phase 5)
   - Deterministic nested samples: Baseline (99 taxis), Medium (250 taxis), Large (500 taxis)
   - File manifest tracking and exclusion reporting
   - Recomputed support counts and cross-sample pattern persistence
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Paths and configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import clustering
import config
import data_loader as dl
import departure_optimizer as do
import pattern_evaluation as pe
import preprocessing as pp
import recommendation as rc
import route_candidates as rcan
import sequential
import spatial
import synthetic_benchmark as sb
import trajectory as tr



def run_synthetic_experiments(output_dir: Path) -> dict[str, Any]:
    """Execute Phase 3 Synthetic Benchmark and export results."""
    print("\n" + "=" * 70)
    print("PHASE 3: RUNNING SYNTHETIC PLANTED BENCHMARK WITH REALIZED SUPPORT")
    print("=" * 70)

    cfg = sb.SyntheticBenchmarkConfig(n_taxis=50, n_days=7, seed=42)
    syn_data = sb.generate_synthetic_dataset(cfg)
    n_seqs = len(syn_data["sequences"])
    print(f"Generated {n_seqs} synthetic sequences (50 taxis x 7 days).")

    syn_results = sb.run_synthetic_evaluation(syn_data, min_support=0.10, max_pattern_length=4)

    planted_df = syn_results["realized_planted_table"]
    pair_df = syn_results["method_comparison_pair_level"]
    pat_df = syn_results["method_comparison_pattern_level"]

    # Export CSVs
    planted_df.to_csv(output_dir / "synthetic_benchmark_planted_table.csv", index=False)
    pair_df.to_csv(output_dir / "synthetic_benchmark_pair_level.csv", index=False)
    pat_df.to_csv(output_dir / "synthetic_benchmark_pattern_level.csv", index=False)

    print("\n[Realized Planted Patterns in Synthetic Sample]")
    print(planted_df[["pattern", "target_injection_prob", "realized_match_count", "realized_support_pct", "is_realized_frequent", "recovered_by_prefixspan"]].to_string(index=False))

    print("\n[Pair-Level Evaluation Metrics on Synthetic Data]")
    print(pair_df[["method", "predicted_pairs", "reference_pairs", "TP", "FP", "FN", "precision", "recall", "f1", "runtime_sec"]].to_string(index=False))

    print("\n[Pattern-Level Evaluation Metrics on Synthetic Data]")
    print(pat_df[["method", "patterns_mined", "reference_patterns", "TP", "FP", "FN", "precision", "recall", "f1", "runtime_sec"]].to_string(index=False))

    return syn_results


def run_baseline_evaluation(
    cleaned_df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    """Execute Phase 2 Real-Data Evaluation on the 99-taxi baseline."""
    print("\n" + "=" * 70)
    print("PHASE 2: RUNNING REAL-DATA EVALUATION ON BASELINE (99 TAXIS)")
    print("=" * 70)

    # 1. Assign 500m grid cells
    cell_df = spatial.assign_spatial_cells(cleaned_df, cell_size_meters=config.GRID_CELL_SIZE_METERS)
    sequences = sequential.build_taxi_sequences(cell_df)
    n_sequences = len(sequences)
    print(f"Constructed {n_sequences} taxi-day sequences from 500m grid.")

    # 2. Build empirical reference set (observed transitions with >= 15 occurrences across >= 2 taxis)
    ref_df = pe.build_reference_od_patterns(sequences, min_occurrence_count=15, min_taxis=2)
    print(f"Empirical reference set constructed: {len(ref_df)} transitions (min_count=15, min_taxis=2).")
    ref_df.to_csv(output_dir / "evaluation_reference_patterns.csv", index=False)

    # 3. Mine PrefixSpan
    t0 = time.time()
    ps_df = sequential.mine_patterns(
        sequences,
        min_support=config.DEFAULT_MIN_SUPPORT,
        max_pattern_length=config.MAX_PATTERN_LENGTH,
    )
    t_ps = time.time() - t0
    print(f"PrefixSpan mined {len(ps_df)} patterns in {t_ps:.3f}s (min_support={config.DEFAULT_MIN_SUPPORT*100}%).")

    # 4. Mine Direct Contiguous Transitions
    t0 = time.time()
    direct_df = sequential.mine_contiguous_patterns(
        sequences,
        min_support=config.DEFAULT_MIN_SUPPORT,
        max_pattern_length=2,
    )
    t_direct = time.time() - t0
    print(f"Direct contiguous mined {len(direct_df)} transitions in {t_direct:.3f}s.")

    # 5. Mine Unordered Co-occurrence
    t0 = time.time()
    cooc_df = pe.mine_cooccurrence_pairs(
        sequences,
        min_support=config.DEFAULT_MIN_SUPPORT,
    )
    t_cooc = time.time() - t0
    print(f"Unordered co-occurrence mined {len(cooc_df)} pairs in {t_cooc:.3f}s.")

    # 6. Evaluate strictly at Pair Level
    summary_ps = pe.summarize_method_consistent_pairs("PrefixSpan (Gapped Sequential)", ps_df, ref_df, t_ps, is_subsequence=True)
    summary_direct = pe.summarize_method_consistent_pairs("Direct Adjacent Transition", direct_df, ref_df, t_direct, is_subsequence=False)
    summary_cooc = pe.summarize_method_consistent_pairs("Unordered Co-occurrence (Symmetric)", cooc_df, ref_df, t_cooc, is_subsequence=True)

    comparison_df = pd.DataFrame([summary_ps, summary_direct, summary_cooc])
    comparison_df.to_csv(output_dir / "evaluation_method_comparison.csv", index=False)

    prec_rec_f1_df = comparison_df[["method", "evaluation_unit", "TP", "FP", "FN", "precision", "recall", "f1"]]
    prec_rec_f1_df.to_csv(output_dir / "evaluation_precision_recall_f1.csv", index=False)

    # 7. Order-agnostic co-occurrence evaluation
    order_agnostic_summary = pe.evaluate_order_agnostic_cooccurrence(cooc_df, ref_df, t_cooc)
    order_agnostic_df = pd.DataFrame([order_agnostic_summary])
    order_agnostic_df.to_csv(output_dir / "evaluation_cooccurrence_order_agnostic.csv", index=False)

    # 8. Bounded universe accuracy
    matched_ps = pe.match_patterns_to_reference(ps_df, ref_df, order_sensitive=True)
    accuracy_info = pe.bounded_universe_accuracy(matched_ps, ref_df, ps_df)
    if accuracy_info:
        with open(output_dir / "evaluation_bounded_accuracy.json", "w") as f:
            json.dump(accuracy_info, f, indent=2)

    print("\n[Corrected Pair-Level Method Comparison]")
    print(comparison_df[["method", "predicted_pairs", "reference_pairs", "TP", "FP", "FN", "precision", "recall", "f1", "runtime_sec"]].to_string(index=False))

    print("\n[Order-Agnostic Co-occurrence Baseline]")
    print(order_agnostic_df[["method", "evaluation_unit", "TP", "FP", "FN", "precision", "recall", "f1"]].to_string(index=False))

    if accuracy_info:
        print(f"\n[Bounded Universe Accuracy]: Universe Size={accuracy_info['universe_size']}, Accuracy={accuracy_info['accuracy']:.4f} (TN Dominance={accuracy_info['tn_dominance_ratio']*100:.2f}%)")

    return {
        "sequences": sequences,
        "ref_df": ref_df,
        "comparison_df": comparison_df,
        "accuracy_info": accuracy_info,
    }


def run_parameter_sensitivity(
    sequences: dict[str, list[str]],
    cleaned_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    """Execute Phase 4: Support Threshold and Grid Resolution Sensitivity."""
    print("\n" + "=" * 70)
    print("PHASE 4: RUNNING PARAMETER SENSITIVITY EXPERIMENTS")
    print("=" * 70)

    n_sequences = len(sequences)

    # 4.1 Support Threshold Sensitivity (1%, 2%, 5%)
    support_thresholds = [0.01, 0.02, 0.05]
    support_rows = []
    pattern_sets_by_support: dict[str, set] = {}

    for sup in support_thresholds:
        min_sup_count = max(2, math.ceil(sup * n_sequences))
        t0 = time.time()
        pats = sequential.mine_patterns(
            sequences,
            min_support=sup,
            max_pattern_length=config.MAX_PATTERN_LENGTH,
        )
        t_run = time.time() - t0

        avg_len = float(pats["pattern_length"].mean()) if len(pats) > 0 else None
        max_len = int(pats["pattern_length"].max()) if len(pats) > 0 else None
        n_unique_cells = len({c for p in pats["pattern"] for c in p}) if len(pats) > 0 else 0

        # Reference agreement metrics
        summary = pe.summarize_method_consistent_pairs(
            f"PrefixSpan ({sup*100:.0f}%)", pats, ref_df, t_run, is_subsequence=True
        )

        support_rows.append({
            "min_support_fraction": sup,
            "min_support_pct": f"{sup*100:.1f}%",
            "min_support_count": min_sup_count,
            "total_sequences": n_sequences,
            "n_patterns": len(pats),
            "runtime_sec": round(t_run, 3),
            "avg_pattern_length": round(avg_len, 2) if avg_len else None,
            "max_pattern_length": max_len,
            "n_unique_cells": n_unique_cells,
            "ref_TP": summary["TP"],
            "ref_FP": summary["FP"],
            "ref_FN": summary["FN"],
            "ref_precision": summary["precision"],
            "ref_recall": summary["recall"],
            "ref_f1": summary["f1"],
        })
        pattern_sets_by_support[f"{sup*100:.0f}%"] = set(pats["pattern"].apply(tuple))

    support_df = pd.DataFrame(support_rows)
    support_df.to_csv(output_dir / "evaluation_support_thresholds.csv", index=False)
    print("\n[Support Threshold Sensitivity Results]")
    print(support_df[["min_support_pct", "min_support_count", "n_patterns", "runtime_sec", "avg_pattern_length", "ref_precision", "ref_recall", "ref_f1"]].to_string(index=False))

    # 4.2 Jaccard Pattern Stability across Support Thresholds
    jaccard_df = pe.pairwise_jaccard_table(pattern_sets_by_support)
    jaccard_df.to_csv(output_dir / "evaluation_pattern_stability.csv", index=False)
    print("\n[Jaccard Stability Across Support Thresholds (Same 500m Grid)]")
    print(jaccard_df[["set_a", "set_b", "n_a", "n_b", "n_intersection", "n_union", "jaccard"]].to_string(index=False))

    # 4.3 Grid Resolution Sensitivity (300m, 500m, 800m)
    grid_sizes = [300.0, 500.0, 800.0]
    grid_rows = []

    for cell_size in grid_sizes:
        t0 = time.time()
        c_df = spatial.assign_spatial_cells(cleaned_df, cell_size_meters=cell_size)
        g_summary = spatial.cell_grid_summary(c_df)
        g_seqs = sequential.build_taxi_sequences(c_df)
        t_seq = time.time() - t0

        t0_mine = time.time()
        g_pats = sequential.mine_patterns(
            g_seqs,
            min_support=config.DEFAULT_MIN_SUPPORT,
            max_pattern_length=config.MAX_PATTERN_LENGTH,
        )
        t_mine = time.time() - t0_mine

        grid_rows.append({
            "grid_cell_size_m": cell_size,
            "n_occupied_cells": len(g_summary),
            "avg_points_per_cell": round(g_summary["n_points"].mean(), 2) if len(g_summary) > 0 else None,
            "n_sequences": len(g_seqs),
            "min_support_count": max(2, math.ceil(config.DEFAULT_MIN_SUPPORT * len(g_seqs))),
            "n_patterns_at_5pct": len(g_pats),
            "avg_pattern_length": round(g_pats["pattern_length"].mean(), 2) if len(g_pats) > 0 else None,
            "max_pattern_length": int(g_pats["pattern_length"].max()) if len(g_pats) > 0 else None,
            "preprocessing_runtime_sec": round(t_seq, 3),
            "mining_runtime_sec": round(t_mine, 3),
            "total_runtime_sec": round(t_seq + t_mine, 3),
        })

    grid_df = pd.DataFrame(grid_rows)
    grid_df.to_csv(output_dir / "evaluation_grid_sizes.csv", index=False)
    print("\n[Grid Resolution Sensitivity Results (300m, 500m, 800m)]")
    print(grid_df[["grid_cell_size_m", "n_occupied_cells", "avg_points_per_cell", "n_sequences", "n_patterns_at_5pct", "total_runtime_sec"]].to_string(index=False))

    return {
        "support_df": support_df,
        "jaccard_df": jaccard_df,
        "grid_df": grid_df,
    }


def run_sample_scaling(output_dir: Path) -> pd.DataFrame:
    """Execute Phase 5: Nested Deterministic T-Drive Sample Scaling (99 -> 250 -> 500)."""
    print("\n" + "=" * 70)
    print("PHASE 5: RUNNING NESTED T-DRIVE SAMPLE EXPANSION (99 -> 250 -> 500 TAXIS)")
    print("=" * 70)

    raw_files = dl.find_taxi_files()
    total_available = len(raw_files)
    print(f"Total T-Drive raw taxi files available: {total_available}")

    # Deterministic Nested Sampling with Fixed Seed 42
    rng = random.Random(42)
    s100_files = rng.sample(raw_files, 100)

    rem1 = [f for f in raw_files if f not in s100_files]
    s250_files = s100_files + rng.sample(rem1, 150)

    rem2 = [f for f in raw_files if f not in s250_files]
    s500_files = s250_files + rng.sample(rem2, 250)

    # Save manifest
    manifest = {
        "seed": 42,
        "total_available_files": total_available,
        "sample_100_files": [f.name for f in s100_files],
        "sample_250_files": [f.name for f in s250_files],
        "sample_500_files": [f.name for f in s500_files],
        "nesting_verified": True,
    }
    with open(output_dir / "sample_expansion_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    schema = dl.detect_schema(s100_files)

    sample_targets = [
        ("Baseline (100 requested)", s100_files),
        ("Medium (250 requested)", s250_files),
        ("Large (500 requested)", s500_files),
    ]

    scaling_rows = []
    pattern_sets_by_sample: dict[str, set] = {}

    for label, files in sample_targets:
        print(f"\nProcessing {label} ({len(files)} files)...")
        t0 = time.time()

        # Load files
        frames = []
        failed_files = []
        for p in files:
            try:
                df = dl._load_single_taxi_file(p, schema["delimiter"], schema["header_detected"])
                frames.append(df)
            except Exception as e:
                failed_files.append((p.name, str(e)))

        raw_df = pd.concat(frames, ignore_index=True)
        raw_points = len(raw_df)

        # Clean
        t_clean0 = time.time()
        cleaned_df, report = pp.run_cleaning_pipeline(raw_df, verbose=False)
        clean_points = len(cleaned_df)
        active_taxis = cleaned_df[config.COL_TAXI_ID].nunique()

        # 500m Grid assignment
        cell_df = spatial.assign_spatial_cells(cleaned_df, cell_size_meters=config.GRID_CELL_SIZE_METERS)
        n_unique_cells = cell_df[spatial.COL_CELL_ID].nunique()

        # Sequences
        seqs = sequential.build_taxi_sequences(cell_df)
        n_seqs = len(seqs)

        # Support threshold at 5% (recomputed exactly for this sample size)
        min_sup_count = max(2, math.ceil(config.DEFAULT_MIN_SUPPORT * n_seqs))

        # Mine PrefixSpan
        t_mine0 = time.time()
        pats = sequential.mine_patterns(
            seqs,
            min_support=config.DEFAULT_MIN_SUPPORT,
            max_pattern_length=config.MAX_PATTERN_LENGTH,
        )
        t_mine = time.time() - t_mine0
        total_time = time.time() - t0

        avg_len = float(pats["pattern_length"].mean()) if len(pats) > 0 else None
        max_len = int(pats["pattern_length"].max()) if len(pats) > 0 else None

        # Build sample reference and compute agreement
        sample_ref = pe.build_reference_od_patterns(seqs, min_occurrence_count=15, min_taxis=2)
        summary = pe.summarize_method_consistent_pairs(label, pats, sample_ref, t_mine, is_subsequence=True)

        scaling_rows.append({
            "sample_label": label,
            "target_taxis": len(files),
            "active_taxis_processed": active_taxis,
            "dropped_taxis_count": len(files) - active_taxis,
            "raw_gps_points": raw_points,
            "cleaned_gps_points": clean_points,
            "points_removed_pct": round(100 * (raw_points - clean_points) / raw_points, 2),
            "n_taxi_day_sequences": n_seqs,
            "unique_occupied_cells_500m": n_unique_cells,
            "min_support_pct": "5.0%",
            "min_support_count_required": min_sup_count,
            "n_discovered_patterns": len(pats),
            "avg_pattern_length": round(avg_len, 2) if avg_len else None,
            "max_pattern_length": max_len,
            "reference_transitions_count": len(sample_ref),
            "ref_TP": summary["TP"],
            "ref_FP": summary["FP"],
            "ref_FN": summary["FN"],
            "ref_precision": summary["precision"],
            "ref_recall": summary["recall"],
            "ref_f1": summary["f1"],
            "total_pipeline_time_sec": round(total_time, 2),
        })

        pattern_sets_by_sample[label] = set(pats["pattern"].apply(tuple))
        print(f"  -> Active taxis: {active_taxis}, GPS points: {clean_points:,}, Sequences: {n_seqs}, Patterns: {len(pats)}, Time: {total_time:.2f}s")

    scaling_df = pd.DataFrame(scaling_rows)
    scaling_df.to_csv(output_dir / "sample_expansion_comparison.csv", index=False)

    # Persistence of patterns across nested samples
    base_pats = pattern_sets_by_sample["Baseline (100 requested)"]
    med_pats = pattern_sets_by_sample["Medium (250 requested)"]
    large_pats = pattern_sets_by_sample["Large (500 requested)"]

    common_all = base_pats & med_pats & large_pats
    common_base_med = base_pats & med_pats

    persistence_info = {
        "baseline_patterns_count": len(base_pats),
        "medium_patterns_count": len(med_pats),
        "large_patterns_count": len(large_pats),
        "common_baseline_and_medium": len(common_base_med),
        "common_all_three_samples": len(common_all),
        "persistence_rate_baseline_in_all_samples_pct": round(100 * len(common_all) / len(base_pats), 2) if len(base_pats) > 0 else 0,
    }
    with open(output_dir / "sample_expansion_persistence.json", "w") as f:
        json.dump(persistence_info, f, indent=2)

    print("\n[Sample Expansion Summary Table]")
    print(scaling_df[["sample_label", "active_taxis_processed", "cleaned_gps_points", "n_taxi_day_sequences", "min_support_count_required", "n_discovered_patterns", "ref_f1", "total_pipeline_time_sec"]].to_string(index=False))

    print(f"\n[Pattern Persistence across Nested Samples]: {len(common_all)} of {len(base_pats)} baseline patterns ({persistence_info['persistence_rate_baseline_in_all_samples_pct']}%) persisted across all 3 sample sizes.")

    return scaling_df


def run_route_and_departure_analytics(
    cleaned_df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    """Execute Phase 5: Historical Route Discovery, Recommendation, and Departure-Time Analytics."""
    print("\n" + "=" * 70)
    print("PHASE 5: HISTORICAL ROUTE, RECOMMENDATION, AND DEPARTURE ANALYTICS")
    print("=" * 70)

    # 1. Assign spatial cells
    cell_df = spatial.assign_spatial_cells(cleaned_df, cell_size_meters=config.GRID_CELL_SIZE_METERS)
    cell_lookup = spatial.build_cell_lookup(cell_df)

    # 2. Segment trajectories into trips
    segmented_df = tr.segment_trajectories(cell_df, max_gap_sec=1800.0, min_points=config.MIN_TRIP_POINTS)
    n_trips = segmented_df["trip_id"].nunique()
    print(f"Segmented trajectories into {n_trips:,} discrete trips/sessions.")

    # 3. Chronological Train/Holdout Evaluation Split
    # Discovery period: Feb 2 - Feb 6, 2008 (5 days)
    # Holdout evaluation period: Feb 7 - Feb 8, 2008 (2 days)
    train_df = segmented_df[segmented_df[config.COL_TIMESTAMP].dt.date <= pd.to_datetime("2008-02-06").date()].copy()
    test_df = segmented_df[segmented_df[config.COL_TIMESTAMP].dt.date > pd.to_datetime("2008-02-06").date()].copy()
    print(f"Chronological split: {len(train_df):,} discovery points (Feb 2-6), {len(test_df):,} holdout points (Feb 7-8).")

    # 4. Discover frequent OD cell pairs on discovery data
    frequent_ods = rcan.find_frequent_od_pairs(train_df, min_trips=3)
    print(f"Discovered {len(frequent_ods)} OD cell pairs with >= 3 historical trips in discovery set.")

    # 5. Evaluate Route Candidates & Recommendations across OD pairs
    rec_rows = []
    dep_rows = []
    holdout_validation_rows = []

    for _, od_row in frequent_ods.head(20).iterrows():
        orig = od_row["origin_cell"]
        dest = od_row["destination_cell"]

        # Route discovery on training data
        cand_result = rcan.discover_route_candidates(
            train_df, orig, dest, min_evidence_trips=config.MIN_ROUTE_EVIDENCE_TRIPS, cell_lookup=cell_lookup
        )
        rec_report = rc.compare_route_alternatives(cand_result, min_evidence_trips=config.MIN_ROUTE_EVIDENCE_TRIPS)

        rec_rows.append({
            "origin_cell": orig,
            "destination_cell": dest,
            "discovery_trips": od_row["trip_count"],
            "status": rec_report["status"],
            "has_recommendation": rec_report["has_recommendation"],
            "recommended_path": rec_report["recommended_route"]["path_str"] if rec_report["recommended_route"] else None,
            "recommended_median_duration_sec": rec_report["recommended_route"]["median_duration_sec"] if rec_report["recommended_route"] else None,
            "n_alternatives": len(rec_report["alternative_routes"]),
            "duration_advantage_sec": rec_report["duration_advantage_sec"],
        })

        # Departure-time window evaluation
        if not cand_result["raw_journeys"].empty:
            sample_time = cand_result["raw_journeys"].iloc[0]["origin_time"]
            dep_report = do.evaluate_departure_windows(
                cand_result["raw_journeys"],
                target_departure_time=sample_time,
                min_trips_per_window=2,
            )
            if dep_report["status"] == "SUFFICIENT_DATA" and dep_report["best_historical_window"]:
                bw = dep_report["best_historical_window"]
                dep_rows.append({
                    "origin_cell": orig,
                    "destination_cell": dest,
                    "target_time": str(sample_time),
                    "base_window_median_sec": dep_report["base_window_median_sec"],
                    "best_window_center": bw["window_center_time"],
                    "best_window_shift_min": bw["shift_minutes"],
                    "best_window_median_sec": bw["median_duration_sec"],
                    "observed_diff_sec": bw["diff_vs_base_sec"],
                })

        # Chronological Holdout Validation:
        test_journeys = rcan.find_od_journeys(test_df, orig, dest)
        if test_journeys and rec_report["recommended_route"]:
            rec_path = rec_report["recommended_route"]["cell_path"]
            matching_test_journeys = [j for j in test_journeys if j["cell_path"] == rec_path]
            for tj in matching_test_journeys:
                observed_dur = tj["duration_sec"]
                predicted_med = rec_report["recommended_route"]["median_duration_sec"]
                ae = abs(observed_dur - predicted_med)
                holdout_validation_rows.append({
                    "origin_cell": orig,
                    "destination_cell": dest,
                    "test_trip_id": tj["trip_id"],
                    "test_origin_time": str(tj["origin_time"]),
                    "predicted_median_sec": predicted_med,
                    "observed_test_duration_sec": observed_dur,
                    "absolute_error_sec": ae,
                    "percentage_error": round(100 * ae / observed_dur, 2) if observed_dur > 0 else 0,
                })

    rec_df = pd.DataFrame(rec_rows)
    dep_df = pd.DataFrame(dep_rows)
    holdout_df = pd.DataFrame(holdout_validation_rows)

    # Coverage metrics
    total_queries = len(rec_df)
    n_recommended = int(rec_df["has_recommendation"].sum()) if not rec_df.empty else 0
    coverage_pct = round(100 * n_recommended / total_queries, 2) if total_queries > 0 else 0.0

    rec_df.to_csv(output_dir / "route_recommendation_evaluation.csv", index=False)
    dep_df.to_csv(output_dir / "departure_window_evaluation.csv", index=False)
    holdout_df.to_csv(output_dir / "chronological_holdout_validation.csv", index=False)

    print(f"\n[Route Recommendation Coverage]: Evaluated {total_queries} OD pairs -> {n_recommended} recommended ({coverage_pct}% coverage).")
    if not holdout_df.empty:
        mae = holdout_df["absolute_error_sec"].mean()
        med_ae = holdout_df["absolute_error_sec"].median()
        print(f"[Chronological Holdout Validation on Unseen Future Days (Feb 7-8)]: MAE = {mae:.1f}s, Median AE = {med_ae:.1f}s across {len(holdout_df)} test journeys.")

    return {
        "rec_df": rec_df,
        "dep_df": dep_df,
        "holdout_df": holdout_df,
        "coverage_pct": coverage_pct,
    }


def main():
    print("=" * 80)
    print("URBAN TAXI MOBILITY: UNIFIED EXPERIMENT RUNNER")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"Project Root: {PROJECT_ROOT}")
    print("=" * 80)

    output_dir = PROJECT_ROOT / "outputs" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run Phase 3 Synthetic Benchmark
    syn_results = run_synthetic_experiments(output_dir)

    # 2. Load 99-taxi baseline cleaned sample
    cleaned_path = PROJECT_ROOT / "data" / "processed" / "cleaned_sample.csv"
    if not cleaned_path.exists():
        print("cleaned_sample.csv not found; loading and cleaning baseline...")
        raw_df = dl.load_tdrive_sample(n_taxis=100, seed=42, verbose=False)
        cleaned_df, _ = pp.run_cleaning_pipeline(raw_df, verbose=False)
    else:
        cleaned_df = pd.read_csv(cleaned_path)
        cleaned_df[config.COL_TIMESTAMP] = pd.to_datetime(cleaned_df[config.COL_TIMESTAMP])

    # 3. Run Phase 2 Baseline Real-Data Evaluation
    base_results = run_baseline_evaluation(cleaned_df, output_dir)

    # 4. Run Phase 4 Parameter Sensitivity
    sens_results = run_parameter_sensitivity(
        base_results["sequences"],
        cleaned_df,
        base_results["ref_df"],
        output_dir,
    )

    # 5. Run Phase 5 Historical Route and Departure-Time Analytics
    route_dep_results = run_route_and_departure_analytics(cleaned_df, output_dir)

    # 6. Run Phase 5 Nested Sample Scaling
    scaling_df = run_sample_scaling(output_dir)

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY! OUTPUTS SAVED TO outputs/results/")
    print("=" * 80)


if __name__ == "__main__":
    main()

