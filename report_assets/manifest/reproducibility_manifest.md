# Reproducibility Manifest: Urban Taxi Mobility Pattern Discovery

**Project Title:** Urban Taxi Mobility Pattern Discovery Using Sequential and Spatial Data Mining
**Target Paper:** IEEE Conference Research Paper (8–10 Pages)
**Verification Date:** October 13, 2026
**Auditor:** Research Software Auditor & Scientific Reproducibility Reviewer

---

## 1. System & Execution Environment

The codebase was executed, verified, and audited on the following primary environment:

- **Operating System:** Windows 11 / Windows Server (PowerShell 5.1 / PowerShell 7)
- **Python Runtime:** Python 3.11.x (64-bit)
- **Primary Package Dependencies:**
  - `pandas`: 2.2+
  - `numpy`: 1.26+
  - `scipy`: 1.13+
  - `scikit-learn`: 1.4+
  - `matplotlib`: 3.10+
  - `seaborn`: 0.13+
  - `folium`: 0.16+
  - `streamlit`: 1.35+
  - `pytest`: 8.2+ / 9.1+

To recreate the exact Python virtual environment:
```bash
python -m venv .venv
# Windows PowerShell activation:
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Automated Test Suite Verification

The repository contains 5 modular test suites in `tests/`:
1. `tests/test_preprocessing.py`: Validates coordinate purging, null GPS filtering, out-of-region tagging, and deduplication.
2. `tests/test_spatial.py`: Validates planar grid indexing, coordinate bounding, and cell vocabulary creation.
3. `tests/test_clustering.py`: Validates Haversine metric calculations, radian conversion, and DBSCAN cluster labeling.
4. `tests/test_sequential.py`: Validates PrefixSpan recursive projections, sequence extraction, and direct $n$-gram mining.
5. `tests/test_evaluation.py`: Validates confusion matrix math ($TP, FP, FN$), division-by-zero guards, pair-level mapping, and chronological splitting.

### Execution Command:
```bash
python -m pytest tests/ -v
```

### Verified Test Outcome:
```text
============================== 51 passed in 4.31s ==============================
- tests/test_preprocessing.py: 12 passed
- tests/test_spatial.py: 8 passed
- tests/test_clustering.py: 8 passed
- tests/test_sequential.py: 11 passed
- tests/test_evaluation.py: 12 passed
```
**Status: 100% Pass Rate (0 failures, 0 warnings).**

---

## 3. Dataset Requirements & Local Placement

- **Dataset:** Microsoft Research T-Drive trajectory dataset (Beijing, China; Feb 2–8, 2008).
- **Directory Location:** `data/raw/taxi_log_2008_by_id/`
- **File Format:** One text file per taxi (e.g., `1.txt`, `2.txt`, ..., `10357.txt`).
  - Columns: `taxi_id, timestamp, longitude, latitude` (comma-separated, headerless).
- **Local Fleet Availability:**
  - Full archive on disk: 10,357 taxi text files (~780 MB compressed, ~1.4 GB uncompressed).
  - Baseline sample: 100 sampled driver files (129,545 raw records).
  - Expanded samples: 250 driver files (408,044 raw records) and 500 driver files (907,479 raw records).

---

## 4. End-to-End Pipeline Regeneration Commands

All numerical tables, model results, and publication figures can be regenerated deterministically via the following sequential commands:

### Step 1: Run Full Baseline Analysis Pipeline
Executes data loading, cleaning, spatial grid assignment, Haversine DBSCAN clustering, PrefixSpan sequential mining, contiguous transition mining, and route analytics.
```bash
python src/experiment_runner.py
```
*Outputs generated:*
- `outputs/tables/cleaning_summary.csv`
- `outputs/tables/cluster_summary.csv`
- `outputs/tables/spatial_grid_summary.csv`
- `outputs/results/frequent_patterns.csv`
- `outputs/results/spatial_clusters.csv`
- `outputs/maps/initial_taxi_map.html`

### Step 2: Execute Controlled Optimization & Chronological Validation Study
Executes systematic sensitivity grids, chronological train/validation/test holdout experiments, and baseline vs. best method comparisons.
```bash
python src/optimization_study.py
```
*Outputs generated:*
- `outputs/results/evaluation_support_thresholds.csv`
- `outputs/results/evaluation_grid_sizes.csv`
- `outputs/results/baseline_vs_best_comparison.csv`
- `outputs/results/chronological_holdout_validation.csv`
- `outputs/results/chronological_pattern_validation.csv`
- `outputs/results/sample_expansion_comparison.csv`

### Step 3: Execute Controlled Synthetic Benchmark
Runs the synthetic trajectory generator (Seed = 42, 350 sequences, 5 planted patterns, 13 transitions) and evaluates planted corridor recovery across support thresholds.
```bash
python -m src.synthetic_benchmark
```
*Outputs generated:*
- `outputs/results/synthetic_benchmark_tuning.csv`
- `outputs/results/synthetic_benchmark_pair_level.csv`
- `outputs/results/synthetic_benchmark_pattern_level.csv`
- `outputs/results/synthetic_benchmark_planted_table.csv`

### Step 4: Generate Publication Figures (Figs 8–14)
Executes the standalone, reproducible plotting script to create 300 DPI publication-quality PNG figures from the generated CSVs.
```bash
python report_assets/plots/generate_all_plots.py
```
*Outputs generated in `report_assets/plots/`:*
- `fig8_clustering_kdistance.png` (DBSCAN Parameter Sensitivity)
- `fig9_support_sensitivity.png` (PrefixSpan Support Volume vs. Precision/Recall)
- `fig10_grid_sensitivity.png` (Spatial Discretization Vocabulary & Density)
- `fig11_method_comparison.png` (Empirical Reference: Baseline vs. Tuned)
- `fig12_synthetic_recovery.png` (Synthetic Ground-Truth Planted Recovery)
- `fig13_sample_scaling.png` (Fleet Expansion Runtime & Throughput)
- `fig14_temporal_activity.png` (Hourly & Time-Segment Diurnal Profiles)

### Step 5: Launch Interactive Streamlit Dashboard (Optional Inspection)
Launches the dual-mode interactive analytics dashboard for live map rendering and departure window querying.
```bash
streamlit run dashboard/app.py
```

---

## 5. Artifact Validation Checklist

| Asset | Verification Method | Status |
| :--- | :--- | :---: |
| `tests/` Test Suites | Automated pass via `pytest` (51/51 tests) | Verified |
| Table I (`table1_dataset_preprocessing.csv`) | Cross-checked against raw taxi log parser | Verified |
| Table II (`table2_algorithm_parameters.csv`) | Code inspect of `config.py`, `spatial.py`, `sequential.py` | Verified |
| Table III (`table3_evaluation_metrics_definitions.csv`) | Mathematical formulation checked in `pattern_evaluation.py` | Verified |
| Table IV (`table4_clustering_parameters.csv`) | DBSCAN sensitivity grid verified against source CSV | Verified |
| Table V (`table5_baseline_vs_tuned.csv`) | Provenance traced to `baseline_vs_best_comparison.csv` | Verified |
| Table VI (`table6_synthetic_benchmark.csv`) | Synthetic seed=42 run verified against planted table | Verified |
| Table VII (`table7_chronological_validation.csv`) | Chronological train/val/test split verified | Verified |
| Table VIII (`table8_sample_scaling.csv`) | Multi-fleet runtimes (99, 249, 497 taxis) verified | Verified |
| Figs 1–4 (Mermaid Diagrams) | Valid syntax in `report_assets/diagrams/source/*.mmd` | Verified |
| Figs 5–7 (PlantUML Models) | Valid syntax in `report_assets/diagrams/source/*.puml` | Verified |
| Figs 8–14 (300 DPI PNG Plots) | Generated via `generate_all_plots.py` at 300 DPI | Verified |
