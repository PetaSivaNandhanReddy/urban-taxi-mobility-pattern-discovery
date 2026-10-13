# Visual Asset Quality & Scientific Verification Report

**Project Title:** Urban Taxi Mobility Pattern Discovery Using Sequential and Spatial Data Mining
**Repository:** `PetaSivaNandhanReddy/urban-taxi-mobility-pattern-discovery`
**Branch:** `feature/taxi-mobility-analytics`
**Audit & Review Date:** October 13, 2026
**Auditor / Reviewer:** Research Software Auditor & Scientific Figure Preparation Assistant
**Contact Sheet:** [`report_assets/review/figure_contact_sheet.png`](file:///C:/Users/user/Downloads/urban-taxi-mobility/urban-taxi-mobility/report_assets/review/figure_contact_sheet.png)

---

## 1. Executive Summary & Review Scope

This quality report provides a comprehensive visual and numerical audit of the 14 publication figures (7 architectural/UML vector diagrams and 7 empirical scientific plots) and 8 publication tables prepared for the upcoming 8–10 page IEEE conference research paper.

Every figure and table was inspected for:
1. **Mathematical and Numerical Provenance:** Strict traceability against local raw data and output CSVs in `outputs/`.
2. **Visual & Typographic Legibility:** Verification that font sizes, line weights, contrast, and layout are readable in a two-column IEEE format.
3. **Absence of Distortion or Misleading Claims:** Ensuring the empirical reference is explicitly designated as historical observation agreement rather than independent ground truth, with synthetic benchmark results distinctly isolated.
4. **Architectural Realism:** Ensuring all UML and workflow diagrams reflect actual modules, function signatures, and operational workflows present in the codebase.

---

## 2. Diagram Visual Quality Review (Figs 1–7)

All 7 diagrams have been rendered into high-resolution PNG and vector SVG formats under `report_assets/diagrams/exports/`, with editable source definitions preserved in `report_assets/diagrams/source/`.

| Figure ID & Title | Source File | Render Formats | Checks Performed & Defects Evaluated | Disposition | Reason & Scientific Recommendations |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **Fig. 1: Overall System Architecture** | `fig1_system_architecture.mmd` | PNG, SVG | Evaluated module coverage (`data_loader`, `preprocessing`, `spatial`, `clustering`, `sequential`, `pattern_evaluation`, `dashboard`). Initial vertical layout refactored to balanced 5-stage LR pipeline. No text clipping; distinct categorical color palette. | **APPROVED** | Recommended for Section III (System Architecture) as a full two-column banner figure. |
| **Fig. 2: Data Preprocessing Workflow** | `fig2_data_preprocessing_workflow.mmd` | PNG, SVG | Evaluated against Table I attrition (129,545 raw down to 116,913 clean). Fixed delimiter escaping around coordinate brackets `[-90, 90]`. Refactored from tall vertical stack into 3-phase LR workflow (Ingestion, Cleaning, Trajectory Validation). | **APPROVED** | Recommended for Section III (Dataset & Preprocessing) to visually explain the coordinate purge and duplicate removal. |
| **Fig. 3: Spatial Analysis Workflow** | `fig3_spatial_analysis_workflow.mmd` | PNG, SVG | Checked dual-branch architecture (500m planar grid discretization vs. Haversine DBSCAN clustering). Verified radian conversion formula and hotspot mapping. Refactored into a balanced horizontal flow. | **APPROVED** | Recommended for Section IV (Spatial Discretization & Hotspots). |
| **Fig. 4: Sequential Mining Workflow** | `fig4_sequential_mining_workflow.mmd` | PNG, SVG | Verified taxi-day splitting (`taxi_id::date`, 630 seqs), run-length dwell compression, and dual mining branches (Direct n-gram vs. PrefixSpan). Fixed syntax escaping in text nodes. Checked pair decomposition mapping $(u \to v, u \neq v)$. | **APPROVED** | Recommended for Section IV (Sequential Mining & Evaluation). |
| **Fig. 5: UML Use-Case Model** | `fig5_uml_use_case.puml` | PNG, SVG | Evaluated actor roles (Analyst, Transit Planner, Commuter) against actual dashboard capabilities. Eliminated deprecated `handwritten` warning and resolved wide banner layout via stacked 3-package architecture. Standard UML use-case notation verified. | **APPROVED** | Recommended for Section III (System Architecture & Operational Modes). |
| **Fig. 6: UML Activity Workflow** | `fig6_uml_activity.puml` | PNG, SVG | Traced end-to-end execution path across `experiment_runner.py` and `optimization_study.py`. Verified fork/join nodes for parallel mining, chronological holdout, and synthetic benchmarking. Exported at 300 DPI. | **APPROVED** | Recommended for Section IV (Methodology & Pipeline Execution). |
| **Fig. 7: UML Sequence Diagram** | `fig7_uml_sequence.puml` | PNG, SVG | Verified modular function calls between Dashboard, ExperimentRunner, Preprocessing, Spatial, PrefixSpan, Evaluator, and Storage. Confirmed function-based invocation rather than microservice abstraction. Exported at 300 DPI. | **APPROVED** | Recommended for Section III (Software Architecture & Component Interaction). |

---

## 3. Scientific Empirical Plot Review (Figs 8–14)

All 7 empirical plots are generated at 300 DPI via `report_assets/plots/generate_all_plots.py` reading directly from verified CSV outputs in `outputs/`.

| Figure ID & Title | Source Data File | Plotted Values Audit | Visual & Contrast Checks | Disposition | Reason & Scientific Recommendations |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **Fig. 8: DBSCAN Hotspot Parameter Sensitivity** | `outputs/tables/dbscan_parameter_comparison.csv` | Cross-checked $\epsilon \in [150, 800]\text{ m}$ for $\text{min\_samples}=15$. Selected point $\epsilon=300\text{ m}, N=52$ core clusters, noise=66.26% on sample calibration correctly plotted. | Dual-axis (clusters formed in Navy vs. noise % in Crimson). Elbow line at $\epsilon=300\text{ m}$ clearly indicated. | **APPROVED** | Recommended for Section IV. Caption notes that 52 clusters correspond to the 5,000-point sample calibration run while 237 clusters correspond to the full dataset. |
| **Fig. 9: Support Threshold Sensitivity (Volume vs Metrics)** | `outputs/results/evaluation_support_thresholds.csv` | Verified mined pattern counts (39,785 at 1%, 3,426 at 2%, 22 at 5%) and empirical reference metrics ($P, R, F1$). | Dual-axis: logarithmic pattern count on left; linear ratio $[0.0, 1.0]$ on right. Clearly labeled as "(Empirical Ref)". | **APPROVED** | Recommended for Section V (Sensitivity Analysis) to illustrate the volume vs. precision trade-off. |
| **Fig. 10: Spatial Grid Resolution Sensitivity** | `outputs/results/evaluation_grid_sizes.csv` | Verified occupied cells (8,925 at 300m, 5,413 at 500m, 3,322 at 800m) and point densities (13.1, 21.6, 35.2). | Grouped bar chart with exact counts and densities annotated above bars. High-contrast Navy and Amber color tones. | **APPROVED** | Recommended for Section V (Spatial Sensitivity). |
| **Fig. 11: Empirical Reference: Baseline vs Tuned** | `outputs/results/baseline_vs_best_comparison.csv` | Traced direct baseline ($F1=0.11$), direct tuned ($F1=0.79$), PrefixSpan baseline ($F1=0.40$), and PrefixSpan Top-5 composite ($P=0.75$). | 3-bar metric clusters ($P, R, F1$) with exact values annotated. Title explicitly states "Empirical Reference Benchmark". | **APPROVED** | Recommended for Section V (Empirical Evaluation). Clearly conveys relative improvements while respecting reference caveats. |
| **Fig. 12: Synthetic Ground-Truth Planted Recovery** | `outputs/results/synthetic_benchmark_tuning.csv` | Verified 100% planted recall ($13/13$ pairs) across 8% to 15% support; verified precision increase to 0.5652 at tuned 13% ($F1=0.7222$). | Forest green line at $1.0$ recall; precision in Navy; F1 in Crimson. Tuned 13% support threshold highlighted. | **APPROVED** | Recommended for Section V (Synthetic Benchmark). Provides uncompromised proof of PrefixSpan planted corridor recovery. |
| **Fig. 13: Fleet Expansion Scalability & Runtime** | `outputs/results/sample_expansion_comparison.csv` | Verified points (116.9k, 371.4k, 833.3k) and runtimes (0.97s, 3.96s, 8.19s) across 99, 249, 497 active taxis. | Dual-axis bar chart with data labels. Slate and Teal palette. Sub-10-second processing clearly readable. | **APPROVED** | Recommended for Section VI (Scalability & Fleet Expansion). |
| **Fig. 14: Diurnal Spatiotemporal Mobility Profiles** | `outputs/tables/activity_by_hour.csv` & `outputs/tables/activity_by_segment.csv` | Verified hourly fix volume and active fleet (peak 96 taxis at 14:00); verified segment totals (Morning 24.3k, Afternoon 32.2k, Evening 23.7k, Night 36.7k). | Two-panel figure: (a) Hourly line plot with dual axis; (b) Segment bar chart with thousands formatting. | **APPROVED** | Recommended for Section IV (Spatiotemporal Mobility Dynamics). |

---

## 4. Publication Table Audit Review (Tables I–VIII)

All 8 tables were audited against their respective source files in `outputs/` and verified in `report_assets/tables/`. Full review criteria and findings are documented in [`report_assets/review/table_review.csv`](file:///C:/Users/user/Downloads/urban-taxi-mobility/urban-taxi-mobility/report_assets/review/table_review.csv).

| Table ID | Table Name | Source Output File | Primary Checked Data | Disposition | Notes |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **Table I** | Dataset and Preprocessing Pipeline Summary | `outputs/tables/cleaning_summary.csv` | 129,545 raw down to 116,913 clean (9.75% purge); Taxi 4375 dropped (311 null fixes); 332 out-of-region retained | **APPROVED** | Resolves the 99 vs. 100 active taxi discrepancy. |
| **Table II** | Core Algorithm and Parameter Specification | `src/config.py`, `spatial.py`, `sequential.py` | 500m planar grid, $\epsilon=300\text{ m}, \text{min}=15$, PrefixSpan $s=5\%, L=4$, direct $s=2\%$, weights $0.4/0.3/0.3$ | **APPROVED** | Formalizes algorithmic parameters and roles. |
| **Table III** | Evaluation Metrics and Mathematical Formulations | `src/pattern_evaluation.py` | Mathematical set formulas for $TP, FP, FN, P, R, F1, \Delta F1$; directed pair unit $(u \to v, u \neq v)$; zero guards | **APPROVED** | Provides mathematical rigor for Section IV. |
| **Table IV** | Spatial Clustering Parameter Calibration (DBSCAN) | `outputs/tables/dbscan_parameter_comparison.csv` | 16 parameter runs ($\epsilon \in [150, 800]\text{ m}, \text{min} \in [5, 25]$); clusters formed, noise % | **APPROVED** | Highlights calibrated configuration ($\epsilon=300\text{ m}, \text{min}=15$). |
| **Table V** | Empirical Reference: Baseline vs. Tuned Comparison | `outputs/results/baseline_vs_best_comparison.csv` | Real T-Drive (35 ref pairs): direct tuned $F1=0.7941$ (+634.6%), PrefixSpan Top-5 $P=0.7500$ (+36.4%) | **APPROVED** | Distinguishes empirical reference from ground truth. |
| **Table VI** | Synthetic Benchmark Planted Ground-Truth Recovery | `outputs/results/synthetic_benchmark_tuning.csv` | 5 support levels (8% to 15%); 100% planted recall; tuned $F1=0.7222$ (+97.2% relative gain) | **APPROVED** | Confirms ground-truth planted corridor recovery. |
| **Table VII** | Chronological Holdout Validation & Untouched Recurrence | `outputs/results/chronological_pattern_validation.csv` | Train Feb 2–5 (364 seqs), Val Feb 6 (90 seqs, 39 ref pairs), Test Feb 7–8 (176 seqs, 22 ref pairs) | **APPROVED** | Demonstrates chronological validation without leakage. |
| **Table VIII** | Fleet Expansion Computational Scalability | `outputs/results/sample_expansion_comparison.csv` | Nested cohorts: 99 taxis (0.97s), 249 taxis (3.96s), 497 taxis (8.19s); 833k clean points | **APPROVED** | Provides concrete computational scalability evidence. |

---

## 5. Recommended Asset Selection for 8–10 Page IEEE Paper

Given the typical 8–10 page limit of an IEEE conference paper (two-column format, ~800 words per page):

### High-Priority Core Assets (Essential for Acceptance):
1. **Figures:**
   - **Fig. 1 (System Architecture):** Spans 2 columns at the top of Section III.
   - **Fig. 2 (Preprocessing Workflow):** Spans 1 column in Section III.
   - **Fig. 11 (Baseline vs. Tuned Comparison):** Spans 1 column in Section V.
   - **Fig. 12 (Synthetic Ground-Truth Recovery):** Spans 1 column in Section V.
   - **Fig. 13 (Fleet Expansion Scalability):** Spans 1 column in Section VI.
   - **Fig. 14 (Diurnal Mobility Profiles):** Spans 2 columns in Section IV.
2. **Tables:**
   - **Table I (Dataset Preprocessing & Cleaning):** Section III.
   - **Table II (Algorithm Parameters):** Section IV.
   - **Table V (Empirical Reference Evaluation):** Section V.
   - **Table VI (Synthetic Benchmark Recovery):** Section V.
   - **Table VII (Chronological Holdout Validation):** Section V.
   - **Table VIII (Fleet Expansion Scalability):** Section VI.

### Secondary Supporting Assets (Ideal for 10-Page Expanded Version / Appendices):
- **Fig. 3 (Spatial Workflow)** & **Fig. 4 (Sequential Workflow)**: Section IV.
- **Fig. 5 (UML Use-Case)**, **Fig. 6 (UML Activity)** & **Fig. 7 (UML Sequence)**: Section III (Software Design).
- **Fig. 8 (DBSCAN Sensitivity)**, **Fig. 9 (Support Sensitivity)** & **Fig. 10 (Grid Sensitivity)**: Section V.
- **Table III (Metric Definitions)** & **Table IV (DBSCAN Grid Calibration)**: Section IV.

---

## 6. Review Package Disposition

- **Diagrams (Figs 1–7):** **ALL 7 APPROVED** (rendered in PNG and SVG with balanced layouts).
- **Empirical Plots (Figs 8–14):** **ALL 7 APPROVED** (300 DPI, mathematically validated against source CSVs).
- **Tables (Tables I–VIII):** **ALL 8 APPROVED** (100% numerical provenance verified).
- **Contact Sheet:** Generated and visually verified at `report_assets/review/figure_contact_sheet.png`.
- **Review Table:** Formally documented in `report_assets/review/table_review.csv`.
