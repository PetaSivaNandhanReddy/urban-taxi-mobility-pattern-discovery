# IEEE Report Assets: Urban Taxi Mobility Pattern Discovery

This directory contains the verified scientific assets, data tables, architectural diagrams, publication-ready figures, and reproducibility manifests prepared for drafting the 8–10 page IEEE conference research paper:

> **"Urban Taxi Mobility Pattern Discovery Using Sequential and Spatial Data Mining"**
> *Target Venue:* IEEE Conference on Intelligent Transportation Systems / Big Data / Spatial Data Mining
> *Dataset:* Microsoft Research T-Drive GPS Trajectory Dataset (Beijing, China; Feb 2–8, 2008)

---

## Directory Organization

```text
report_assets/
├── README.md                               # This documentation and usage guide
├── audit/                                  # Audit reports, inventory, provenance, and literature
│   ├── repository_audit.md                 # Comprehensive scientific and software audit report
│   ├── implementation_inventory.csv        # Detailed method-by-method implementation matrix
│   ├── results_provenance.csv              # Exact traceability of all reported metrics to source CSVs
│   └── literature_matrix.csv               # 15 verified, peer-reviewed references with DOIs and roles
├── manifest/                               # Production manifests and reproducibility checklists
│   ├── figure_manifest.csv                 # Specification of all 14 publication figures (captions, sections)
│   ├── table_manifest.csv                  # Specification of all 8 paper tables (Tables I through VIII)
│   └── reproducibility_manifest.md         # Environment details, exact commands, and runtimes
├── tables/                                 # Publication-ready CSV data tables for IEEE paper
│   ├── table1_dataset_preprocessing.csv    # Table I: Data cleaning stages, record attrition, active fleet
│   ├── table2_algorithm_parameters.csv     # Table II: Algorithmic parameters and operational roles
│   ├── table3_evaluation_metrics_definitions.csv # Table III: Evaluation metrics, math formulas, units
│   ├── table4_clustering_parameters.csv    # Table IV: DBSCAN parameter calibration grid (eps & min_samples)
│   ├── table5_baseline_vs_tuned.csv        # Table V: Empirical reference baseline vs. tuned comparison
│   ├── table6_synthetic_benchmark.csv      # Table VI: Synthetic benchmark planted ground-truth recovery
│   ├── table7_chronological_validation.csv # Table VII: Chronological holdout validation (Feb 6 & Feb 7-8)
│   └── table8_sample_scaling.csv           # Table VIII: Fleet expansion scaling (99, 249, 497 taxis)
├── diagrams/                               # Architectural and UML system diagrams
│   └── source/                             # Editable vector source files
│       ├── fig1_system_architecture.mmd    # Fig. 1: Mermaid end-to-end research architecture
│       ├── fig2_data_preprocessing_workflow.mmd # Fig. 2: Mermaid multi-stage cleaning pipeline
│       ├── fig3_spatial_analysis_workflow.mmd   # Fig. 3: Mermaid spatial grid & DBSCAN workflow
│       ├── fig4_sequential_mining_workflow.mmd  # Fig. 4: Mermaid PrefixSpan & contiguous mining flow
│       ├── fig5_uml_use_case.puml          # Fig. 5: PlantUML actor and operational use cases
│       ├── fig6_uml_activity.puml          # Fig. 6: PlantUML end-to-end pipeline activity model
│       └── fig7_uml_sequence.puml          # Fig. 7: PlantUML modular interaction sequence
└── plots/                                  # Reproducible scientific plots (300 DPI publication PNGs)
    ├── generate_all_plots.py               # Standalone Python script regenerating Figs 8–14
    ├── fig8_clustering_kdistance.png       # Fig. 8: DBSCAN elbow curve and noise sensitivity
    ├── fig9_support_sensitivity.png        # Fig. 9: PrefixSpan support volume vs. empirical metrics
    ├── fig10_grid_sensitivity.png          # Fig. 10: Spatial grid resolution vs. vocabulary and density
    ├── fig11_method_comparison.png         # Fig. 11: Real T-Drive empirical baseline vs. tuned comparison
    ├── fig12_synthetic_recovery.png        # Fig. 12: Synthetic ground-truth planted corridor recovery
    ├── fig13_sample_scaling.png            # Fig. 13: Fleet scaling throughput and execution runtime
    └── fig14_temporal_activity.png         # Fig. 14: Diurnal hourly profiles and time-segment dynamics
```

---

## Guide for Drafting the IEEE Research Paper in Claude

When drafting the 8–10 page IEEE conference research paper, adhere to the following verified principles:

1. **Mapping Assets to Paper Sections:**
   - **Section I (Introduction):** Reference urban transit motivations and cited literature from [`literature_matrix.csv`](audit/literature_matrix.csv).
   - **Section II (Related Work):** Organize the literature matrix into 4 themes: Trajectory Mining, Spatial Clustering, Sequential Mining, and Transportation Applications.
   - **Section III (System Architecture & Preprocessing):** Include **Fig. 1** (Architecture), **Fig. 2** (Preprocessing), and **Table I** (Cleaning attrition).
   - **Section IV (Methodology: Spatial & Sequential Mining):** Include **Fig. 3** (Spatial workflow), **Fig. 4** (Sequential workflow), **Fig. 14** (Temporal activity), **Table II** (Parameters), **Table III** (Metric formulas), and **Table IV** (DBSCAN calibration).
   - **Section V (Experimental Results & Benchmarking):**
     - Empirical Reference: Include **Table V** and **Fig. 11** (Method comparison).
     - Parameter Sensitivity: Include **Fig. 8** (DBSCAN), **Fig. 9** (Support), and **Fig. 10** (Grid size).
     - Synthetic Benchmark: Include **Table VI** and **Fig. 12** (Planted recovery).
     - Chronological Holdout: Include **Table VII** (Validation on Feb 6, Test on Feb 7–8).
   - **Section VI (Computational Scalability & Fleet Expansion):** Include **Table VIII** and **Fig. 13** (Scaling throughput).
   - **Section VII (Discussion & Threats to Validity):** Address the empirical reference limitations and single-week temporal scope.
   - **Section VIII (Conclusion & Future Work):** Conclude with actionable transit planning recommendations.

2. **Strict Scientific Honesty & Terminology:**
   - *Never* describe the empirical reference as "ground truth". Refer to it as the "Empirical Reference Benchmark" derived from 35 observed adjacent transitions ($s \ge 1.5\%$).
   - Point out that the empirical reference favors direct contiguous mining due to its 1-hop nature.
   - Emphasize the **Synthetic Benchmark** as the objective proof that PrefixSpan achieves $100\%$ recall on true planted corridors.
   - Do not claim seasonal or annual temporal generalization; the dataset represents a single week in February 2008.
