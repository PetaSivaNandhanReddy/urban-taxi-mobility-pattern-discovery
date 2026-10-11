# Urban Taxi Mobility Pattern Discovery for Smart Transportation Planning

**Final-year M.Tech Business Analytics research project.**
Sequential and spatial data mining on urban taxi GPS trajectories, using
the Microsoft Research **T-Drive** dataset as a generic benchmark
(methodology designed to later accept an Indian taxi dataset as well).

> [!NOTE]
> **HISTORICAL ARCHIVE NOTICE**: This file preserves the initial Phase 1 project milestone notes for provenance.
> For the authoritative, complete, and finalized project documentation, see the primary [README.md](README.md) and the comprehensive research report in [reports/final_research_report.md](reports/final_research_report.md).

> **Status: Archived Phase 1 Milestone.** Dataset acquisition, project setup, data
> understanding, loading, cleaning, trajectory reconstruction, EDA, and an
> initial map.
---

## 1. Project Overview

This project builds an end-to-end mobility analytics pipeline that takes
raw taxi GPS trajectories and produces, across all planned phases:

1. Clean taxi trajectories
2. Spatial zones/grid representation
3. Frequent sequential movement patterns
4. Spatial taxi hotspots
5. Temporal mobility patterns
6. Integrated sequential + spatial mobility corridors
7. Transportation-planning insights
8. An interactive Streamlit dashboard

The dataset used to build and validate the system is T-Drive (Beijing).
**This is treated purely as a benchmark** — nothing in the code assumes
Beijing- or China-specific logic beyond a configurable bounding-box sanity
check (see `src/config.py`), so the pipeline can later accept an Indian
taxi GPS dataset with only configuration changes.

---

## 2. Dataset — Microsoft Research T-Drive

- **Source (official):** Microsoft Research T-Drive project page —
  search "Microsoft Research T-Drive trajectory data sample" or check
  `https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/`
  for the current official download links. Microsoft has changed hosting
  for this dataset before, so **verify the link resolves before relying on it.**
- **Content:** ~10,357 taxis, ~15 million GPS points, one week of Beijing
  taxi trajectories.
- **Format:** the official release is **one `.txt` file per taxi**
  (e.g. `1.txt`, `2.txt`, ... `10357.txt`), comma-separated, no header,
  with columns `taxi_id, timestamp, longitude, latitude`. This is
  documented in the T-Drive readme, but this project's loader **verifies
  it against your actual files rather than assuming it blindly**
  (`src/data_loader.py::detect_schema`).
- **Secondary mirror (fallback only):** if the official source is
  inconvenient, the dataset is also commonly mirrored on Kaggle under
  searches like "T-Drive taxi trajectory Beijing" — verify the mirror's
  contents match the official schema before trusting it, since mirrors can
  repackage the data differently.

### 2.1 Where to place the files

1. Download the archive (likely named something like
   `taxi_log_2008_by_id.zip` or similar — **confirm the actual filename
   from your download**, do not assume).
2. Extract it. However it extracts (directly into a folder of `.txt`
   files, or into a nested subfolder), place the **top-level extracted
   folder** under:
   ```
   urban-taxi-mobility/data/raw/
   ```
   The loader searches **recursively** under `data/raw/` for files named
   as plain integers (`1.txt`, `2.txt`, ...), so it does not matter
   whether your files sit directly in `data/raw/` or in
   `data/raw/taxi_log_2008_by_id/`.
3. Verify the extraction:
   ```
   python src/data_loader.py
   ```
   This prints how many taxi files were found, the detected schema, and a
   preview of loaded rows. If it raises `DataLoadError`, the message tells
   you exactly what's missing.

**Raw data must remain untouched** — nothing in this codebase writes to
`data/raw/`. Samples and cleaned data are written to `data/samples/` and
`data/processed/` instead.

---

## 3. Installation

### 3.1 What you must do manually

See **"WHAT I MUST DO MANUALLY"** at the end of this document for the
exact commands (Python install check, virtual environment, package
installation, dataset download).

### 3.2 Opening the project in VS Code

1. Open VS Code.
2. `File > Open Folder...` → select the `urban-taxi-mobility` folder.
3. Open a terminal in VS Code (`` Ctrl+` ``) — it opens already inside the
   project folder.
4. Create and activate a virtual environment, then install dependencies
   (exact commands below in "What I Must Do Manually").
5. Install the Jupyter extension for VS Code if prompted (needed to run
   `notebooks/01_data_exploration.ipynb` inside VS Code). When opening the
   notebook, select the project's virtual environment as the kernel.

---

## 4. Folder Structure

```
urban-taxi-mobility/
├── data/
│   ├── raw/           # Extracted T-Drive files go here (untouched by code)
│   ├── processed/     # Cleaned data, reconstructed trajectories (generated)
│   └── samples/       # Development-sized samples (generated)
├── notebooks/
│   └── 01_data_exploration.ipynb   # Phase 1 EDA notebook
├── src/
│   ├── config.py         # All paths + sampling parameters — edit here
│   ├── data_loader.py    # Flexible T-Drive loader + schema detection
│   ├── preprocessing.py  # Cleaning pipeline with stage-by-stage reporting
│   └── trajectory.py     # Per-taxi trajectory reconstruction + diagnostics
├── outputs/
│   ├── figures/    # EDA charts (PNG)
│   ├── maps/       # Interactive Folium maps (HTML)
│   ├── tables/     # Summary tables (CSV)
│   └── results/    # Reserved for later-phase results (mining, clustering)
├── dashboard/      # Reserved for the Streamlit app (later phase)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 5. Data Format (as standardized by this project)

After loading (`src/data_loader.py`), all data uses these standardized
column names regardless of the raw file's exact header/order:

| Column      | Type              | Description                          |
|-------------|-------------------|---------------------------------------|
| `taxi_id`   | string            | Unique taxi identifier                |
| `timestamp` | string → datetime | Parsed in `preprocessing.py`          |
| `lon`       | float             | Longitude                             |
| `lat`       | float             | Latitude                              |

After trajectory reconstruction (`src/trajectory.py`), these derived
columns are added: `time_diff_sec`, `distance_m`, `speed_kmh`,
`is_suspicious_jump`.

---

## 6. Preprocessing Approach

`src/preprocessing.py` runs a staged pipeline, reporting record counts
removed at every stage (never silent):

1. Remove missing/empty `taxi_id`
2. Parse timestamps; remove rows that fail to parse
3. Remove missing/non-numeric coordinates
4. Remove physically impossible coordinates (outside ±90/±180, or exactly
   `(0, 0)` — a common GPS null-fix sentinel)
5. **Flag** (not delete) points outside the expected Beijing bounding box —
   pass `drop_out_of_region=True` to actually remove them after review
6. Remove exact duplicate `(taxi_id, timestamp, lat, lon)` rows

The full audit trail is saved to `outputs/tables/cleaning_summary.csv`
each time the notebook runs.

---

## 7. How to Run Phase 1

```bash
# from the project root, with your virtual environment activated
python src/data_loader.py        # sanity check: confirms files are found + schema
python src/preprocessing.py      # sanity check: runs cleaning on a tiny synthetic fixture
python src/trajectory.py         # sanity check: runs trajectory reconstruction on a tiny synthetic fixture
```

Then open and run all cells in `notebooks/01_data_exploration.ipynb`
(via VS Code's Jupyter integration, or `jupyter notebook` from the
terminal). It will:

- detect and report the actual raw file schema
- load a configurable sample (`config.SAMPLE_TAXI_COUNT`, default 100 taxis)
- clean it and save the audit trail
- reconstruct trajectories and save diagnostics
- produce 11 EDA charts under `outputs/figures/`
- produce an interactive map under `outputs/maps/`

To change the sample size (e.g. to run on all taxis for a final pass),
edit `SAMPLE_TAXI_COUNT` in `src/config.py`.

---

## 8. Current Progress

**Completed (Phase 1):**
- Project structure, configuration, `.gitignore`
- Dataset acquisition instructions
- Schema-detecting, flexible T-Drive loader with configurable sampling
- Staged, reported data cleaning pipeline
- Trajectory reconstruction with time/distance/speed diagnostics and
  suspicious-jump flagging
- Full EDA notebook (11+ analyses) with saved figures
- Initial interactive map (sampled points + example trajectories)

**Not yet started (future phases — do not implement until explicitly requested):**
- Spatial zone/grid representation
- Frequent sequential pattern mining
- Spatial hotspot detection (DBSCAN)
- Temporal mobility pattern analysis
- Sequential + spatial corridor integration
- Transportation-planning insight generation
- Streamlit dashboard

---

## WHAT I MUST DO MANUALLY

These are the only actions you personally need to perform. Exact commands
are given for Windows (PowerShell, since you're in VS Code on Windows).

### 1. Confirm Python is installed
```powershell
python --version
```
If this fails, install Python 3.10+ from https://www.python.org/downloads/
(check "Add python.exe to PATH" during install).

### 2. Confirm VS Code is installed
Download from https://code.visualstudio.com/ if not already installed.
Install the **Python** and **Jupyter** extensions from the VS Code
Extensions panel.

### 3. Create the project folder and open it in VS Code
```powershell
mkdir urban-taxi-mobility
cd urban-taxi-mobility
code .
```
(All the files Claude created should be placed into this folder — see
"WHAT CLAUDE CREATED" below.)

### 4. Create and activate a virtual environment
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```
If PowerShell blocks the activation script, run PowerShell as Administrator
once and execute `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`,
then retry activation.

### 5. Install dependencies
```powershell
pip install -r requirements.txt
```

### 6. Download the T-Drive dataset
- Go to the official Microsoft Research T-Drive page (search
  "Microsoft Research T-Drive trajectory data sample" — verify the link
  before downloading, hosting has changed over time).
- Download the release archive.
- Note the exact filename you receive — it may not match older
  documentation.

### 7. Extract the archive
- Right-click the downloaded `.zip` → "Extract All..."
- Move the extracted folder (or its contents) into:
  `urban-taxi-mobility\data\raw\`

### 8. Verify the dataset loads correctly
```powershell
python src\data_loader.py
```
Confirm it prints a file count and a preview of loaded rows without
raising `DataLoadError`.

### 9. Run the Phase 1 notebook
Open `notebooks/01_data_exploration.ipynb` in VS Code, select the `venv`
kernel when prompted, and run all cells (`Run All`).

---

## WHAT CLAUDE CREATED

```
urban-taxi-mobility/
├── data/raw/.gitkeep
├── data/processed/.gitkeep
├── data/samples/.gitkeep
├── notebooks/01_data_exploration.ipynb
├── src/config.py
├── src/data_loader.py
├── src/preprocessing.py
├── src/trajectory.py
├── outputs/figures/        (created empty; populated when you run the notebook)
├── outputs/maps/           (created empty; populated when you run the notebook)
├── outputs/tables/         (created empty; populated when you run the notebook)
├── outputs/results/        (reserved for later phases)
├── dashboard/              (reserved for later phases)
├── requirements.txt
├── .gitignore
└── README.md
```

All four Python modules (`config.py`, `data_loader.py`, `preprocessing.py`,
`trajectory.py`) were tested end-to-end against a synthetic multi-taxi
fixture during development (schema detection, missing-value handling,
null-island coordinate rejection, duplicate removal, trajectory speed/
distance calculation, and map generation all verified to run without
errors) — that test fixture is **not** included in your copy of the
project, since Phase 1 must run on your real downloaded data.

---

## PHASE 1 COMPLETION CHECK

Run these in order after downloading and extracting the dataset:

1. **Loader sanity check**
   ```powershell
   python src\data_loader.py
   ```
   ✅ Pass: prints a nonzero file count, a detected schema, and a 10-row preview.
   ❌ Fail: raises `DataLoadError` — re-check `data/raw/` contents.

2. **Preprocessing sanity check**
   ```powershell
   python src\preprocessing.py
   ```
   ✅ Pass: prints a 6-stage cleaning summary and a small cleaned table, using
   the built-in synthetic fixture (no real data needed for this check).

3. **Trajectory sanity check**
   ```powershell
   python src\trajectory.py
   ```
   ✅ Pass: prints a small trajectory table with `speed_kmh` and
   `is_suspicious_jump` columns, and a diagnostics summary.

4. **Full notebook run**
   Open and "Run All" on `notebooks/01_data_exploration.ipynb`.
   ✅ Pass:
   - No cell raises an exception
   - `outputs/figures/` contains 6 PNG files
   - `outputs/maps/initial_taxi_map.html` exists and opens in a browser,
     showing sampled points and colored trajectory lines
   - `outputs/tables/cleaning_summary.csv` and
     `outputs/tables/trajectory_length_stats.csv` exist
   - `data/processed/cleaned_sample.csv` and
     `data/processed/trajectories_sample.csv` exist and are non-empty

5. **Spot-check the numbers make sense**
   - `n_taxis` in the notebook output matches (or is ≤) your configured
     `SAMPLE_TAXI_COUNT`
   - Date range printed falls within the T-Drive collection week
   - Median sampling interval is on the order of tens of seconds to a few
     minutes (T-Drive's typical GPS ping rate) — a wildly different number
     suggests a timestamp parsing issue worth investigating

---

**Do not proceed to sequential mining, spatial clustering (DBSCAN), or the
dashboard until you explicitly request the next phase.**
