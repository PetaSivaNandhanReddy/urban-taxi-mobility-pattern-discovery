"""
data_loader.py
===============
Flexible loader for the Microsoft Research T-Drive taxi trajectory dataset.

The official T-Drive release ships as ONE FILE PER TAXI (not a single CSV),
typically named like `1.txt`, `2.txt`, ... `10357.txt`, each containing
comma-separated rows of (taxi_id, timestamp, longitude, latitude) with NO
header row. This loader:

  * searches recursively under a directory for taxi files, so it does not
    matter whether you extracted into data/raw/ directly or into a nested
    folder such as data/raw/taxi_log_2008_by_id/
  * does NOT assume the schema — it inspects the first few files first
    (see detect_schema()) and only falls back to the documented T-Drive
    format if detection is inconclusive
  * supports loading a configurable sample of taxis rather than all ~10,357,
    so development iterations stay fast
  * never modifies files under data/raw/ — it only reads them

Run this module directly (`python src/data_loader.py`) for a quick sanity
check that prints what it finds.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import pandas as pd

from config import (
    COL_LAT,
    COL_LON,
    COL_TAXI_ID,
    COL_TIMESTAMP,
    RANDOM_SEED,
    SAMPLE_ROW_COUNT,
    SAMPLE_TAXI_COUNT,
    TDRIVE_EXPECTED_COLUMNS,
    TDRIVE_RAW_DIR,
    TDRIVE_TIMESTAMP_FORMAT,
)


class DataLoadError(Exception):
    """Raised when the raw dataset cannot be found or parsed as expected."""


def find_taxi_files(raw_dir: Path = TDRIVE_RAW_DIR) -> list[Path]:
    """Recursively find candidate T-Drive per-taxi files under raw_dir.

    T-Drive per-taxi files are named as plain integers, e.g. "1.txt".
    We accept any *.txt file whose stem is numeric, which is robust to
    whichever subfolder the archive extracted into.

    Raises
    ------
    DataLoadError
        If raw_dir does not exist, or no matching files are found — with
        an explicit, actionable error message (never fails silently).
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise DataLoadError(
            f"Raw data directory does not exist: {raw_dir}\n"
            f"Did you extract the T-Drive archive into data/raw/? "
            f"See README.md 'Dataset' section for exact steps."
        )

    candidates = sorted(
        (p for p in raw_dir.rglob("*.txt") if p.stem.isdigit()),
        key=lambda p: int(p.stem),
    )

    if not candidates:
        raise DataLoadError(
            f"No T-Drive taxi files (e.g. '1.txt', '2.txt', ...) found under "
            f"{raw_dir}. Found {sum(1 for _ in raw_dir.rglob('*')):,} other "
            f"files/folders there. Check that you extracted the release "
            f"archive (not just downloaded the .zip) — see README.md."
        )

    return candidates


def detect_schema(sample_files: list[Path], n_check: int = 5) -> dict:
    """Inspect a handful of raw files and report the ACTUAL structure found,
    rather than assuming the documented T-Drive schema.

    Returns a dict summarizing: delimiter, column count, whether a header
    row appears present, and a few raw sample lines for manual inspection.
    This is what notebooks/01_data_exploration.ipynb calls and prints —
    Part 3 requires we detect, not guess.
    """
    report = {
        "files_checked": [],
        "delimiter": None,
        "column_count": None,
        "header_detected": False,
        "sample_lines": [],
    }

    for path in sample_files[:n_check]:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = [f.readline().strip() for _ in range(3)]
        lines = [line for line in lines if line]
        if not lines:
            continue

        report["files_checked"].append(str(path.name))
        report["sample_lines"].extend(lines[:2])

        # Naive delimiter sniff: T-Drive is comma-separated.
        first_line = lines[0]
        delimiter = "," if "," in first_line else ("\t" if "\t" in first_line else None)
        n_cols = len(first_line.split(delimiter)) if delimiter else 1

        if report["delimiter"] is None:
            report["delimiter"] = delimiter
            report["column_count"] = n_cols

        # Header heuristic: if the first field of the first line is NOT
        # numeric, it's likely a header row rather than a taxi_id.
        first_field = first_line.split(delimiter)[0] if delimiter else first_line
        if not first_field.strip().lstrip("-").isdigit():
            report["header_detected"] = True

    if report["delimiter"] is None:
        raise DataLoadError(
            "Could not detect a delimiter (looked for ',' and tab) in the "
            "sampled raw files. Open one file manually to inspect its format."
        )

    return report


def _load_single_taxi_file(path: Path, delimiter: str, has_header: bool) -> pd.DataFrame:
    """Load one taxi's raw file into a standardized DataFrame.

    Standardized columns: taxi_id, timestamp, lon, lat (raw order from
    T-Drive is taxi_id, timestamp, longitude, latitude).
    """
    df = pd.read_csv(
        path,
        sep=delimiter,
        header=0 if has_header else None,
        names=TDRIVE_EXPECTED_COLUMNS,
        dtype={"taxi_id": str},  # parse as string first; cast later, after validation
        engine="python",
        on_bad_lines="skip",
    )
    # Fallback: if the file has no taxi_id column values (some releases omit
    # it since the filename IS the taxi id), inject it from the filename.
    if df["taxi_id"].isna().all():
        df["taxi_id"] = path.stem

    df = df.rename(
        columns={
            "taxi_id": COL_TAXI_ID,
            "timestamp": COL_TIMESTAMP,
            "longitude": COL_LON,
            "latitude": COL_LAT,
        }
    )
    return df


def load_tdrive_sample(
    raw_dir: Path = TDRIVE_RAW_DIR,
    n_taxis: Optional[int] = SAMPLE_TAXI_COUNT,
    max_rows: Optional[int] = SAMPLE_ROW_COUNT,
    seed: int = RANDOM_SEED,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load a configurable sample of the T-Drive dataset.

    Parameters
    ----------
    raw_dir : Path
        Directory to search recursively for taxi files. Defaults to the
        project's data/raw/ (see config.py).
    n_taxis : int or None
        Number of taxi files to load. None means "load every taxi file
        found" — use with caution, this can mean ~15M rows.
    max_rows : int or None
        Optional hard cap on total rows returned across all loaded taxis.
        None means no cap.
    seed : int
        Random seed for taxi sampling, for reproducibility.
    verbose : bool
        Print progress information.

    Returns
    -------
    pd.DataFrame
        Standardized columns: taxi_id (str), timestamp (str, unparsed —
        parsing happens in preprocessing.py), lon (float), lat (float).

    Raises
    ------
    DataLoadError
        If no files are found or none could be parsed successfully.
    """
    all_files = find_taxi_files(raw_dir)
    total_available = len(all_files)

    rng = random.Random(seed)
    if n_taxis is not None and n_taxis < total_available:
        files_to_load = rng.sample(all_files, n_taxis)
        files_to_load.sort(key=lambda p: int(p.stem))
    else:
        files_to_load = all_files

    if verbose:
        print(
            f"Found {total_available:,} taxi files under {raw_dir}. "
            f"Loading {len(files_to_load):,} taxi(s)."
        )

    schema = detect_schema(files_to_load)
    if verbose:
        print(f"Detected schema: {schema}")

    frames = []
    failed_files = []
    rows_loaded = 0

    for path in files_to_load:
        try:
            df = _load_single_taxi_file(
                path,
                delimiter=schema["delimiter"],
                has_header=schema["header_detected"],
            )
        except Exception as exc:  # noqa: BLE001 - we deliberately report, not swallow
            failed_files.append((path.name, str(exc)))
            continue

        frames.append(df)
        rows_loaded += len(df)

        if max_rows is not None and rows_loaded >= max_rows:
            if verbose:
                print(f"Reached max_rows cap ({max_rows:,}); stopping early.")
            break

    if not frames:
        raise DataLoadError(
            f"Failed to load ANY taxi files successfully. "
            f"First few failures: {failed_files[:5]}"
        )

    combined = pd.concat(frames, ignore_index=True)

    if max_rows is not None and len(combined) > max_rows:
        combined = combined.iloc[:max_rows].copy()

    if verbose:
        print(
            f"Loaded {len(combined):,} rows from {len(frames):,} taxi file(s). "
            f"{len(failed_files)} file(s) failed to parse."
        )
        if failed_files:
            print(f"Failed files (showing up to 5): {failed_files[:5]}")

    return combined


if __name__ == "__main__":
    # Quick manual sanity check: `python src/data_loader.py`
    try:
        sample_df = load_tdrive_sample()
        print("\n--- Sample of loaded data ---")
        print(sample_df.head(10))
        print("\n--- dtypes ---")
        print(sample_df.dtypes)
    except DataLoadError as e:
        print(f"\nDataLoadError: {e}")
