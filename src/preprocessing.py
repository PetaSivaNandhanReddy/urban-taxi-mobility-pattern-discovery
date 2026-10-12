"""
preprocessing.py
=================
Robust cleaning pipeline for raw T-Drive taxi GPS data.

Design principle (per project requirements): report at every stage how
many records are removed, and do NOT aggressively or silently drop unusual
GPS points. "Invalid" here means structurally invalid (missing / malformed
/ physically impossible) — not merely unusual.

Pipeline stages, in order:
  1. Missing taxi_id removal
  2. Missing / malformed timestamp removal (+ timestamp parsing)
  3. Missing latitude/longitude removal
  4. Invalid latitude/longitude removal (outside physically valid ranges)
  5. Out-of-region flagging (Beijing bounding box) — flagged, not deleted,
     unless drop_out_of_region=True is explicitly passed
  6. Duplicate record removal

Each stage's row count is captured in a CleaningReport for full traceability.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from config import (
    BEIJING_LAT_RANGE,
    BEIJING_LON_RANGE,
    COL_LAT,
    COL_LON,
    COL_TAXI_ID,
    COL_TIMESTAMP,
    TDRIVE_TIMESTAMP_FORMAT,
)


import time


@dataclass
class CleaningReport:
    """Tracks record counts, exclusions, and timings through every cleaning stage."""

    stage_counts: dict = field(default_factory=dict)
    stage_durations: dict = field(default_factory=dict)
    exclusions: dict = field(default_factory=dict)
    total_duration_sec: float = 0.0

    def record(self, stage_name: str, n_rows: int, duration_sec: float = 0.0) -> None:
        self.stage_counts[stage_name] = n_rows
        if duration_sec > 0:
            self.stage_durations[stage_name] = round(duration_sec, 4)

    def record_exclusion(self, reason: str, count: int) -> None:
        self.exclusions[reason] = count

    def summary(self) -> str:
        lines = ["Data Cleaning Summary", "=" * 40]
        prev = None
        for stage, count in self.stage_counts.items():
            duration_str = f" ({self.stage_durations.get(stage, 0):.3f}s)" if stage in self.stage_durations else ""
            if prev is None:
                lines.append(f"{stage}: {count:,}{duration_str}")
            else:
                removed = prev - count
                lines.append(f"{stage}: {count:,}  (removed {removed:,}){duration_str}")
            prev = count
        if self.total_duration_sec > 0:
            lines.append(f"Total Cleaning Duration: {self.total_duration_sec:.2f}s")
        return "\n".join(lines)

    def as_dataframe(self) -> pd.DataFrame:
        rows = []
        prev = None
        for stage, count in self.stage_counts.items():
            removed = 0 if prev is None else prev - count
            dur = self.stage_durations.get(stage, 0.0)
            rows.append({
                "stage": stage,
                "records_remaining": count,
                "records_removed": removed,
                "stage_duration_sec": dur,
            })
            prev = count
        return pd.DataFrame(rows)



def clean_taxi_id(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Remove rows with missing/empty taxi_id."""
    df = df.copy()
    df[COL_TAXI_ID] = df[COL_TAXI_ID].astype(str).str.strip()
    df = df[(df[COL_TAXI_ID].notna()) & (df[COL_TAXI_ID] != "") & (df[COL_TAXI_ID] != "nan")]
    report.record("After taxi_id cleaning", len(df))
    return df


def clean_timestamps(
    df: pd.DataFrame,
    report: CleaningReport,
    timestamp_format: str = TDRIVE_TIMESTAMP_FORMAT,
) -> pd.DataFrame:
    """Parse timestamps and drop rows where parsing fails (malformed/missing).

    Tries the documented T-Drive format first, then falls back to pandas'
    general parser for any rows that fail (in case the actual download's
    formatting differs slightly, e.g. no leading zeros).
    """
    df = df.copy()
    parsed = pd.to_datetime(df[COL_TIMESTAMP], format=timestamp_format, errors="coerce")

    still_missing = parsed.isna()
    if still_missing.any():
        fallback_parsed = pd.to_datetime(df.loc[still_missing, COL_TIMESTAMP], errors="coerce")
        parsed.loc[still_missing] = fallback_parsed

    df[COL_TIMESTAMP] = parsed
    df = df[df[COL_TIMESTAMP].notna()]
    report.record("After timestamp cleaning", len(df))
    return df


def clean_coordinates(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Remove rows with missing or physically impossible lat/lon values.

    Physically impossible = outside valid Earth ranges (lat: -90..90,
    lon: -180..180) or exactly (0, 0) (a common GPS null-fix sentinel).
    This does NOT filter to the Beijing region — that is a separate,
    reported (not silently applied) step in flag_out_of_region().
    """
    df = df.copy()
    df[COL_LAT] = pd.to_numeric(df[COL_LAT], errors="coerce")
    df[COL_LON] = pd.to_numeric(df[COL_LON], errors="coerce")

    df = df[df[COL_LAT].notna() & df[COL_LON].notna()]
    df = df[df[COL_LAT].between(-90, 90) & df[COL_LON].between(-180, 180)]
    df = df[~((df[COL_LAT] == 0) & (df[COL_LON] == 0))]

    report.record("After coordinate validity cleaning", len(df))
    return df


def flag_out_of_region(
    df: pd.DataFrame,
    report: CleaningReport,
    lat_range: tuple = BEIJING_LAT_RANGE,
    lon_range: tuple = BEIJING_LON_RANGE,
    drop_out_of_region: bool = False,
) -> pd.DataFrame:
    """Flag (and optionally drop) points outside the expected Beijing region.

    By default this ONLY adds a boolean `in_region` column and reports the
    count — it does not delete anything, per the "do not blindly delete
    unusual GPS points" requirement. Pass drop_out_of_region=True to
    actually remove them after you've reviewed the flagged count.
    """
    df = df.copy()
    df["in_region"] = df[COL_LAT].between(*lat_range) & df[COL_LON].between(*lon_range)

    n_out = int((~df["in_region"]).sum())
    print(f"Flagged {n_out:,} point(s) outside the expected Beijing bounding box "
          f"(lat {lat_range}, lon {lon_range}). Not removed unless drop_out_of_region=True.")

    if drop_out_of_region:
        df = df[df["in_region"]]
        report.record("After out-of-region removal", len(df))
    else:
        report.record("After out-of-region flagging (none dropped)", len(df))

    return df


def clean_duplicates(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Remove exact duplicate (taxi_id, timestamp, lat, lon) records."""
    df = df.copy()
    before = len(df)
    df = df.drop_duplicates(subset=[COL_TAXI_ID, COL_TIMESTAMP, COL_LAT, COL_LON])
    report.record("After duplicate removal", len(df))
    return df


def run_cleaning_pipeline(
    raw_df: pd.DataFrame,
    drop_out_of_region: bool = False,
    verbose: bool = True,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Run the full cleaning pipeline end-to-end and return (clean_df, report).

    This is the single entry point notebooks/scripts should call.
    """
    total_t0 = time.time()
    report = CleaningReport()
    report.record("Original records", len(raw_df))

    t0 = time.time()
    df = clean_taxi_id(raw_df, report)
    report.record("After taxi_id cleaning", len(df), duration_sec=time.time() - t0)

    t0 = time.time()
    df = clean_timestamps(df, report)
    report.record("After timestamp cleaning", len(df), duration_sec=time.time() - t0)

    t0 = time.time()
    df = clean_coordinates(df, report)
    report.record("After coordinate validity cleaning", len(df), duration_sec=time.time() - t0)

    t0 = time.time()
    df = flag_out_of_region(df, report, drop_out_of_region=drop_out_of_region)
    out_of_reg_stage = "After out-of-region removal" if drop_out_of_region else "After out-of-region flagging (none dropped)"
    report.record(out_of_reg_stage, len(df), duration_sec=time.time() - t0)

    t0 = time.time()
    df = clean_duplicates(df, report)
    report.record("After duplicate removal", len(df), duration_sec=time.time() - t0)

    t0 = time.time()
    df = df.sort_values([COL_TAXI_ID, COL_TIMESTAMP]).reset_index(drop=True)
    report.total_duration_sec = round(time.time() - total_t0, 3)
    report.record("Final records", len(df), duration_sec=time.time() - t0)

    if verbose:
        print("\n" + report.summary())

    return df, report


if __name__ == "__main__":
    # Quick manual sanity check with a tiny synthetic fixture (clearly
    # labelled as a test fixture, per project rules — not real T-Drive data).
    test_fixture = pd.DataFrame(
        {
            COL_TAXI_ID: ["1", "1", "1", "", "2", "2"],
            COL_TIMESTAMP: [
                "2008-02-02 15:36:08",
                "2008-02-02 15:36:08",  # exact duplicate of row above
                "not-a-timestamp",
                "2008-02-02 15:37:00",
                "2008-02-02 15:36:08",
                "2008-02-02 15:37:00",
            ],
            COL_LON: [116.51, 116.51, 116.52, 116.53, 0.0, 116.54],
            COL_LAT: [39.92, 39.92, 39.93, 39.94, 0.0, 39.95],
        }
    )
    print("Running cleaning pipeline on a synthetic TEST FIXTURE (not real data):\n")
    cleaned, rpt = run_cleaning_pipeline(test_fixture)
    print("\nCleaned output:")
    print(cleaned)
