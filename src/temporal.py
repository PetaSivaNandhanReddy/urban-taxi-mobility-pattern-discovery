"""
temporal.py
===========
Temporal mobility analysis — Phase 2, Part 8.

Summarizes taxi GPS activity by hour, named time-of-day segment
(morning/afternoon/evening/night), and weekday vs weekend, so activity
patterns over time can be visualized and compared. Assumes
sequential.add_temporal_labels() has already been applied to add `hour`,
`time_segment`, and `day_type` columns.
"""

from __future__ import annotations

import pandas as pd

from config import COL_TAXI_ID, COL_TIMESTAMP, TIME_SEGMENTS


def activity_by_date(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize activity by calendar date, reporting points, active taxis, and points per taxi.

    Ensures visibility into daily observation counts across the 7-day observation period.
    """
    work_df = df.copy()
    if "_date" not in work_df.columns:
        work_df["_date"] = work_df[COL_TIMESTAMP].dt.date

    summary = (
        work_df.groupby("_date")
        .agg(
            n_points=("_date", "count"),
            n_active_taxis=(COL_TAXI_ID, "nunique"),
        )
        .reset_index()
        .rename(columns={"_date": "date"})
    )
    dt_series = pd.to_datetime(summary["date"])
    summary["day_name"] = dt_series.dt.day_name()
    summary["day_type"] = dt_series.dt.dayofweek.map(lambda d: "weekend" if d >= 5 else "weekday")
    summary["points_per_taxi"] = (summary["n_points"] / summary["n_active_taxis"].replace(0, 1)).round(1)
    return summary.sort_values("date").reset_index(drop=True)


def activity_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Count of GPS points and distinct active taxis per hour of day (0-23)."""
    summary = (
        df.groupby("hour")
        .agg(n_points=("hour", "count"), n_active_taxis=(COL_TAXI_ID, "nunique"))
        .reindex(range(24), fill_value=0)
        .reset_index()
    )
    return summary


def activity_by_segment(df: pd.DataFrame) -> pd.DataFrame:
    """Count of GPS points and distinct active taxis per named time segment."""
    summary = (
        df.groupby("time_segment")
        .agg(n_points=("time_segment", "count"), n_active_taxis=(COL_TAXI_ID, "nunique"))
        .reindex(list(TIME_SEGMENTS.keys()), fill_value=0)
        .reset_index()
    )
    return summary


def activity_by_day_type(df: pd.DataFrame, normalize: bool = True) -> pd.DataFrame:
    """Count of GPS points and active taxis for weekday vs weekend, normalized by day count.

    Because T-Drive spans 5 weekdays vs 2 weekend days (and partial first/last days),
    raw counts alone misrepresent weekly intensity. Normalization computes points per
    observed calendar day.
    """
    work_df = df.copy()
    if COL_TIMESTAMP in work_df.columns:
        work_df["_date"] = work_df[COL_TIMESTAMP].dt.date
        dates_per_day_type = work_df.groupby("day_type")["_date"].nunique().to_dict()
    else:
        dates_per_day_type = {"weekday": 5, "weekend": 2}

    summary = (
        work_df.groupby("day_type")
        .agg(n_points=("day_type", "count"), n_active_taxis=(COL_TAXI_ID, "nunique"))
        .reindex(["weekday", "weekend"], fill_value=0)
        .reset_index()
    )

    if normalize:
        summary["n_days_observed"] = summary["day_type"].map(dates_per_day_type).fillna(1).astype(int)
        summary["points_per_day"] = (
            summary["n_points"] / summary["n_days_observed"].replace(0, 1)
        ).round(1)
        summary["points_per_active_taxi_per_day"] = (
            summary["points_per_day"] / summary["n_active_taxis"].replace(0, 1)
        ).round(2)

    return summary



def activity_by_hour_and_day_type(df: pd.DataFrame) -> pd.DataFrame:
    """Hourly activity, split by weekday vs weekend, for a comparison chart."""
    summary = (
        df.groupby(["day_type", "hour"])
        .size()
        .reset_index(name="n_points")
    )
    # Ensure every (day_type, hour) combination exists, even if count is 0
    full_index = pd.MultiIndex.from_product(
        [["weekday", "weekend"], range(24)], names=["day_type", "hour"]
    )
    summary = (
        summary.set_index(["day_type", "hour"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )
    return summary


if __name__ == "__main__":
    # Quick manual sanity check with a tiny synthetic fixture (clearly
    # labelled as a test fixture, not real T-Drive data).
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from sequential import add_temporal_labels

    test_fixture = pd.DataFrame(
        {
            COL_TAXI_ID: ["1", "1", "2", "2", "3"],
            "timestamp": pd.to_datetime(
                [
                    "2008-02-04 08:00:00",  # Monday, morning
                    "2008-02-04 22:00:00",  # Monday, night
                    "2008-02-09 08:00:00",  # Saturday, morning
                    "2008-02-09 14:00:00",  # Saturday, afternoon
                    "2008-02-04 18:00:00",  # Monday, evening
                ]
            ),
        }
    )
    labeled = add_temporal_labels(test_fixture)
    print("Labeled fixture:")
    print(labeled)

    print("\nBy hour:")
    print(activity_by_hour(labeled))
    print("\nBy segment:")
    print(activity_by_segment(labeled))
    print("\nBy day type:")
    print(activity_by_day_type(labeled))
