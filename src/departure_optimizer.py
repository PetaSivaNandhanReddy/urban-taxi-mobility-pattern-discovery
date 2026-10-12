"""
departure_optimizer.py
======================
Historical departure-time window evaluation — Phase 5.

Evaluates how historical journey durations varied across nearby departure-time windows
(e.g., -15, -10, -5, +5, +10, +15 minutes) for comparable historical trips.

Key Methodological Constraints & Rules:
1. Observational, NOT Causal:
   Calculates differences in observed historical durations across departure windows.
   Does NOT claim that shifting departure time caused a travel-time improvement.
   (Different drivers, road conditions, and passenger circumstances contribute to differences.)
2. Sample-size thresholds:
   Requires a configurable minimum number of historical journeys (min_trips_per_window)
   before computing window statistics. If a window has fewer trips, it is reported as
   "INSUFFICIENT_DATA" rather than computing noisy or misleading medians.
3. Timestamp alignment:
   Matches journeys by time-of-day window (hour:minute) within comparable historical
   calendar periods (weekday vs weekend) in the February 2-8, 2008 T-Drive baseline.
4. No real-time or predictive claims:
   Strictly descriptive historical analysis.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import Any

import numpy as np
import pandas as pd

from config import DEPARTURE_TIME_WINDOWS_MIN, MIN_ROUTE_EVIDENCE_TRIPS


def _filter_journeys_by_tod_window(
    journeys_df: pd.DataFrame,
    target_hour: int,
    target_minute: int,
    half_width_minutes: int = 15,
    day_type_filter: str | None = None,
) -> pd.DataFrame:
    """Filter journeys to those departing within [target_tod - half_width, target_tod + half_width]."""
    if journeys_df.empty:
        return pd.DataFrame()

    work_df = journeys_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(work_df["origin_time"]):
        work_df["origin_time"] = pd.to_datetime(work_df["origin_time"])

    # Optional day-type filter (weekday vs weekend)
    if day_type_filter is not None:
        work_df["_dow"] = work_df["origin_time"].dt.dayofweek
        if day_type_filter == "weekday":
            work_df = work_df[work_df["_dow"] < 5]
        elif day_type_filter == "weekend":
            work_df = work_df[work_df["_dow"] >= 5]

    # Convert departure time to minutes past midnight
    dep_minutes = work_df["origin_time"].dt.hour * 60 + work_df["origin_time"].dt.minute
    target_min_val = target_hour * 60 + target_minute

    # Circular difference for time-of-day (modulo 1440 minutes in a day)
    minute_diff = (dep_minutes - target_min_val).abs()
    circular_diff = np.minimum(minute_diff, 1440 - minute_diff)

    return work_df[circular_diff <= half_width_minutes].reset_index(drop=True)


def evaluate_departure_windows(
    journeys_df: pd.DataFrame,
    target_departure_time: datetime | str,
    window_shifts_minutes: list[int] | None = None,
    half_width_minutes: int = 15,
    min_trips_per_window: int = MIN_ROUTE_EVIDENCE_TRIPS,
    match_day_type: bool = True,
) -> dict[str, Any]:
    """Evaluate historical travel times across nearby departure-time windows.

    Parameters
    ----------
    journeys_df : pd.DataFrame
        DataFrame of historical journeys between an OD pair (from route_candidates.py).
    target_departure_time : datetime or str
        The reference departure datetime (e.g. "2008-02-04 08:30:00").
    window_shifts_minutes : list[int] or None
        List of departure window shifts in minutes (e.g. [-15, -10, -5, 0, 5, 10, 15]).
        Defaults to [-15, -10, -5, 0, 5, 10, 15].
    half_width_minutes : int
        Window radius in minutes around each shifted center.
    min_trips_per_window : int
        Minimum observed journeys required to compute valid statistics for a window.
    match_day_type : bool
        If True, restricts comparison to the same day type (weekday vs weekend) as target_departure_time.

    Returns
    -------
    dict
        Structured departure evaluation report containing:
        - status: "SUFFICIENT_DATA" or "INSUFFICIENT_DATA"
        - target_departure_time: str
        - day_type_evaluated: str
        - base_window_median_sec: float or None
        - windows_summary: pd.DataFrame
        - best_historical_window: dict or None
        - summary: str
        - limitations: list[str]
    """
    if window_shifts_minutes is None:
        shifts = [-15, -10, -5, 0, 5, 10, 15]
    else:
        shifts = sorted(set(window_shifts_minutes))
        if 0 not in shifts:
            shifts.append(0)
            shifts.sort()

    target_dt = pd.to_datetime(target_departure_time)
    day_type = "weekend" if target_dt.dayofweek >= 5 else "weekday"
    day_type_arg = day_type if match_day_type else None

    limitations = [
        "Differences reflect historical associations in the February 2-8, 2008 sample, not causal improvements.",
        "Observed trip counts are based on sampled taxi logs without passenger occupancy confirmation.",
        "Traffic conditions on any specific future day may differ substantially from the 2008 sample.",
        "No live traffic feeds or predictive travel-time algorithms are used.",
    ]

    if journeys_df.empty:
        return {
            "status": "INSUFFICIENT_DATA",
            "target_departure_time": str(target_dt),
            "day_type_evaluated": day_type,
            "base_window_median_sec": None,
            "windows_summary": pd.DataFrame(),
            "best_historical_window": None,
            "summary": "Zero historical journeys available for departure-time window evaluation.",
            "limitations": limitations,
        }

    records = []
    base_median_sec = None

    for shift_min in shifts:
        shifted_time = target_dt + timedelta(minutes=shift_min)
        sh_hour, sh_minute = shifted_time.hour, shifted_time.minute

        matching = _filter_journeys_by_tod_window(
            journeys_df,
            target_hour=sh_hour,
            target_minute=sh_minute,
            half_width_minutes=half_width_minutes,
            day_type_filter=day_type_arg,
        )

        n_trips = len(matching)
        n_taxis = matching["taxi_id"].nunique() if n_trips > 0 else 0

        if n_trips >= min_trips_per_window:
            durs = matching["duration_sec"].values
            med_sec = float(np.median(durs))
            mean_sec = float(np.mean(durs))
            iqr_sec = float(np.percentile(durs, 75) - np.percentile(durs, 25)) if len(durs) > 1 else 0.0
            stat = "SUFFICIENT"
        else:
            med_sec = np.nan
            mean_sec = np.nan
            iqr_sec = np.nan
            stat = "INSUFFICIENT_DATA"

        if shift_min == 0 and not np.isnan(med_sec):
            base_median_sec = med_sec

        records.append({
            "shift_minutes": shift_min,
            "window_center_time": f"{sh_hour:02d}:{sh_minute:02d}",
            "n_historical_trips": n_trips,
            "n_taxis": n_taxis,
            "status": stat,
            "median_duration_sec": med_sec,
            "mean_duration_sec": mean_sec,
            "iqr_duration_sec": iqr_sec,
        })

    windows_df = pd.DataFrame(records)

    # Compute duration difference relative to the base (shift = 0) window
    if base_median_sec is not None:
        windows_df["diff_vs_base_sec"] = (windows_df["median_duration_sec"] - base_median_sec).round(1)
        windows_df["diff_vs_base_pct"] = (
            100.0 * windows_df["diff_vs_base_sec"] / base_median_sec
        ).round(1)
    else:
        windows_df["diff_vs_base_sec"] = np.nan
        windows_df["diff_vs_base_pct"] = np.nan

    valid_windows = windows_df[windows_df["status"] == "SUFFICIENT"].copy()

    if valid_windows.empty:
        return {
            "status": "INSUFFICIENT_DATA",
            "target_departure_time": str(target_dt),
            "day_type_evaluated": day_type,
            "base_window_median_sec": None,
            "windows_summary": windows_df,
            "best_historical_window": None,
            "summary": (
                f"Insufficient historical sample size: None of the tested departure windows "
                f"met the threshold of {min_trips_per_window} journeys."
            ),
            "limitations": limitations,
        }

    best_idx = valid_windows["median_duration_sec"].idxmin()
    best_window = valid_windows.loc[best_idx].to_dict()

    if base_median_sec is not None and best_window["shift_minutes"] != 0:
        saved_sec = base_median_sec - best_window["median_duration_sec"]
        saved_pct = round(100.0 * saved_sec / base_median_sec, 1)
        direction = "earlier" if best_window["shift_minutes"] < 0 else "later"
        abs_shift = abs(best_window["shift_minutes"])
        summary = (
            f"Historical observations show that departures around {best_window['window_center_time']} "
            f"({abs_shift} minutes {direction} than requested {target_dt.strftime('%H:%M')}) "
            f"exhibited an observed median journey duration of {best_window['median_duration_sec']:.0f}s "
            f"across {int(best_window['n_historical_trips'])} historical trips "
            f"({saved_sec:.0f}s / {saved_pct:.1f}% lower than the base window's {base_median_sec:.0f}s). "
            f"Note: This is an observed historical correlation in the Feb 2-8, 2008 sample and does not "
            f"guarantee that departing {abs_shift} minutes {direction} caused the travel-time reduction."
        )
    else:
        summary = (
            f"The requested departure window ({target_dt.strftime('%H:%M')}) had an observed median duration of "
            f"{best_window['median_duration_sec']:.0f}s across {int(best_window['n_historical_trips'])} trips. "
            f"No alternative departure window demonstrated a lower observed median travel time."
        )

    return {
        "status": "SUFFICIENT_DATA",
        "target_departure_time": str(target_dt),
        "day_type_evaluated": day_type,
        "base_window_median_sec": base_median_sec,
        "windows_summary": windows_df,
        "best_historical_window": best_window,
        "summary": summary,
        "limitations": limitations,
    }
