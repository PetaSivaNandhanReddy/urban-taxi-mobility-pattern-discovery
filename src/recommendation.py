"""
recommendation.py
=================
Evidence-based alternative-route comparison and recommendation — Phase 5.

This module evaluates competing historical routes between an origin and destination
spatial cell based strictly on empirical trajectory evidence.

Key Methodological Principles:
1. Multi-path requirement: An alternative-route recommendation is only generated
   when at least TWO distinct candidate routes have sufficient supporting historical trips.
2. Robust metrics: Comparisons use median journey duration (to resist outlier pings and
   atypical traffic stalls) and interquartile range (IQR) for journey variability.
3. Strict continuity: Discontinuous routes with large spatial gaps are flagged and
   deprioritized over continuous road-adjacent movements.
4. Insufficient-evidence handling: Returns an explicit "INSUFFICIENT_EVIDENCE" status
   when the dataset cannot justify a defensible comparison.
5. No live traffic claims: Reports observed historical performance from Feb 2-8, 2008 only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config import MIN_ROUTE_EVIDENCE_TRIPS


def compare_route_alternatives(
    candidate_result: dict[str, Any],
    min_evidence_trips: int = MIN_ROUTE_EVIDENCE_TRIPS,
) -> dict[str, Any]:
    """Compare competing historical candidate routes between origin and destination.

    Parameters
    ----------
    candidate_result : dict
        Output from route_candidates.discover_route_candidates().
    min_evidence_trips : int
        Minimum trips required for a route to be eligible for comparison.

    Returns
    -------
    dict
        Structured recommendation report containing:
        - has_recommendation: bool
        - status: "RECOMMENDED", "SINGLE_ROUTE_ONLY", or "INSUFFICIENT_EVIDENCE"
        - origin_cell: str
        - destination_cell: str
        - recommended_route: dict or None
        - alternative_routes: list[dict]
        - duration_advantage_sec: float or None
        - duration_advantage_pct: float or None
        - summary: str
        - limitations: list[str]
    """
    origin = candidate_result.get("origin_cell", "UNKNOWN")
    dest = candidate_result.get("destination_cell", "UNKNOWN")
    candidates_df = candidate_result.get("candidate_routes", pd.DataFrame())

    limitations = [
        "Analysis is based on historical Microsoft T-Drive GPS observations from February 2-8, 2008.",
        "Observed taxi movements reflect fleet behavior, which may differ from general traffic or personal vehicles.",
        "No real-time traffic conditions or live road incidents are incorporated.",
    ]

    if candidates_df.empty or len(candidates_df) == 0:
        return {
            "has_recommendation": False,
            "status": "INSUFFICIENT_EVIDENCE",
            "origin_cell": origin,
            "destination_cell": dest,
            "recommended_route": None,
            "alternative_routes": [],
            "duration_advantage_sec": None,
            "duration_advantage_pct": None,
            "summary": (
                f"Insufficient historical evidence: Zero routes between {origin} and {dest} "
                f"met the threshold of {min_evidence_trips} supporting trips."
            ),
            "limitations": limitations,
        }

    # Filter to physically continuous paths if any exist
    continuous_candidates = candidates_df[candidates_df["is_continuous"]].copy()
    pool = continuous_candidates if not continuous_candidates.empty else candidates_df.copy()

    # Sort by median duration ascending
    sorted_pool = pool.sort_values("median_duration_sec", ascending=True).reset_index(drop=True)

    if len(sorted_pool) == 1:
        single = sorted_pool.iloc[0].to_dict()
        return {
            "has_recommendation": False,
            "status": "SINGLE_ROUTE_ONLY",
            "origin_cell": origin,
            "destination_cell": dest,
            "recommended_route": single,
            "alternative_routes": [],
            "duration_advantage_sec": 0.0,
            "duration_advantage_pct": 0.0,
            "summary": (
                f"Only a single historical path was observed with >= {min_evidence_trips} trips "
                f"(Median duration: {single['median_duration_sec']:.0f}s across {single['n_trips']} trips). "
                f"No competing alternative routes exist in the sample for comparison."
            ),
            "limitations": limitations,
        }

    # At least two competing alternatives exist
    best_route = sorted_pool.iloc[0].to_dict()
    second_best = sorted_pool.iloc[1].to_dict()

    advantage_sec = round(second_best["median_duration_sec"] - best_route["median_duration_sec"], 1)
    advantage_pct = round(
        100.0 * advantage_sec / second_best["median_duration_sec"]
        if second_best["median_duration_sec"] > 0
        else 0.0,
        1,
    )

    alt_list = sorted_pool.iloc[1:].to_dict(orient="records")

    summary = (
        f"Recommended historical route: '{best_route['path_str']}' "
        f"(Median duration: {best_route['median_duration_sec']:.0f}s, IQR: {best_route['iqr_duration_sec']:.0f}s, "
        f"supported by {best_route['n_trips']} trips across {best_route['n_taxis']} taxis). "
        f"It was historically {advantage_sec:.0f}s ({advantage_pct:.1f}%) faster in median travel time "
        f"than the next-best alternative route '{second_best['path_str']}' "
        f"(Median: {second_best['median_duration_sec']:.0f}s, supported by {second_best['n_trips']} trips)."
    )

    return {
        "has_recommendation": True,
        "status": "RECOMMENDED",
        "origin_cell": origin,
        "destination_cell": dest,
        "recommended_route": best_route,
        "alternative_routes": alt_list,
        "duration_advantage_sec": advantage_sec,
        "duration_advantage_pct": advantage_pct,
        "summary": summary,
        "limitations": limitations,
    }
