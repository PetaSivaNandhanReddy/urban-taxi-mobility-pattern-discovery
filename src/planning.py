"""
planning.py
===========
Phase 4, Parts 7 & 8 — Transportation planning insights and business
analytics KPIs.

PART 7 — TRANSPORTATION PLANNING INSIGHTS
--------------------------------------------
Every insight generated here is a templated sentence that names its
supporting evidence (a cluster's point/taxi count, a corridor's support
and taxi count, a time segment) and is phrased as something that "may
warrant further investigation" — never a directive like "build a taxi
stand here." No insight is generated without a concrete number from the
mined results backing it. Insights are descriptive hypotheses for a human
planner to investigate further, not conclusions.

PART 8 — KPIs
---------------
A flat, single-row summary table of the headline numbers a business
analytics stakeholder would want at a glance.
"""

from __future__ import annotations

import pandas as pd

from config import COL_TAXI_ID, COL_TIMESTAMP


# ---------------------------------------------------------------------------
# Part 7 — Transportation planning insights
# ---------------------------------------------------------------------------
def generate_hotspot_insights(cluster_summary_df: pd.DataFrame, top_n: int = 3) -> list[dict]:
    """One insight per top-N non-noise cluster, citing its point/taxi evidence."""
    insights = []
    non_noise = cluster_summary_df[cluster_summary_df["cluster_id"] != -1].sort_values(
        "n_points", ascending=False
    ).head(top_n)

    for _, row in non_noise.iterrows():
        insights.append(
            {
                "category": "recurring taxi hotspot",
                "insight": (
                    f"Cluster {int(row['cluster_id'])} (centered near "
                    f"{row['center_lat']:.4f}, {row['center_lon']:.4f}) recorded "
                    f"{int(row['n_points']):,} GPS observations in the sample — "
                    f"among the busiest spatial clusters found. This area may warrant "
                    f"further investigation for taxi pickup infrastructure because of "
                    f"its sustained observed activity volume."
                ),
                "evidence": f"cluster_id={int(row['cluster_id'])}, n_points={int(row['n_points'])}",
            }
        )
    return insights


def generate_corridor_insights(corridor_table: pd.DataFrame, top_n: int = 3) -> list[dict]:
    """One insight per top-N corridor, citing its support/taxi-count evidence."""
    insights = []
    top_corridors = corridor_table.sort_values("support_percentage", ascending=False).head(top_n)

    for _, row in top_corridors.iterrows():
        insights.append(
            {
                "category": "high-activity mobility corridor",
                "insight": (
                    f"The route {row['cluster_path']} recurred with "
                    f"{row['support_percentage']:.1f}% support "
                    f"(observed for {row.get('n_taxis', row.get('support_count', 'multiple'))} taxis). "
                    f"This corridor may warrant further traffic study because it "
                    f"represents a repeated, evidenced movement pattern between "
                    f"distinct activity clusters, not an isolated trip."
                ),
                "evidence": f"cluster_path={row['cluster_path']}, support_pct={row['support_percentage']:.2f}",
            }
        )
    return insights


def generate_temporal_insights(segment_summary: pd.DataFrame, day_type_summary: pd.DataFrame) -> list[dict]:
    """Insights about time-dependent demand, citing the activity-count evidence."""
    insights = []

    if len(segment_summary) and segment_summary["n_points"].sum() > 0:
        peak_segment = segment_summary.loc[segment_summary["n_points"].idxmax()]
        insights.append(
            {
                "category": "time-dependent pickup demand area",
                "insight": (
                    f"The '{peak_segment['time_segment']}' segment accounted for "
                    f"{peak_segment['n_points']:,} of {segment_summary['n_points'].sum():,} "
                    f"GPS observations in the sample "
                    f"({100 * peak_segment['n_points'] / segment_summary['n_points'].sum():.1f}%). "
                    f"Demand-responsive scheduling for this period may warrant further "
                    f"investigation given this concentration of observed activity."
                ),
                "evidence": f"time_segment={peak_segment['time_segment']}, n_points={int(peak_segment['n_points'])}",
            }
        )

    if len(day_type_summary) and day_type_summary["n_points"].sum() > 0:
        wd = day_type_summary.set_index("day_type").get("n_points", pd.Series(dtype=float))
        if "weekday" in wd.index and "weekend" in wd.index and wd.sum() > 0:
            weekday_share = 100 * wd["weekday"] / wd.sum()
            insights.append(
                {
                    "category": "temporal change in mobility demand",
                    "insight": (
                        f"Weekday observations made up {weekday_share:.1f}% of the sample's "
                        f"GPS activity versus {100 - weekday_share:.1f}% on weekends. This "
                        f"differential may warrant further investigation into whether fleet "
                        f"allocation should vary by day type."
                    ),
                    "evidence": f"weekday_pct={weekday_share:.2f}",
                }
            )

    return insights


def compile_planning_insights(
    cluster_summary_df: pd.DataFrame,
    corridor_table: pd.DataFrame,
    segment_summary: pd.DataFrame,
    day_type_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Compile all Part 7 insight categories into one DataFrame."""
    all_insights = []
    all_insights.extend(generate_hotspot_insights(cluster_summary_df))
    if len(corridor_table):
        all_insights.extend(generate_corridor_insights(corridor_table))
    all_insights.extend(generate_temporal_insights(segment_summary, day_type_summary))

    return pd.DataFrame(all_insights)


# ---------------------------------------------------------------------------
# Part 8 — Business analytics KPIs
# ---------------------------------------------------------------------------
def compute_kpis(
    labeled_df: pd.DataFrame,
    cluster_summary_df: pd.DataFrame,
    patterns_df: pd.DataFrame,
    corridor_table: pd.DataFrame,
    segment_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Compute the Part 8 KPI summary as a single-row DataFrame.

    KPIs: total taxis, total GPS points, total spatial cells, total
    spatial clusters, total frequent sequences, top sequence, top
    hotspot, peak mobility hour, most active period, most recurring
    corridor.
    """
    from spatial import COL_CELL_ID

    n_taxis = labeled_df[COL_TAXI_ID].nunique()
    n_points = len(labeled_df)
    n_cells = labeled_df[COL_CELL_ID].nunique() if COL_CELL_ID in labeled_df.columns else None
    n_clusters = int((cluster_summary_df["cluster_id"] != -1).sum())
    n_sequences = len(patterns_df)

    top_sequence = " -> ".join(patterns_df.iloc[0]["pattern"]) if len(patterns_df) else "N/A"
    top_sequence_support = patterns_df.iloc[0]["support_percentage"] if len(patterns_df) else None

    non_noise = cluster_summary_df[cluster_summary_df["cluster_id"] != -1]
    top_hotspot = (
        f"Cluster {int(non_noise.loc[non_noise['n_points'].idxmax(), 'cluster_id'])}"
        if len(non_noise) else "N/A"
    )

    hourly_counts = labeled_df["hour"].value_counts() if "hour" in labeled_df.columns else pd.Series(dtype=int)
    peak_hour = int(hourly_counts.idxmax()) if len(hourly_counts) else None

    most_active_period = (
        segment_summary.loc[segment_summary["n_points"].idxmax(), "time_segment"]
        if len(segment_summary) and segment_summary["n_points"].sum() > 0 else "N/A"
    )

    most_recurring_corridor = (
        corridor_table.sort_values("support_percentage", ascending=False).iloc[0]["cluster_path"]
        if len(corridor_table) else "N/A"
    )

    return pd.DataFrame([{
        "total_taxis_analyzed": n_taxis,
        "total_gps_points": n_points,
        "total_spatial_cells": n_cells,
        "total_spatial_clusters": n_clusters,
        "total_frequent_sequences": n_sequences,
        "top_sequence": top_sequence,
        "top_sequence_support_pct": round(top_sequence_support, 2) if top_sequence_support is not None else None,
        "top_hotspot": top_hotspot,
        "peak_mobility_hour": peak_hour,
        "most_active_period": most_active_period,
        "most_recurring_corridor": most_recurring_corridor,
    }])


if __name__ == "__main__":
    # Quick manual sanity check with a tiny synthetic fixture (clearly
    # labelled as a test fixture, not real T-Drive data).
    cluster_summary_fixture = pd.DataFrame({
        "cluster_id": [-1, 0, 1],
        "n_points": [5, 40, 25],
        "center_lat": [0.0, 39.90, 39.95],
        "center_lon": [0.0, 116.40, 116.45],
    })
    corridor_fixture = pd.DataFrame({
        "cluster_path": ["Cluster 0 -> Cluster 1"],
        "support_percentage": [45.0],
        "n_taxis": [9],
    })
    segment_fixture = pd.DataFrame({
        "time_segment": ["morning", "afternoon", "evening", "night"],
        "n_points": [50, 20, 60, 10],
    })
    day_type_fixture = pd.DataFrame({"day_type": ["weekday", "weekend"], "n_points": [100, 40]})

    print("Generating Part 7 planning insights from a synthetic TEST FIXTURE (not real data):\n")
    insights_df = compile_planning_insights(
        cluster_summary_fixture, corridor_fixture, segment_fixture, day_type_fixture
    )
    for _, row in insights_df.iterrows():
        print(f"[{row['category']}] {row['insight']}\n")

    print("Part 8 KPIs:")
    labeled_fixture = pd.DataFrame({
        COL_TAXI_ID: ["1", "1", "2"],
        "hour": [8, 8, 17],
    })
    patterns_fixture = pd.DataFrame({
        "pattern": [("CELL_A", "CELL_B")],
        "support_percentage": [66.7],
    })
    kpis = compute_kpis(labeled_fixture, cluster_summary_fixture, patterns_fixture, corridor_fixture, segment_fixture)
    print(kpis.T)
