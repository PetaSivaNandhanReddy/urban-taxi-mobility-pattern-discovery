"""
app.py
======
Beijing Taxi Mobility Intelligence — Streamlit dashboard.

This dashboard is a PURE VISUALIZATION LAYER over the already-completed
analytical pipeline (Notebooks 01-04). It never re-runs DBSCAN or
PrefixSpan, and never touches raw GPS trajectories. It only reads the
processed CSV tables that the notebooks already wrote to
`output/results/` and `output/tables/`, and caches them with
`@st.cache_data` so switching filters/tabs is instant.

All numbers shown come directly from those CSVs. Where the processed
dataset does not contain a metric (e.g. per-hotspot hourly breakdowns,
or a > 0 corridor count), the dashboard says so explicitly instead of
estimating, interpolating, or inventing a value.

Terminology: this project measures observed HISTORICAL taxi
ACTIVITY/MOBILITY in a fixed GPS sample, not live traffic conditions.
Nothing here is a real-time feed or a prediction.

RUNNING THIS APP
-----------------
    streamlit run dashboard/app.py

DATA LOCATION
-------------
Expects a folder named `output/` (containing `results/` and `tables/`,
exactly as produced by the notebooks) either next to this file or one
level up from it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# ===========================================================================
# PAGE CONFIG
# ===========================================================================
st.set_page_config(
    page_title="Beijing Taxi Mobility Intelligence",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===========================================================================
# THEME / CSS
# ===========================================================================
BG = "#0B0F14"
PANEL = "#131B23"
PANEL_ALT = "#0F161D"
BORDER = "#22303B"
TEXT = "#E7EDF2"
MUTED = "#8CA0AE"
AMBER = "#F2A93B"
AMBER_DIM = "#8A6425"
TEAL = "#2FD1C5"
ROSE = "#EF5B76"

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    color: {TEXT};
}}
.stApp {{
    background: radial-gradient(circle at 15% 0%, #101922 0%, {BG} 45%);
}}
h1, h2, h3 {{
    font-family: 'Space Grotesk', sans-serif !important;
    color: {TEXT} !important;
    letter-spacing: -0.01em;
}}
[data-testid="stSidebar"] {{
    background: {PANEL_ALT};
    border-right: 1px solid {BORDER};
}}
[data-testid="stSidebar"] * {{ color: {TEXT}; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
.stTabs [data-baseweb="tab"] {{
    background: {PANEL};
    border-radius: 8px 8px 0 0;
    color: {MUTED};
    padding: 8px 16px;
}}
.stTabs [aria-selected="true"] {{
    background: {PANEL_ALT} !important;
    color: {AMBER} !important;
    border-bottom: 2px solid {AMBER} !important;
}}
div[data-testid="stMetric"] {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 14px 16px 10px 16px;
}}
div[data-testid="stMetricLabel"] {{ color: {MUTED} !important; }}
div[data-testid="stMetricValue"] {{ color: {AMBER} !important; font-family: 'Space Grotesk', sans-serif; }}

.mob-card {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 10px;
}}
.mob-eyebrow {{
    color: {TEAL};
    font-size: 0.78rem;
    font-weight: 600;
    margin-bottom: 2px;
}}
.mob-banner {{
    border-left: 3px solid {AMBER};
    background: {PANEL};
    padding: 10px 16px;
    border-radius: 6px;
    color: {MUTED};
    font-size: 0.92rem;
    margin-bottom: 16px;
}}
.mob-zero {{
    border: 1px dashed {BORDER};
    border-radius: 12px;
    padding: 28px 20px;
    text-align: center;
    color: {MUTED};
}}
hr {{ border-color: {BORDER} !important; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(color=TEXT, family="Inter"),
        colorway=[AMBER, TEAL, ROSE, "#7C9CBF", "#C97EE0"],
        margin=dict(l=10, r=10, t=45, b=10),
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
)


def style_fig(fig, height=360):
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=height)
    return fig


# ===========================================================================
# DATA LOADING — reads ONLY precomputed pipeline outputs, never raw GPS data
# ===========================================================================
APP_DIR = Path(__file__).resolve().parent
_CANDIDATES = [
    APP_DIR.parent / "outputs",
    APP_DIR / "outputs",
    APP_DIR / "output",
    APP_DIR.parent / "output",
    APP_DIR / "data" / "output",
]
DATA_DIR = next((p for p in _CANDIDATES if p.exists()), _CANDIDATES[0])
RESULTS_DIR = DATA_DIR / "results"
TABLES_DIR = DATA_DIR / "tables"


@st.cache_data(show_spinner=False)
def _read_csv(path_str: str):
    p = Path(path_str)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        return df if len(df.columns) > 0 else None
    except pd.errors.EmptyDataError:
        return None
    except Exception:
        return None


@st.cache_data(show_spinner="Loading processed pipeline outputs...")
def load_all():
    def r(name):
        return _read_csv(str(RESULTS_DIR / name))

    def t(name):
        return _read_csv(str(TABLES_DIR / name))

    data = {
        "kpis": r("final_results.csv"),
        "pipeline_params": r("phase2_research_metrics.csv"),
        "cluster_summary": t("cluster_summary.csv"),
        "grid_summary": t("spatial_grid_summary.csv"),
        "activity_hour": t("activity_by_hour.csv"),
        "activity_segment": t("activity_by_segment.csv"),
        "activity_day_type": t("activity_by_day_type.csv"),
        "ranked_patterns": t("nb04_ranked_patterns.csv"),
        "patterns_by_segment": r("patterns_by_time_segment.csv"),
        "patterns_by_day_type": r("patterns_by_day_type.csv"),
        "weekday_weekend_common": r("weekday_weekend_common_patterns.csv"),
        "corridor_table": t("nb04_corridor_table.csv"),
        "rare_transitions": t("nb04_rare_transitions.csv"),
        "cleaning_summary": t("cleaning_summary.csv"),
        "trajectory_stats": t("trajectory_length_stats.csv"),
        "planning_insights": r("transportation_planning_insights.csv"),
        "method_comparison": r("evaluation_method_comparison.csv"),
        "precision_recall_f1": r("evaluation_precision_recall_f1.csv"),
        "cooccurrence_order_agnostic": r("evaluation_cooccurrence_order_agnostic.csv"),
        "evaluation_summary": r("evaluation_summary.csv"),
        "support_sensitivity": r("evaluation_support_thresholds.csv"),
        "grid_sensitivity": r("evaluation_grid_sizes.csv"),
        "pattern_stability": r("evaluation_pattern_stability.csv"),
        "synthetic_planted": r("synthetic_benchmark_planted_table.csv"),
        "synthetic_pair_eval": r("synthetic_benchmark_pair_level.csv"),
        "synthetic_pattern_eval": r("synthetic_benchmark_pattern_level.csv"),
        "sample_scaling": r("sample_expansion_comparison.csv"),
    }
    return data


DATA = load_all()


def available(key: str) -> bool:
    df = DATA.get(key)
    return df is not None and len(df) > 0


def not_available(msg="Not available in the current processed dataset."):
    st.caption(f"ⓘ {msg}")


# ---------------------------------------------------------------------------
# small formatting helpers
# ---------------------------------------------------------------------------
def fmt_int(x) -> str:
    try:
        return f"{int(x):,}"
    except (ValueError, TypeError):
        return "—"


def fmt_pct(x, dp=2) -> str:
    try:
        return f"{float(x):.{dp}f}%"
    except (ValueError, TypeError):
        return "—"


def format_pattern(raw) -> str:
    """Normalize both storage styles pipeline CSVs use for patterns:
    "('CELL_01', 'CELL_02')" (tuple-string) and "CELL_01 -> CELL_02"
    (already arrow-joined) into one consistent 'CELL_01 → CELL_02' form.
    """
    s = str(raw)
    if "->" in s:
        return s.replace("->", "→")
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (tuple, list)):
            return " → ".join(str(x) for x in parsed)
    except (ValueError, SyntaxError):
        pass
    return s


def kpi_row(items):
    """items: list of (label, value, help) tuples, rendered as st.metric grid."""
    cols = st.columns(len(items))
    for col, (label, value, help_text) in zip(cols, items):
        col.metric(label, value, help=help_text or None)


# ===========================================================================
# MAP BUILDER (shared by Administrator + Customer modes)
# ===========================================================================
def build_beijing_map(highlight_cluster_id=None, show_markers=True, show_heat=True):
    cs = DATA.get("cluster_summary")
    grid = DATA.get("grid_summary")

    if cs is None or len(cs) == 0:
        return None

    non_noise = cs[cs["is_noise"] == False].copy()  # noqa: E712
    if len(non_noise) == 0:
        return None

    center_lat = non_noise["center_lat"].median()
    center_lon = non_noise["center_lon"].median()

    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles="cartodbdark_matter",
        control_scale=True,
    )

    max_pts = max(non_noise["n_points"].max(), 1)
    non_noise = non_noise.sort_values("n_points", ascending=False).reset_index(drop=True)
    top_ids = set(non_noise.head(10)["cluster_id"])

    if show_markers:
        marker_layer = folium.FeatureGroup(name="Hotspot clusters (markers)", show=True)
        for rank, row in non_noise.iterrows():
            cid = int(row["cluster_id"])
            share = 100.0 * row["n_points"] / non_noise["n_points"].sum()
            is_selected = highlight_cluster_id is not None and cid == highlight_cluster_id
            color = AMBER if cid in top_ids else TEAL
            if is_selected:
                color = ROSE
            radius = 6 + 16 * np.sqrt(row["n_points"] / max_pts)
            popup_html = (
                f"<b>Hotspot cluster {cid}</b><br>"
                f"Rank: #{rank + 1} of {len(non_noise)} by GPS-point volume<br>"
                f"GPS points observed: {int(row['n_points']):,}<br>"
                f"Share of all clustered activity: {share:.2f}%<br>"
                f"Center: {row['center_lat']:.4f}, {row['center_lon']:.4f}"
            )
            folium.CircleMarker(
                location=[row["center_lat"], row["center_lon"]],
                radius=float(radius) * (1.3 if is_selected else 1.0),
                color=color,
                weight=2 if is_selected else 1,
                fill=True,
                fill_color=color,
                fill_opacity=0.75 if is_selected else 0.55,
                popup=folium.Popup(popup_html, max_width=280),
                tooltip=f"Cluster {cid} — {int(row['n_points']):,} pts",
            ).add_to(marker_layer)
        marker_layer.add_to(fmap)

    if show_heat and grid is not None and len(grid) > 0:
        heat_layer = folium.FeatureGroup(name="Activity intensity (heatmap)", show=True)
        heat_data = grid[["centroid_lat", "centroid_lon", "n_points"]].values.tolist()
        HeatMap(heat_data, radius=14, blur=20, max_zoom=13, min_opacity=0.25).add_to(heat_layer)
        heat_layer.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)

    legend_html = f"""
    <div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999;
        background: {PANEL}; border: 1px solid {BORDER}; border-radius: 10px;
        padding: 10px 14px; color: {TEXT}; font-family: Inter, sans-serif; font-size: 12px;">
      <div style="font-weight:600; margin-bottom:6px;">Legend</div>
      <div><span style="color:{AMBER};">●</span> Top-10 hotspot cluster (by GPS points)</div>
      <div><span style="color:{TEAL};">●</span> Other hotspot cluster</div>
      <div><span style="color:{ROSE};">●</span> Selected hotspot</div>
      <div style="margin-top:4px; color:{MUTED};">Marker size ∝ √(observed GPS points)</div>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))
    return fmap


# ===========================================================================
# SIDEBAR — mode + navigation
# ===========================================================================
st.sidebar.markdown("### 🚕 Beijing Taxi Mobility")
mode = st.sidebar.radio("Dashboard mode", ["Administrator", "Customer"], index=0)
st.sidebar.divider()

ADMIN_SECTIONS = [
    "Overview",
    "Beijing Mobility Map",
    "Temporal Intelligence",
    "Sequential Patterns",
    "Mobility Corridors",
    "Taxi Allocation",
    "Rare Mobility",
    "Methodology & Evaluation",
    "Export",
]
CUSTOMER_SECTIONS = [
    "Where is activity high?",
    "When is activity high?",
    "Historical patterns",
    "Hotspot lookup",
    "Early-departure guidance",
]

if mode == "Administrator":
    section = st.sidebar.radio("Section", ADMIN_SECTIONS, index=0)
else:
    section = st.sidebar.radio("Section", CUSTOMER_SECTIONS, index=0)

st.sidebar.divider()
st.sidebar.markdown(
    f"""
    <div style="background:{PANEL}; border:1px solid {BORDER}; border-radius:8px; padding:10px 12px; margin-bottom:12px; font-size:0.78rem;">
      <div style="color:{TEAL}; font-weight:600; margin-bottom:4px;">RUN CONFIGURATION</div>
      <div>Grid Size: <b>500 m</b></div>
      <div>Min Support: <b>5.0% (≥32 taxi-days)</b></div>
      <div>Sample: <b>99 Taxis (630 Sequences)</b></div>
      <div>Period: <b>2008-02-02 to 2008-02-08</b></div>
      <div style="color:{MUTED}; margin-top:4px;">Historical Trajectory Analysis</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.caption(
    "All results are computed once by the offline analytical pipeline "
    "(Notebooks 01–05) and read here from cached CSV tables — this "
    "dashboard never re-runs DBSCAN or PrefixSpan."
)
if DATA.get("kpis") is not None:
    st.sidebar.caption(f"Data source: `{DATA_DIR}`")

# ===========================================================================
# HEADER
# ===========================================================================
st.title("Beijing Taxi Mobility Intelligence")
st.caption("Urban Mobility Pattern Discovery from T-Drive Trajectories")
st.markdown(
    f"""<div class="mob-banner">All insights on this dashboard describe
    historical taxi <b>activity / mobility patterns</b> observed in a fixed
    GPS sample — not live traffic conditions. Nothing here is real-time,
    and nothing is a guaranteed prediction.</div>""",
    unsafe_allow_html=True,
)

kpis = DATA.get("kpis")
params = DATA.get("pipeline_params")
ranked = DATA.get("ranked_patterns")

total_taxi_days = None
active_dates = None
if ranked is not None and "total_taxi_days" in ranked.columns and len(ranked) > 0:
    total_taxi_days = int(ranked["total_taxi_days"].iloc[0])
if ranked is not None and "total_distinct_dates" in ranked.columns and len(ranked) > 0:
    active_dates = int(ranked["total_distinct_dates"].iloc[0])

n_corridors = 0
if params is not None and "n_hotspot_to_hotspot_corridors" in params.columns:
    n_corridors = int(params["n_hotspot_to_hotspot_corridors"].iloc[0])

CORRIDOR_ZERO_MSG = (
    "No frequent hotspot-to-hotspot corridors met the current support "
    "and multi-hotspot criteria."
)

# ===========================================================================
# ADMINISTRATOR MODE
# ===========================================================================
if mode == "Administrator":

    # ------------------------------------------------------------------
    if section == "Overview":
        if kpis is None or len(kpis) == 0:
            st.error("KPI table (`results/final_results.csv`) not found.")
        else:
            k = kpis.iloc[0]
            row1 = [
                ("Total taxis", fmt_int(k.get("total_taxis_analyzed")), "Distinct taxis analyzed in the sample"),
                ("Taxi-days", fmt_int(total_taxi_days) if total_taxi_days else "—",
                 "Taxi-day sequences used for sequential pattern mining"),
                ("GPS points", fmt_int(k.get("total_gps_points")), "Cleaned GPS observations analyzed"),
                ("Active dates", fmt_int(active_dates) if active_dates else "—", "Distinct calendar dates covered"),
            ]
            row2 = [
                ("Hotspots", fmt_int(k.get("total_spatial_clusters")), "Non-noise DBSCAN spatial clusters"),
                ("Frequent patterns", fmt_int(k.get("total_frequent_sequences")), "PrefixSpan patterns above the mining support threshold"),
                ("Mobility corridors", fmt_int(n_corridors), "Frequent sequences spanning 2+ distinct hotspots"),
                ("Peak activity hour", f"{fmt_int(k.get('peak_mobility_hour'))}:00", "Hour of day with the most GPS observations"),
            ]
            kpi_row(row1)
            st.write("")
            kpi_row(row2)

            if available("method_comparison"):
                mc = DATA["method_comparison"]
                ps_rows = mc[mc["method"].str.contains("PrefixSpan")]
                if len(ps_rows) > 0:
                    psr = ps_rows.iloc[0]
                    row3 = [
                        ("PrefixSpan Precision", fmt_pct(float(psr.get("precision", 0)) * 100, 1), "TP / (TP + FP) evaluated on distinct ordered cell pairs"),
                        ("PrefixSpan Recall", fmt_pct(float(psr.get("recall", 0)) * 100, 1), "TP / (TP + FN) evaluated on distinct ordered cell pairs"),
                        ("PrefixSpan F1 Score", fmt_pct(float(psr.get("f1", 0)) * 100, 1), "Harmonic mean of pair precision & recall"),
                        ("Reference Transitions", fmt_int(psr.get("reference_pairs")), "Observed recurring transitions meeting reference criteria"),
                    ]
                    st.write("")
                    kpi_row(row3)

            st.write("")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(
                    f"""<div class="mob-card"><div class="mob-eyebrow">TOP SEQUENCE</div>
                    <div style="font-size:1.1rem;">{format_pattern(k.get('top_sequence'))}</div>
                    <div style="color:{MUTED}; font-size:0.85rem; margin-top:4px;">
                    support {fmt_pct(k.get('top_sequence_support_pct'), 2)}</div></div>""",
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f"""<div class="mob-card"><div class="mob-eyebrow">TOP HOTSPOT / MOST ACTIVE PERIOD LABEL</div>
                    <div style="font-size:1.1rem;">{k.get('top_hotspot')} · {k.get('most_active_period')}</div>
                    <div style="color:{MUTED}; font-size:0.85rem; margin-top:4px;">
                    as reported by the pipeline's KPI summary (`final_results.csv`)</div></div>""",
                    unsafe_allow_html=True,
                )

            if params is not None and len(params) > 0:
                p = params.iloc[0]
                with st.expander("Pipeline parameters used to produce these results"):
                    st.write(
                        f"Grid cell size: **{p.get('grid_cell_size_m')} m** · "
                        f"DBSCAN eps: **{p.get('dbscan_eps_m')} m**, min_samples: **{int(p.get('dbscan_min_samples'))}** · "
                        f"Sequence mining support threshold: **{p.get('min_support_used')}** "
                        f"(method: `{p.get('mining_method')}`)"
                    )

            if available("cleaning_summary"):
                with st.expander("Data provenance — cleaning pipeline (Notebook 01)"):
                    st.dataframe(DATA["cleaning_summary"], hide_index=True, width="stretch")

            if available("planning_insights"):
                with st.expander("Planning insights generated by the pipeline (Notebook 04)"):
                    st.caption("Descriptive observations from the analysis — not live recommendations.")
                    st.dataframe(
                        DATA["planning_insights"][["category", "insight", "evidence"]],
                        hide_index=True, width="stretch",
                    )

    # ------------------------------------------------------------------
    elif section == "Beijing Mobility Map":
        st.subheader("Beijing Mobility Map")
        st.caption(
            "Hotspot clusters are DBSCAN-derived spatial clusters; the heatmap layer "
            "shows GPS-observation density across all spatial grid cells. Toggle layers "
            "top-right."
        )
        layer_cols = st.columns(2)
        show_markers = layer_cols[0].checkbox("Show hotspot markers", value=True)
        show_heat = layer_cols[1].checkbox("Show activity heatmap", value=True)

        fmap = build_beijing_map(show_markers=show_markers, show_heat=show_heat)
        if fmap is None:
            st.warning("No hotspot cluster data available to render the map.")
        else:
            st_folium(fmap, width=None, height=560, returned_objects=[])
        not_available(
            "Hour-of-day filtering on this map is not available — the processed "
            "dataset does not include a per-hotspot / per-cell hourly breakdown. "
            "See 'Temporal Intelligence' for hour-of-day patterns aggregated "
            "across all locations."
        )

        if available("cluster_summary"):
            with st.expander("Full hotspot cluster table"):
                cs = DATA["cluster_summary"]
                cs_show = cs[cs["is_noise"] == False].sort_values("n_points", ascending=False)  # noqa: E712
                st.dataframe(
                    cs_show[["cluster_id", "n_points", "center_lat", "center_lon"]]
                    .rename(columns={"cluster_id": "Cluster", "n_points": "GPS points",
                                      "center_lat": "Center lat", "center_lon": "Center lon"}),
                    hide_index=True, width="stretch",
                )

    # ------------------------------------------------------------------
    elif section == "Temporal Intelligence":
        st.subheader("Temporal Intelligence")

        if available("activity_hour"):
            hr = DATA["activity_hour"].sort_values("hour")
            fig = px.bar(
                hr, x="hour", y="n_points", color="n_active_taxis",
                labels={"hour": "Hour of day", "n_points": "GPS points", "n_active_taxis": "Active taxis"},
                title="Hourly taxi activity",
                color_continuous_scale=[TEAL, AMBER],
            )
            st.plotly_chart(style_fig(fig), width="stretch")
        else:
            not_available("Hourly activity table not found.")

        c1, c2 = st.columns(2)
        with c1:
            if available("activity_day_type"):
                dtdf = DATA["activity_day_type"]
                fig = px.bar(
                    dtdf, x="day_type", y="n_points", color="day_type",
                    labels={"day_type": "Day type", "n_points": "GPS points"},
                    title="Weekday vs weekend activity",
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(style_fig(fig, 320), width="stretch")
            else:
                not_available()
        with c2:
            if available("activity_segment"):
                seg = DATA["activity_segment"]
                order = ["morning", "afternoon", "evening", "night"]
                seg["time_segment"] = pd.Categorical(seg["time_segment"], categories=order, ordered=True)
                seg = seg.sort_values("time_segment")
                fig = px.bar(
                    seg, x="time_segment", y="n_points", color="time_segment",
                    labels={"time_segment": "Segment", "n_points": "GPS points"},
                    title="Activity by time-of-day segment",
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(style_fig(fig, 320), width="stretch")
            else:
                not_available()

        if kpis is not None and len(kpis) > 0:
            k = kpis.iloc[0]
            st.info(
                f"Peak activity hour: **{fmt_int(k.get('peak_mobility_hour'))}:00**. "
                f"Pipeline-labeled most-active period: **{k.get('most_active_period')}**."
            )

    # ------------------------------------------------------------------
    elif section == "Sequential Patterns":
        st.subheader("Frequent Sequential Movement Patterns (PrefixSpan)")
        st.caption(
            "PrefixSpan mines ordered sequences of grid cells that recur across many "
            "taxi-days; a pattern's cells may be non-contiguous in the original "
            "trajectory (gaps are allowed as long as the relative order matches)."
        )

        if not available("ranked_patterns"):
            not_available("Ranked pattern table (`nb04_ranked_patterns.csv`) not found.")
        else:
            rp = DATA["ranked_patterns"].copy()
            rp["pattern_display"] = rp["pattern"].apply(format_pattern)
            rp = rp.sort_values("composite_score", ascending=False)

            col_chart, col_table = st.columns([3, 2])
            with col_chart:
                top15 = rp.head(15).sort_values("support_percentage")
                fig = px.bar(
                    top15, x="support_percentage", y="pattern_display", orientation="h",
                    labels={"support_percentage": "Support (%)", "pattern_display": "Pattern"},
                    title="Top patterns by support",
                )
                st.plotly_chart(style_fig(fig, 440), width="stretch")
            with col_table:
                st.markdown("**Ranked by composite score**")
                display_cols = {
                    "pattern_display": "Pattern",
                    "support_count": "Support count",
                    "support_percentage": "Support %",
                    "n_taxi_days_matched": "Taxi-days matched",
                    "n_taxis_matched": "Taxis matched",
                    "dominant_time_segment": "Dominant segment",
                    "composite_score": "Ranking score",
                }
                st.dataframe(
                    rp[list(display_cols)].rename(columns=display_cols).round(3),
                    hide_index=True, width="stretch", height=440,
                )
            st.caption(
                "Ranking score is the pipeline's composite score, combining "
                "normalized support, taxi coverage, temporal consistency, spatial "
                "density, pattern length, and recurrence across dates."
            )

    # ------------------------------------------------------------------
    elif section == "Mobility Corridors":
        st.subheader("Mobility Corridors")
        st.markdown(
            "A **corridor** is a frequent sequence whose cells map onto **2 or more "
            "distinct, non-noise hotspot clusters** — i.e. it connects real hotspots, "
            "not just repeated visits within one. This is a purely descriptive "
            "statistical recurrence; **no causal claim** is made."
        )
        st.metric("Mobility corridors identified", fmt_int(n_corridors))

        if n_corridors == 0:
            st.markdown(f'<div class="mob-zero">{CORRIDOR_ZERO_MSG}</div>', unsafe_allow_html=True)
            st.caption(
                "This matches the pipeline's own KPI summary "
                "(`most_recurring_corridor = N/A`, `n_hotspot_to_hotspot_corridors = 0` "
                "in `phase2_research_metrics.csv`) — the corridor tables produced by "
                "Notebook 04 are empty by design, not by a dashboard bug."
            )
        else:
            ct = DATA["corridor_table"]
            st.dataframe(ct, hide_index=True, width="stretch")

    # ------------------------------------------------------------------
    elif section == "Taxi Allocation":
        st.subheader("Suggested Taxi Allocation")
        st.markdown(
            "**Suggested allocation — not an optimal or real-time dispatch solution.** "
            "It distributes the taxis you specify across hotspots in proportion to "
            "*historically observed* GPS-point volume for each hotspot cluster."
        )

        if not available("cluster_summary"):
            not_available("Hotspot cluster table not found.")
        else:
            cs = DATA["cluster_summary"]
            non_noise = cs[cs["is_noise"] == False].sort_values("n_points", ascending=False).reset_index(drop=True)  # noqa: E712

            available_taxis = st.number_input(
                "Available taxis to allocate", min_value=1, max_value=5000, value=50, step=1
            )

            total_pts = non_noise["n_points"].sum()
            shares = non_noise["n_points"] / total_pts
            raw = shares * available_taxis
            floor_alloc = np.floor(raw).astype(int)
            remainder = int(available_taxis - floor_alloc.sum())
            # largest-remainder method so the suggested counts sum exactly to input
            order = np.argsort(-(raw - floor_alloc).values)
            alloc = floor_alloc.copy()
            for idx in order[:remainder]:
                alloc.iloc[idx] += 1

            alloc_df = non_noise.copy()
            alloc_df["share_pct"] = (shares * 100).round(2)
            alloc_df["suggested_taxis"] = alloc.values
            st.session_state["allocation_table"] = alloc_df

            c1, c2 = st.columns([2, 3])
            with c1:
                st.markdown("**Allocation by hotspot (all clusters)**")
                st.dataframe(
                    alloc_df[["cluster_id", "n_points", "share_pct", "suggested_taxis"]]
                    .rename(columns={"cluster_id": "Cluster", "n_points": "Historical GPS points",
                                      "share_pct": "Share %", "suggested_taxis": "Suggested taxis"}),
                    hide_index=True, width="stretch", height=420,
                )
            with c2:
                top = alloc_df.head(15)
                fig = px.bar(
                    top, x="cluster_id", y="suggested_taxis",
                    labels={"cluster_id": "Cluster", "suggested_taxis": "Suggested taxis"},
                    title="Suggested taxis — top 15 hotspots by historical activity",
                )
                fig.update_xaxes(type="category")
                st.plotly_chart(style_fig(fig, 440), width="stretch")

            st.download_button(
                "Download full allocation table (CSV)",
                alloc_df.to_csv(index=False).encode("utf-8"),
                file_name="suggested_taxi_allocation.csv",
                mime="text/csv",
            )

    # ------------------------------------------------------------------
    elif section == "Rare Mobility":
        st.subheader("Rare Mobility Transitions")
        st.caption(
            "Occurrence-based rare transitions — cell-to-cell moves observed on very "
            "few taxi-days in the sample. These describe unusual movement recurrence, "
            "not incidents, accidents, or fraud."
        )

        if not available("rare_transitions"):
            not_available("Rare transitions table not found.")
        else:
            rt = DATA["rare_transitions"]
            rule = rt["rule"].iloc[0] if "rule" in rt.columns else "n/a"
            m1, m2, m3 = st.columns(3)
            m1.metric("Rare transitions", fmt_int(len(rt)))
            m2.metric("Rule applied", rule)
            m3.metric(
                "Share of all transitions (sum)",
                fmt_pct(rt["pct_of_all_transitions"].sum()) if "pct_of_all_transitions" in rt.columns else "—",
            )

            top_n = st.slider("Rows to display", 20, 500, 100, step=20)
            display_rt = rt.sort_values("transition_count", ascending=False).head(top_n)
            st.dataframe(
                display_rt.rename(columns={
                    "from_cell": "From cell", "to_cell": "To cell",
                    "transition_count": "Occurrences", "n_taxi_days": "Taxi-days",
                    "pct_of_all_transitions": "% of all transitions", "rule": "Rarity rule",
                }),
                hide_index=True, width="stretch",
            )
            st.download_button(
                "Download full rare-transitions table (CSV)",
                rt.to_csv(index=False).encode("utf-8"),
                file_name="rare_mobility_transitions.csv",
                mime="text/csv",
            )

    # ------------------------------------------------------------------
    elif section == "Methodology & Evaluation":
        st.subheader("Model Evaluation & Scientific Methodology")
        st.markdown(
            """<div class="mob-banner"><b>Academic Evaluation Framework</b>: 
            Urban mobility pattern discovery is unsupervised. This section reports:
            1) Real-data evaluation using strictly uniform <b>ordered cell pairs</b>,
            2) An <b>objective synthetic benchmark</b> with known planted patterns evaluated against <i>realized support</i>,
            3) <b>Parameter sensitivity</b> across support thresholds and spatial grid sizes, and
            4) <b>Nested sample scaling</b> from 99 to 250 and 500 taxis.</div>""",
            unsafe_allow_html=True,
        )

        tab_real, tab_syn, tab_sens, tab_scale, tab_limits = st.tabs([
            "Real-Data Pair Evaluation",
            "Objective Synthetic Benchmark",
            "Parameter Sensitivity & Stability",
            "Sample Size Scaling (99–500 Taxis)",
            "Methodological Limitations",
        ])

        with tab_real:
            st.markdown("### Real-Data Pair-Level Method Comparison")
            st.caption(
                "Evaluated against empirical reference transitions (count ≥ 15 across ≥ 2 taxis) "
                "using strictly identical ordered cell-pair units (origin, destination)."
            )
            if available("method_comparison"):
                mc = DATA["method_comparison"].copy()
                cols = {
                    "method": "Mining Method",
                    "evaluation_unit": "Evaluation Unit",
                    "predicted_pairs": "Predicted Pairs",
                    "reference_pairs": "Reference Pairs",
                    "TP": "TP", "FP": "FP", "FN": "FN",
                    "precision": "Precision", "recall": "Recall", "f1": "F1 Score",
                    "runtime_sec": "Runtime (s)",
                }
                st.dataframe(mc[[c for c in cols if c in mc.columns]].rename(columns=cols), hide_index=True, width="stretch")
            else:
                not_available("Method comparison table not found.")

            st.markdown("#### Order-Agnostic Co-occurrence Baseline")
            st.caption(
                "Because co-occurrence does not model travel direction, here both predictions and "
                "reference transitions are collapsed to undirected sets {origin, destination}."
            )
            if available("cooccurrence_order_agnostic"):
                oa = DATA["cooccurrence_order_agnostic"].copy()
                st.dataframe(oa, hide_index=True, width="stretch")

        with tab_syn:
            st.markdown("### Objective Synthetic Ground-Truth Benchmark")
            st.caption(
                "Tested against known, deliberately planted patterns with controlled noise. "
                "Ground-truth status is decided by REALIZED support in the finite generated sample."
            )
            if available("synthetic_planted"):
                st.markdown("**Realized Support of Planted Patterns in Generated Sample**")
                sp = DATA["synthetic_planted"].copy()
                st.dataframe(sp, hide_index=True, width="stretch")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Pair-Level Ground-Truth Recovery**")
                if available("synthetic_pair_eval"):
                    st.dataframe(DATA["synthetic_pair_eval"][["method", "TP", "FP", "FN", "precision", "recall", "f1"]], hide_index=True, width="stretch")
            with c2:
                st.markdown("**Pattern-Level Ground-Truth Recovery**")
                if available("synthetic_pattern_eval"):
                    st.dataframe(DATA["synthetic_pattern_eval"][["method", "TP", "FP", "FN", "precision", "recall", "f1"]], hide_index=True, width="stretch")

        with tab_sens:
            st.markdown("### Support Threshold & Grid Resolution Sensitivity")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**PrefixSpan Support Thresholds (500m Grid)**")
                if available("support_sensitivity"):
                    st.dataframe(DATA["support_sensitivity"][["min_support_pct", "min_support_count", "n_patterns", "runtime_sec", "ref_f1"]], hide_index=True, width="stretch")
            with c2:
                st.markdown("**Jaccard Stability Across Support Thresholds**")
                if available("pattern_stability"):
                    st.dataframe(DATA["pattern_stability"][["set_a", "set_b", "n_a", "n_b", "n_intersection", "jaccard"]], hide_index=True, width="stretch")

            st.markdown("**Spatial Grid Cell Size Sensitivity (5% Support Threshold)**")
            if available("grid_sensitivity"):
                st.dataframe(DATA["grid_sensitivity"][["grid_cell_size_m", "n_occupied_cells", "avg_points_per_cell", "n_sequences", "n_patterns_at_5pct", "total_runtime_sec"]], hide_index=True, width="stretch")
            st.caption("Note: Direct Jaccard comparison across different grid resolutions is excluded because cell IDs do not represent the same geographic entities.")

        with tab_scale:
            st.markdown("### Deterministic Nested Sample-Size Expansion")
            st.caption(
                "Evaluates whether discovered patterns persist when expanding fleet coverage. "
                "Samples are strictly nested: 99 taxis ⊂ 249 taxis ⊂ 497 taxis."
            )
            if available("sample_scaling"):
                st.dataframe(DATA["sample_scaling"][[
                    "sample_label", "active_taxis_processed", "cleaned_gps_points", "n_taxi_day_sequences",
                    "min_support_count_required", "n_discovered_patterns", "ref_f1", "total_pipeline_time_sec"
                ]], hide_index=True, width="stretch")
            st.caption(
                "Important Takeaway: Because support threshold is relative (5.0%), expanding the fleet from "
                "630 to 3,185 sequences raises the required absolute sequence count from 32 to 160 taxi-days, "
                "which is a significantly stricter cross-fleet coordination threshold."
            )

        with tab_limits:
            st.markdown("### Methodological Assumptions & Limitations")
            st.markdown(
                """
                - **Empirical Reference Set**: Real-world evaluation uses observed physical adjacent transitions meeting minimum recurrence rules. It serves as an agreement validation, not definitive external ground truth.
                - **Structural Bias Toward Direct Mining**: Because reference transitions are adjacent cell pairs, contiguous direct-transition mining structurally achieves 100% precision on recovered pairs.
                - **GPS Traces vs Passenger Demand**: GPS volume represents commercial taxi movements in a sampled fleet. It is an activity proxy, not proof of passenger demand or road congestion.
                - **Bounded-Universe Accuracy Artifact**: True negatives drastically dominate spatial networks (>98%), producing artificially high accuracy (~98.6%). Precision, Recall, and F1 remain the primary metrics.
                - **No Causation**: Frequent sequential patterns are observed statistical orderings, not evidence of causation.
                """
            )

    # ------------------------------------------------------------------
    elif section == "Export":
        st.subheader("Export processed results")
        st.caption("Download the exact processed tables this dashboard is built from.")

        exports = [
            ("Sequential patterns (ranked)", "ranked_patterns", "sequential_patterns.csv"),
            ("Hotspot clusters", "cluster_summary", "hotspot_clusters.csv"),
            ("Rare mobility transitions", "rare_transitions", "rare_mobility_transitions.csv"),
            ("Mobility corridors", "corridor_table", "mobility_corridors.csv"),
            ("Hourly activity", "activity_hour", "activity_by_hour.csv"),
            ("Method comparison (pair-level)", "method_comparison", "evaluation_method_comparison.csv"),
            ("Synthetic benchmark (planted)", "synthetic_planted", "synthetic_benchmark_planted.csv"),
            ("Sample size scaling", "sample_scaling", "sample_expansion_scaling.csv"),
        ]
        for label, key, fname in exports:
            df = DATA.get(key)
            c1, c2 = st.columns([3, 1])
            c1.write(label)
            if df is not None:
                c2.download_button(
                    "Download CSV", df.to_csv(index=False).encode("utf-8"),
                    file_name=fname, mime="text/csv", key=f"dl_{key}",
                )
            else:
                c2.caption("not available")

        if "allocation_table" in st.session_state:
            st.divider()
            c1, c2 = st.columns([3, 1])
            c1.write("Current suggested taxi allocation (from the Allocation tab)")
            c2.download_button(
                "Download CSV",
                st.session_state["allocation_table"].to_csv(index=False).encode("utf-8"),
                file_name="suggested_taxi_allocation.csv", mime="text/csv", key="dl_alloc",
            )
        else:
            st.caption("Visit **Taxi Allocation** first to also export a suggested allocation.")

# ===========================================================================
# CUSTOMER MODE
# ===========================================================================
else:
    st.sidebar.subheader("Refine")
    day_type_choice = st.sidebar.selectbox("Day type", ["All", "Weekday", "Weekend"], index=0)
    segment_choice = st.sidebar.selectbox(
        "Time period", ["All", "morning", "afternoon", "evening", "night"], index=0
    )

    hotspot_options = ["All"]
    hotspot_lookup = {}
    if available("cluster_summary"):
        cs_nn = DATA["cluster_summary"]
        cs_nn = cs_nn[cs_nn["is_noise"] == False].sort_values("n_points", ascending=False)  # noqa: E712
        for _, row in cs_nn.iterrows():
            label = f"Cluster {int(row['cluster_id'])} ({int(row['n_points']):,} pts)"
            hotspot_options.append(label)
            hotspot_lookup[label] = int(row["cluster_id"])
    hotspot_choice = st.sidebar.selectbox("Hotspot / location", hotspot_options, index=0)
    selected_cluster_id = hotspot_lookup.get(hotspot_choice)

    st.caption("All figures below are HISTORICAL observations from the analyzed GPS sample.")

    # ------------------------------------------------------------------
    if section == "Where is activity high?":
        st.subheader("Where is taxi activity historically high?")
        fmap = build_beijing_map(highlight_cluster_id=selected_cluster_id)
        if fmap is None:
            st.warning("No hotspot cluster data available to render the map.")
        else:
            st_folium(fmap, width=None, height=540, returned_objects=[])
        not_available(
            "Filtering the map itself by day type or time period is not available — "
            "the processed dataset's hotspot table is not broken down by time."
        )

    # ------------------------------------------------------------------
    elif section == "When is activity high?":
        st.subheader("When is taxi activity historically high?")
        if available("activity_hour"):
            hr = DATA["activity_hour"].sort_values("hour")
            fig = px.bar(
                hr, x="hour", y="n_points",
                labels={"hour": "Hour of day", "n_points": "GPS points"},
                title="Activity by hour of day (all days)",
            )
            st.plotly_chart(style_fig(fig), width="stretch")
        else:
            not_available()

        c1, c2 = st.columns(2)
        with c1:
            if available("activity_day_type") and day_type_choice != "All":
                dtdf = DATA["activity_day_type"]
                row = dtdf[dtdf["day_type"] == day_type_choice.lower()]
                if len(row):
                    r0 = row.iloc[0]
                    st.metric(f"{day_type_choice} — GPS points", fmt_int(r0["n_points"]))
                    st.metric(f"{day_type_choice} — active taxis", fmt_int(r0["n_active_taxis"]))
            elif available("activity_day_type"):
                fig = px.bar(
                    DATA["activity_day_type"], x="day_type", y="n_points",
                    title="Weekday vs weekend", labels={"day_type": "Day type", "n_points": "GPS points"},
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(style_fig(fig, 320), width="stretch")
        with c2:
            if available("activity_segment") and segment_choice != "All":
                seg = DATA["activity_segment"]
                row = seg[seg["time_segment"] == segment_choice]
                if len(row):
                    r0 = row.iloc[0]
                    st.metric(f"{segment_choice.title()} — GPS points", fmt_int(r0["n_points"]))
                    st.metric(f"{segment_choice.title()} — active taxis", fmt_int(r0["n_active_taxis"]))
            elif available("activity_segment"):
                order = ["morning", "afternoon", "evening", "night"]
                seg = DATA["activity_segment"].copy()
                seg["time_segment"] = pd.Categorical(seg["time_segment"], categories=order, ordered=True)
                fig = px.bar(
                    seg.sort_values("time_segment"), x="time_segment", y="n_points",
                    title="By time-of-day segment", labels={"time_segment": "Segment", "n_points": "GPS points"},
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(style_fig(fig, 320), width="stretch")
        not_available(
            "Hour-level activity broken down jointly by day type is not available in "
            "the current processed dataset — the two views above are separate "
            "aggregates."
        )

    # ------------------------------------------------------------------
    elif section == "Historical patterns":
        st.subheader("Historical Mobility Patterns")
        shown = None
        source_note = ""
        if segment_choice != "All" and available("patterns_by_segment"):
            pbs = DATA["patterns_by_segment"]
            shown = pbs[pbs["time_segment"] == segment_choice].copy()
            source_note = f"Patterns with support during the **{segment_choice}** period."
        elif day_type_choice != "All" and available("patterns_by_day_type"):
            pbd = DATA["patterns_by_day_type"]
            shown = pbd[pbd["day_type"] == day_type_choice.lower()].copy()
            source_note = f"Patterns with support on **{day_type_choice.lower()}s**."
        elif available("ranked_patterns"):
            shown = DATA["ranked_patterns"].copy()
            source_note = "Top overall patterns across the full sample."

        if shown is None or len(shown) == 0:
            st.info("No historical observations match these filters.")
        else:
            st.caption(source_note)
            shown["pattern_display"] = shown["pattern"].apply(format_pattern)
            shown = shown.sort_values("support_percentage", ascending=False).head(15)
            fig = px.bar(
                shown.sort_values("support_percentage"),
                x="support_percentage", y="pattern_display", orientation="h",
                labels={"support_percentage": "Support (%)", "pattern_display": "Pattern"},
                title="Most common historical movement patterns",
            )
            st.plotly_chart(style_fig(fig, 460), width="stretch")

    # ------------------------------------------------------------------
    elif section == "Hotspot lookup":
        st.subheader("Hotspot Activity")
        if selected_cluster_id is None:
            st.info("Pick a hotspot / location from the sidebar to see its historical activity.")
        elif available("cluster_summary"):
            cs = DATA["cluster_summary"]
            cs_nn = cs[cs["is_noise"] == False].sort_values("n_points", ascending=False).reset_index(drop=True)  # noqa: E712
            row = cs_nn[cs_nn["cluster_id"] == selected_cluster_id].iloc[0]
            rank = int(cs_nn[cs_nn["cluster_id"] == selected_cluster_id].index[0]) + 1
            percentile = 100 * (1 - rank / len(cs_nn))

            m1, m2, m3 = st.columns(3)
            m1.metric("GPS points observed", fmt_int(row["n_points"]))
            m2.metric("Rank by activity", f"#{rank} of {len(cs_nn)}")
            m3.metric("Percentile", f"{percentile:.0f}th")

            fmap = build_beijing_map(highlight_cluster_id=selected_cluster_id, show_heat=False)
            if fmap is not None:
                st_folium(fmap, width=None, height=440, returned_objects=[])
            not_available(
                "Hour-of-day or day-type activity specific to this individual hotspot "
                "is not available in the current processed dataset."
            )

    # ------------------------------------------------------------------
    elif section == "Early-departure guidance":
        st.subheader("Early-Departure Guidance")
        st.caption(
            "HISTORICAL guidance only — this reflects past observed patterns, not a "
            "live traffic feed or a guaranteed travel time."
        )
        if available("activity_hour"):
            hr = DATA["activity_hour"].sort_values("n_points", ascending=False)
            top_hours = hr.head(3)["hour"].tolist()
            quiet_hours = hr.tail(3)["hour"].tolist()
            st.markdown(
                f"- Historically, taxi activity is **higher** around "
                f"{', '.join(f'{h}:00' for h in top_hours)}.\n"
                f"- Historically, taxi activity is **lower** around "
                f"{', '.join(f'{h}:00' for h in quiet_hours)}.\n"
            )
            if kpis is not None and len(kpis):
                st.markdown(
                    f"- Pipeline-labeled busiest period overall: "
                    f"**{kpis.iloc[0].get('most_active_period')}**."
                )
        else:
            not_available()
        st.caption(
            "This is descriptive historical guidance, not a prediction and not a "
            "real-time traffic or travel-time guarantee."
        )
