"""Streamlit demo: EV Charging / AV Depot Site Selection (Dubai).

Run with: streamlit run app.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from site_selection_model import (
    BBOX, HOTSPOTS, DEFAULT_WEIGHTS, make_grid, demand_intensity,
    synthetic_traffic, traffic_from_edges, grid_capacity, candidate_points,
    attach_cell_values, score_candidates, greedy_max_coverage, coverage_curve,
    to_xy,
)

st.set_page_config(page_title="AV Depot Site Selection — Dubai", layout="wide")
st.title("EV Charging / AV Depot Site Selection — Dubai")
st.caption("Multi-criteria siting: traffic density × grid capacity × fleet demand. "
           "All defaults are illustrative assumptions — calibrate locally before commercial use.")

DATA = os.path.join(os.path.dirname(__file__), "data")
ASSETS = os.path.join(os.path.dirname(__file__), "assets")


@st.cache_data
def load_inputs():
    cells = make_grid(BBOX, cell_km=1.5)
    cells = demand_intensity(cells)
    edges_path = os.path.join(DATA, "osm_edges.parquet")
    if os.path.exists(edges_path):
        edges = pd.read_parquet(edges_path)
        cells = traffic_from_edges(cells, edges)
        traffic_src = f"OSM ({len(edges)} edges)"
    else:
        edges = None
        cells = synthetic_traffic(cells)
        traffic_src = "synthetic fallback"
    subs = pd.read_csv(os.path.join(DATA, "osm_substations.csv"))[["lat", "lon"]].values.tolist()
    cells = grid_capacity(cells, subs)
    cands = candidate_points(BBOX, spacing_km=1.2, edges=edges, snap_km=0.4)
    cands = attach_cell_values(cands, cells)
    return cells, cands, edges, traffic_src


cells, cands, edges, traffic_src = load_inputs()

st.sidebar.header("Criterion weights")
w_traffic = st.sidebar.slider("Traffic density", 0.0, 1.0, DEFAULT_WEIGHTS["traffic"], 0.05)
w_grid = st.sidebar.slider("Grid capacity", 0.0, 1.0, DEFAULT_WEIGHTS["grid"], 0.05)
w_demand = st.sidebar.slider("Fleet demand", 0.0, 1.0, DEFAULT_WEIGHTS["demand"], 0.05)

st.sidebar.header("Optimizer")
n_depots = st.sidebar.slider("Number of depots", 1, 15, 8, 1)
radius = st.sidebar.slider("Coverage radius (km)", 2.0, 10.0, 5.0, 0.5)

weights = {"traffic": w_traffic, "grid": w_grid, "demand": w_demand}
scored = score_candidates(cands, weights)
selected, fracs = greedy_max_coverage(scored, cells, radius, n_depots, min_sep_km=2.5)

st.subheader(f"Coverage: {fracs[-1]:.1%} of illustrative demand "
             f"with {len(selected)} depots @ R={radius} km")
st.caption(f"Traffic source: {traffic_src}. Candidates: {len(cands)} road-snapped sites.")

col1, col2 = st.columns([3, 2])

with col1:
    fig, ax = plt.subplots(figsize=(8, 7))
    n = int(np.sqrt(len(cells)))
    ax.pcolormesh(cells["x_km"].values.reshape(n, n),
                  cells["y_km"].values.reshape(n, n),
                  cells["demand"].values.reshape(n, n),
                  shading="auto", cmap="YlOrRd", alpha=0.6)
    for _, r in selected.iterrows():
        ax.add_patch(plt.Circle((r["x_km"], r["y_km"]), radius, color="blue",
                                fill=False, lw=1.2, alpha=0.7))
        ax.plot(r["x_km"], r["y_km"], "b*", ms=14, mec="white")
        ax.annotate(f"D{int(r['rank'])}", (r["x_km"], r["y_km"]), fontsize=8,
                    fontweight="bold", xytext=(6, 6), textcoords="offset points")
    ax.set_aspect("equal")
    ax.set_title(f"Selected depots (weights T={w_traffic:.2f} G={w_grid:.2f} D={w_demand:.2f})")
    st.pyplot(fig, use_container_width=True)

with col2:
    ns, curve = coverage_curve(scored, cells, radius, max_n=15, min_sep_km=2.5)
    fig2, ax2 = plt.subplots(figsize=(5, 3.5))
    ax2.plot(ns, np.array(curve) * 100, "o-")
    ax2.axvline(n_depots, color="red", ls="--")
    ax2.set_xlabel("depots"); ax2.set_ylabel("demand covered (%)")
    ax2.set_title("Coverage vs depot count")
    ax2.grid(alpha=0.3)
    st.pyplot(fig2, use_container_width=True)

    st.dataframe(selected[["rank", "lat", "lon", "score", "traffic", "grid", "demand"]]
                 .round(3), use_container_width=True)

st.subheader("Interactive map")
try:
    from streamlit_folium import st_folium
    from site_selection_model import build_folium_map
    fmap = build_folium_map(cells, scored, selected)
    st_folium(fmap, width=1100, height=600)
except Exception as e:  # noqa: BLE001 - static map above already shows results
    st.info(f"Interactive map unavailable ({e}); static map shown above.")
