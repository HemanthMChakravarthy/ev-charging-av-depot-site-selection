"""EV charging / AV depot site-selection model (Dubai case study).

Spatial multi-criteria model that scores candidate depot/charging sites on
traffic density, grid capacity, and fleet demand, then selects depot locations
with a greedy max-coverage optimizer.

All default inputs are ILLUSTRATIVE assumptions for scenario modelling, not
market data. Calibrate with local traffic, grid, and demand data before any
commercial use.

Core (numpy/pandas only): grid, demand, scoring, optimizer.
Optional lazy imports: osmnx (road network), folium (interactive maps),
matplotlib (static charts).
"""

import math

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Study area & illustrative inputs
# ---------------------------------------------------------------------------

BBOX = {"south": 25.05, "north": 25.32, "west": 55.12, "east": 55.42}

# Illustrative fleet-demand hotspots: (name, lat, lon, weight).
# Weights encode relative trip-generation intensity, NOT measured demand.
HOTSPOTS = [
    ("Downtown Dubai / Burj Khalifa", 25.1972, 55.2744, 1.00),
    ("DXB Airport", 25.2532, 55.3657, 0.90),
    ("Dubai Marina / JBR", 25.0805, 55.1403, 0.80),
    ("Business Bay", 25.1857, 55.2664, 0.70),
    ("Deira", 25.2631, 55.2972, 0.60),
    ("Mall of the Emirates / Al Barsha", 25.1182, 55.2003, 0.60),
    ("Dubai Internet City", 25.1100, 55.1680, 0.50),
]

# Illustrative fallback arterials used ONLY if the OSM road download fails:
# (name, [(lat, lon), ...] polyline).  Rough traces of E11 / E44 corridors.
FALLBACK_ARTERIALS = [
    ("Sheikh Zayed Rd (E11, illustrative trace)",
     [(25.0650, 55.1290), (25.1200, 55.1850), (25.1750, 55.2400),
      (25.2200, 55.2850), (25.2550, 55.3300)]),
    ("Al Khail Rd (E44, illustrative trace)",
     [(25.0900, 55.1650), (25.1400, 55.2200), (25.1900, 55.2700),
      (25.2350, 55.3150), (25.2600, 55.3600)]),
]

# Illustrative fallback substation points (coarse grid), used ONLY if OSM
# power=substation nodes are unavailable in the bbox.
FALLBACK_SUBSTATIONS = [
    (25.08, 55.16), (25.08, 55.24), (25.08, 55.32),
    (25.14, 55.20), (25.14, 55.28), (25.14, 55.36),
    (25.20, 55.16), (25.20, 55.24), (25.20, 55.32),
    (25.26, 55.20), (25.26, 55.28), (25.26, 55.36),
]

DEFAULT_WEIGHTS = {"traffic": 0.35, "grid": 0.25, "demand": 0.40}

HIGHWAY_CLASS_WEIGHT = {
    "motorway": 3.0, "trunk": 2.5, "primary": 2.0, "secondary": 1.5,
    "tertiary": 1.0, "unclassified": 0.7, "residential": 0.5, "service": 0.3,
}

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Geo helpers (equirectangular km projection around the bbox centre)
# ---------------------------------------------------------------------------

def _lat0_lon0(bbox=BBOX):
    return (bbox["south"] + bbox["north"]) / 2.0, (bbox["west"] + bbox["east"]) / 2.0


def to_xy(lat, lon, lat0=None, lon0=None):
    """Lat/lon -> km offsets. Accepts scalars or numpy arrays."""
    if lat0 is None or lon0 is None:
        lat0, lon0 = _lat0_lon0()
    x = (np.asarray(lon, dtype=float) - lon0) * 111.32 * math.cos(math.radians(lat0))
    y = (np.asarray(lat, dtype=float) - lat0) * 110.57
    return x, y


def to_latlon(x, y, lat0=None, lon0=None):
    if lat0 is None or lon0 is None:
        lat0, lon0 = _lat0_lon0()
    lon = np.asarray(x, dtype=float) / (111.32 * math.cos(math.radians(lat0))) + lon0
    lat = np.asarray(y, dtype=float) / 110.57 + lat0
    return lat, lon


def dist_matrix_km(ax, ay, bx, by):
    """Pairwise Euclidean distance (km) between point sets A and B."""
    ax = np.asarray(ax)[:, None]
    ay = np.asarray(ay)[:, None]
    bx = np.asarray(bx)[None, :]
    by = np.asarray(by)[None, :]
    return np.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def normalize(s):
    """Min-max normalize a pandas Series / numpy array to [0, 1]."""
    s = np.asarray(s, dtype=float)
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-12:
        return np.zeros_like(s)
    return (s - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Demand grid
# ---------------------------------------------------------------------------

def make_grid(bbox=BBOX, cell_km=1.5):
    """Regular grid of cell centroids over the bbox."""
    lat0, lon0 = _lat0_lon0(bbox)
    x_min, y_min = to_xy(bbox["south"], bbox["west"], lat0, lon0)
    x_max, y_max = to_xy(bbox["north"], bbox["east"], lat0, lon0)
    xs = np.arange(x_min + cell_km / 2, x_max, cell_km)
    ys = np.arange(y_min + cell_km / 2, y_max, cell_km)
    xx, yy = np.meshgrid(xs, ys)
    lat, lon = to_latlon(xx.ravel(), yy.ravel(), lat0, lon0)
    return pd.DataFrame({
        "cell_id": np.arange(xx.size),
        "x_km": xx.ravel(), "y_km": yy.ravel(),
        "lat": lat, "lon": lon,
    })


def demand_intensity(cells, hotspots=HOTSPOTS, sigma_km=4.0, base=0.05):
    """Illustrative fleet-demand intensity per cell.

    Sum of Gaussian kernels around hotspot POIs with distance decay, plus a
    small uniform base.  Labeled illustrative: not measured trip data.
    """
    cells = cells.copy()
    hx, hy = to_xy([h[1] for h in hotspots], [h[2] for h in hotspots])
    w = np.array([h[3] for h in hotspots])
    d2 = dist_matrix_km(cells["x_km"].values, cells["y_km"].values, hx, hy) ** 2
    kernels = np.exp(-d2 / (2 * sigma_km ** 2)) * w[None, :]
    cells["demand_raw"] = kernels.sum(axis=1) + base
    cells["demand"] = normalize(cells["demand_raw"])
    return cells


# ---------------------------------------------------------------------------
# Traffic-density proxy
# ---------------------------------------------------------------------------

def traffic_from_edges(cells, edges):
    """Per-cell traffic proxy from OSM edges.

    edges: DataFrame with x1,y1,x2,y2 (km, edge endpoints) and class_w
    (road-class weight).  Proxy = sum of class-weighted road length whose
    edge midpoint falls in the cell's nearest-centroid assignment.
    """
    edges = edges.copy()
    mx, my = (edges["x1"] + edges["x2"]) / 2.0, (edges["y1"] + edges["y2"]) / 2.0
    length = np.sqrt((edges["x2"] - edges["x1"]) ** 2 + (edges["y2"] - edges["y1"]) ** 2)
    weighted = length * edges["class_w"]
    d = dist_matrix_km(mx.values, my.values,
                       cells["x_km"].values, cells["y_km"].values)
    nearest = d.argmin(axis=1)
    traffic = np.zeros(len(cells))
    np.add.at(traffic, nearest, weighted.values)
    cells = cells.copy()
    cells["traffic_raw"] = traffic
    cells["traffic"] = normalize(traffic)
    return cells


def _point_to_segment_dist(px, py, ax, ay, bx, by):
    t = ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / max((bx - ax) ** 2 + (by - ay) ** 2, 1e-12)
    t = np.clip(t, 0, 1)
    return np.sqrt((px - (ax + t * (bx - ax))) ** 2 + (py - (ay + t * (by - ay))) ** 2)


def synthetic_traffic(cells, arterials=FALLBACK_ARTERIALS, decay_km=2.5):
    """Fallback traffic proxy when OSM is unavailable: decay from illustrative
    arterial traces plus light noise.  Clearly labeled illustrative."""
    cells = cells.copy()
    lat0, lon0 = _lat0_lon0()
    dmin = np.full(len(cells), np.inf)
    for _, pts in arterials:
        ax, ay = to_xy([p[0] for p in pts], [p[1] for p in pts], lat0, lon0)
        for i in range(len(ax) - 1):
            d = _point_to_segment_dist(cells["x_km"].values, cells["y_km"].values,
                                       ax[i], ay[i], ax[i + 1], ay[i + 1])
            dmin = np.minimum(dmin, d)
    raw = np.exp(-dmin / decay_km) + 0.15 * RNG.random(len(cells))
    cells["traffic_raw"] = raw
    cells["traffic"] = normalize(raw)
    return cells


# ---------------------------------------------------------------------------
# Grid-capacity proxy (distance to nearest substation)
# ---------------------------------------------------------------------------

def grid_capacity(cells, substations, decay_km=3.0):
    """Grid-capacity proxy: exp decay of distance to nearest substation.

    substations: iterable of (lat, lon).  Higher = closer to grid headroom.
    Illustrative proxy — not actual feeder capacity data.
    """
    cells = cells.copy()
    sx, sy = to_xy([s[0] for s in substations], [s[1] for s in substations])
    d = dist_matrix_km(cells["x_km"].values, cells["y_km"].values, sx, sy)
    dmin = d.min(axis=1)
    cells["grid_raw"] = np.exp(-dmin / decay_km)
    cells["grid"] = normalize(cells["grid_raw"])
    cells["dist_substation_km"] = dmin
    return cells


# ---------------------------------------------------------------------------
# Candidate sites + scoring
# ---------------------------------------------------------------------------

def candidate_points(bbox=BBOX, spacing_km=1.2, edges=None, snap_km=0.4):
    """Regular candidate lattice; optionally snapped to near-road locations.

    edges: optional DataFrame with x1..y2 (km).  Candidates farther than
    snap_km from every edge midpoint are dropped (depots need road access).
    """
    lat0, lon0 = _lat0_lon0(bbox)
    x_min, y_min = to_xy(bbox["south"], bbox["west"], lat0, lon0)
    x_max, y_max = to_xy(bbox["north"], bbox["east"], lat0, lon0)
    xs = np.arange(x_min + spacing_km / 2, x_max, spacing_km)
    ys = np.arange(y_min + spacing_km / 2, y_max, spacing_km)
    xx, yy = np.meshgrid(xs, ys)
    lat, lon = to_latlon(xx.ravel(), yy.ravel(), lat0, lon0)
    cands = pd.DataFrame({
        "cand_id": np.arange(xx.size),
        "x_km": xx.ravel(), "y_km": yy.ravel(),
        "lat": lat, "lon": lon,
    })
    if edges is not None and len(edges):
        mx = ((edges["x1"] + edges["x2"]) / 2.0).values
        my = ((edges["y1"] + edges["y2"]) / 2.0).values
        d = dist_matrix_km(cands["x_km"].values, cands["y_km"].values, mx, my)
        cands = cands[d.min(axis=1) <= snap_km].reset_index(drop=True)
        cands["cand_id"] = np.arange(len(cands))
    return cands


def attach_cell_values(cands, cells):
    """Copy demand/traffic/grid of the nearest cell onto each candidate."""
    cands = cands.copy()
    d = dist_matrix_km(cands["x_km"].values, cands["y_km"].values,
                       cells["x_km"].values, cells["y_km"].values)
    nearest = cells.iloc[d.argmin(axis=1)].reset_index(drop=True)
    for col in ["demand", "traffic", "grid", "demand_raw", "cell_id"]:
        cands[col] = nearest[col].values
    return cands


def score_candidates(cands, weights=DEFAULT_WEIGHTS):
    """Weighted linear multi-criteria score (criteria pre-normalized)."""
    w = {k: weights[k] for k in ("traffic", "grid", "demand")}
    total = sum(w.values())
    w = {k: v / total for k, v in w.items()}
    cands = cands.copy()
    cands["score"] = (w["traffic"] * cands["traffic"]
                      + w["grid"] * cands["grid"]
                      + w["demand"] * cands["demand"])
    cands["w_traffic"], cands["w_grid"], cands["w_demand"] = w["traffic"], w["grid"], w["demand"]
    return cands.sort_values("score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Greedy max-coverage optimizer
# ---------------------------------------------------------------------------

def greedy_max_coverage(cands, cells, radius_km=5.0, n_depots=8, min_sep_km=2.5):
    """Greedily pick depots maximizing score-weighted newly covered demand.

    A cell is 'covered' if within radius_km of a selected depot.  Each pick
    maximizes sum(demand of newly covered cells) x candidate score, so site
    quality (traffic/grid/demand score) and coverage jointly drive the
    choice.  Candidates within min_sep_km of an already-selected depot are
    skipped (avoids clustering).  Returns (selected DataFrame,
    coverage_fractions list).
    """
    demand = cells["demand_raw"].values
    total_demand = demand.sum()
    dmat = dist_matrix_km(cands["x_km"].values, cands["y_km"].values,
                          cells["x_km"].values, cells["y_km"].values)
    covers = dmat <= radius_km
    cand_xy = cands[["x_km", "y_km"]].values
    cand_score = cands["score"].values

    selected_idx, covered = [], np.zeros(len(cells), dtype=bool)
    coverage_frac = []
    for _ in range(n_depots):
        best, best_gain = None, -1.0
        for i in range(len(cands)):
            if i in selected_idx:
                continue
            if selected_idx:
                sep = np.sqrt(((cand_xy[i] - cand_xy[selected_idx]) ** 2).sum(axis=1))
                if (sep < min_sep_km).any():
                    continue
            gain = (demand[~covered & covers[i]] * cand_score[i]).sum()
            if gain > best_gain:
                best, best_gain = i, gain
        if best is None:
            break
        selected_idx.append(best)
        covered |= covers[best]
        coverage_frac.append(covered @ demand / total_demand)
    selected = cands.iloc[selected_idx].copy().reset_index(drop=True)
    selected["rank"] = np.arange(1, len(selected) + 1)
    return selected, coverage_frac


def coverage_curve(cands, cells, radius_km=5.0, max_n=15, min_sep_km=2.5):
    """Demand-coverage fraction for 1..max_n depots (greedy)."""
    _, fracs = greedy_max_coverage(cands, cells, radius_km, max_n, min_sep_km)
    return list(range(1, len(fracs) + 1)), fracs


def weight_sensitivity(cands, cells, radius_km=5.0, n_depots=8, min_sep_km=2.5,
                       high=0.6):
    """How much does the selected depot set change when one criterion dominates?

    For each criterion, re-score with that criterion at `high` weight (others
    rescaled proportionally) and compute the Jaccard overlap of the selected
    depot id sets vs the base selection.  Returns dict criterion -> overlap.
    """
    base_sel, _ = greedy_max_coverage(cands, cells, radius_km, n_depots, min_sep_km)
    base_ids = set(base_sel["cand_id"])
    out = {}
    for crit in ("traffic", "grid", "demand"):
        w = {k: (1 - high) / 2 for k in ("traffic", "grid", "demand")}
        w[crit] = high
        rescored = score_candidates(cands, w)
        sel, _ = greedy_max_coverage(rescored, cells, radius_km, n_depots, min_sep_km)
        ids = set(sel["cand_id"])
        out[crit] = len(base_ids & ids) / max(len(base_ids | ids), 1)
    return out


# ---------------------------------------------------------------------------
# OSM fetch helpers (lazy import; network-dependent)
# ---------------------------------------------------------------------------

def fetch_osm_network(bbox=BBOX, network_type="drive"):
    """Download the drivable road network with OSMnx. Raises on failure."""
    import osmnx as ox
    G = ox.graph_from_bbox(
        (bbox["west"], bbox["south"], bbox["east"], bbox["north"]),
        network_type=network_type, simplify=True, retain_all=False)
    return G


def graph_to_edges_df(G, lat0=None, lon0=None):
    """OSMnx graph -> edge DataFrame with km endpoints + class weight."""
    import osmnx as ox
    nodes, edges = ox.graph_to_gdfs(G)
    if lat0 is None or lon0 is None:
        lat0, lon0 = _lat0_lon0()

    def class_weight(hw):
        tags = hw if isinstance(hw, list) else [hw]
        return max(HIGHWAY_CLASS_WEIGHT.get(str(t).split("_")[0], 0.4) for t in tags)

    rows = []
    for _, e in edges.iterrows():
        u, v = e["u"], e["v"]
        try:
            x1, y1 = to_xy(nodes.loc[u, "y"], nodes.loc[u, "x"], lat0, lon0)
            x2, y2 = to_xy(nodes.loc[v, "y"], nodes.loc[v, "x"], lat0, lon0)
        except KeyError:
            continue
        rows.append((x1, y1, x2, y2, class_weight(e.get("highway", "unclassified"))))
    return pd.DataFrame(rows, columns=["x1", "y1", "x2", "y2", "class_w"])


def fetch_osm_substations(bbox=BBOX):
    """OSM power=substation nodes in bbox -> list of (lat, lon). Raises on failure."""
    import osmnx as ox
    gdf = ox.features_from_bbox(
        (bbox["west"], bbox["south"], bbox["east"], bbox["north"]),
        tags={"power": "substation"})
    pts = gdf[gdf.geometry.geom_type == "Point"]
    return [(g.y, g.x) for g in pts.geometry]


# ---------------------------------------------------------------------------
# Folium interactive map (lazy import)
# ---------------------------------------------------------------------------

def build_folium_map(cells, cands=None, selected=None, hotspots=HOTSPOTS,
                     zoom_start=11):
    """Interactive map: demand heatmap + candidate dots + depot markers."""
    import folium
    from folium.plugins import HeatMap
    lat0, lon0 = _lat0_lon0()
    m = folium.Map(location=[lat0, lon0], zoom_start=zoom_start,
                   tiles="OpenStreetMap")
    heat_data = [[r["lat"], r["lon"], float(r["demand"])]
                 for _, r in cells.iterrows()]
    HeatMap(heat_data, radius=18, blur=14, max_zoom=13,
            name="Fleet demand intensity").add_to(m)
    for name, hlat, hlon, w in hotspots:
        folium.CircleMarker([hlat, hlon], radius=4, color="black", fill=True,
                            fill_opacity=0.7,
                            popup=f"{name} (w={w})").add_to(m)
    if cands is not None:
        top = cands.head(60)
        for _, r in top.iterrows():
            folium.CircleMarker([r["lat"], r["lon"]], radius=3,
                                color="#3186cc", fill=True, fill_opacity=0.5,
                                popup=(f"candidate {int(r['cand_id'])} · "
                                       f"score {r['score']:.2f}")).add_to(m)
    if selected is not None:
        for _, r in selected.iterrows():
            folium.Marker(
                [r["lat"], r["lon"]],
                popup=(f"<b>Depot {int(r['rank'])}</b><br>"
                       f"score {r['score']:.2f} · demand {r['demand']:.2f} · "
                       f"traffic {r['traffic']:.2f} · grid {r['grid']:.2f}"),
                icon=folium.Icon(color="red", icon="bolt",
                                 prefix="fa")).add_to(m)
    folium.LayerControl().add_to(m)
    return m
