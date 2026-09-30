"""One-shot OSM prefetch for the Dubai case study.

Tries to download the drivable road network + power substations via OSMnx
and caches them under data/.  The notebook uses these files when present and
falls back to documented synthetic proxies otherwise.
"""
import os
import sys
import json
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from site_selection_model import (BBOX, FALLBACK_SUBSTATIONS, fetch_osm_network,
                                  graph_to_edges_df, fetch_osm_substations)

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA, exist_ok=True)

sources = {"network": "synthetic", "substations": "synthetic"}

try:
    print("Downloading OSM drivable network ...", flush=True)
    G = fetch_osm_network(BBOX)
    edges = graph_to_edges_df(G)
    edges.to_parquet(os.path.join(DATA, "osm_edges.parquet"))
    sources["network"] = "osm"
    print(f"  saved {len(edges)} edges", flush=True)
except Exception as e:  # noqa: BLE001 - fallback is the documented path
    print(f"  OSM network failed ({type(e).__name__}: {e}); using synthetic fallback",
          flush=True)

try:
    print("Downloading OSM substations ...", flush=True)
    subs = fetch_osm_substations(BBOX)
    if not subs:
        raise ValueError("no substation nodes returned")
    pd.DataFrame(subs, columns=["lat", "lon"]).to_csv(
        os.path.join(DATA, "osm_substations.csv"), index=False)
    sources["substations"] = "osm"
    print(f"  saved {len(subs)} substations", flush=True)
except Exception as e:  # noqa: BLE001
    print(f"  OSM substations failed ({type(e).__name__}: {e}); using synthetic fallback",
          flush=True)
    pd.DataFrame(FALLBACK_SUBSTATIONS, columns=["lat", "lon"]).to_csv(
        os.path.join(DATA, "osm_substations.csv"), index=False)

with open(os.path.join(DATA, "sources.json"), "w") as f:
    json.dump(sources, f, indent=2)
print("sources:", sources)
print("PREFETCH_OK" if all(v == "osm" for v in sources.values()) else "PREFETCH_FALLBACK")
