"""
scripts/fetch_dem_patches.py
=============================
Fetch DEM (COP-DEM GLO-30) patches from Planetary Computer for the same
tile coordinates as the ESRI patches. Each output is a float32 GeoTIFF
resampled to 512×512 pixels.

Run AFTER fetch_esri_patches.py with the identical --bbox and --zoom.

Usage:
    python scripts/fetch_dem_patches.py --bbox 78.05 30.44 78.09 30.47 --zoom 18
    python scripts/fetch_dem_patches.py --lat 30.455 --lon 78.075 --zoom 18
"""

from __future__ import annotations
import argparse
import math
import os

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.merge import merge as rasterio_merge
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds as transform_from_bounds
from tqdm import tqdm

try:
    import planetary_computer
    import pystac_client
    _PC_AVAILABLE = True
except ImportError:
    _PC_AVAILABLE = False
    print("[dem] planetary-computer / pystac-client not installed — using synthetic DEM fallback.")

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
DEM_COLLECTION = "cop-dem-glo-30"
OUT_SIZE = (512, 512)
DST_CRS = CRS.from_epsg(4326)


# ── tile math (same as fetch_esri_patches.py) ─────────────────────────────────

def latlon_to_tile(lat: float, lon: float, zoom: int):
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def tile_to_bbox(start_x: int, start_y: int, zoom: int):
    """Lat/lon bbox for the 2×2 tile grid (start_x, start_y) used by ESRI script."""
    n = 2 ** zoom
    lon_min = start_x / n * 360.0 - 180.0
    lon_max = (start_x + 2) / n * 360.0 - 180.0
    lat_max = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * start_y / n))))
    lat_min = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (start_y + 2) / n))))
    return lon_min, lat_min, lon_max, lat_max


# ── DEM helpers ────────────────────────────────────────────────────────────────

def _synthetic_dem(lat_center: float, size: tuple = OUT_SIZE) -> np.ndarray:
    """Approximate elevation for Dehradun–Mussoorie region when PC unavailable."""
    base = 600.0 + (lat_center - 30.3) * 4000.0
    rng = np.random.default_rng(seed=int(abs(lat_center) * 1000))
    noise = rng.normal(0, 30, size).astype(np.float32)
    return np.clip(np.full(size, base, dtype=np.float32) + noise, 0, 5000)


def _open_dem_sources(full_bbox):
    """
    Query Planetary Computer STAC once for the full area bbox.
    Returns list of open rasterio datasets (caller must close them).
    Returns None if PC unavailable or no items found.
    """
    if not _PC_AVAILABLE:
        return None

    try:
        catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
        items = list(
            catalog.search(
                collections=[DEM_COLLECTION],
                bbox=list(full_bbox),
            ).items()
        )
        if not items:
            print("[dem] No DEM items found for bbox — will use synthetic fallback.")
            return None

        print(f"[dem] Found {len(items)} DEM tile(s) covering the area.")
        return [rasterio.open(item.assets["data"].href) for item in items]

    except Exception as e:
        print(f"[dem] STAC query failed ({e}) — will use synthetic fallback.")
        return None


def _extract_patch(
    sources: list,
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
) -> np.ndarray:
    """
    Reproject+resample a patch bbox from open rasterio datasets to 512×512.
    Uses rasterio.warp.reproject for clean windowed extraction without
    re-downloading the source data.
    """
    dst_transform = transform_from_bounds(lon_min, lat_min, lon_max, lat_max, OUT_SIZE[1], OUT_SIZE[0])
    dest = np.zeros(OUT_SIZE, dtype=np.float32)

    for src in sources:
        reproject(
            source=rasterio.band(src, 1),
            destination=dest,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=DST_CRS,
            resampling=Resampling.bilinear,
        )

    return dest


def save_dem_tif(dem: np.ndarray, out_path: str, bbox=None):
    """Save float32 DEM as GeoTIFF. Pass bbox=(lon_min,lat_min,lon_max,lat_max) to georeference."""
    transform = (
        transform_from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], dem.shape[1], dem.shape[0])
        if bbox else None
    )
    profile = dict(
        driver="GTiff", dtype=rasterio.float32,
        height=dem.shape[0], width=dem.shape[1],
        count=1, nodata=-9999.0,
    )
    if transform:
        profile["transform"] = transform
        profile["crs"] = DST_CRS

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(dem, 1)


# ── main fetch logic ───────────────────────────────────────────────────────────

def fetch_bbox(bbox, zoom: int, out_dir: str):
    min_lon, min_lat, max_lon, max_lat = bbox
    min_x, min_y = latlon_to_tile(max_lat, min_lon, zoom)
    max_x, max_y = latlon_to_tile(min_lat, max_lon, zoom)

    coords = [(x, y) for x in range(min_x, max_x + 1, 2) for y in range(min_y, max_y + 1, 2)]
    print(f"[dem] {len(coords)} patch(es) to generate @ zoom {zoom}")
    os.makedirs(out_dir, exist_ok=True)

    # ── Open DEM sources ONCE for entire bbox ──
    sources = _open_dem_sources(bbox)
    use_synthetic = sources is None

    try:
        for i, (x, y) in enumerate(tqdm(coords, desc="Generating DEM patches")):
            patch_bbox = tile_to_bbox(x, y, zoom)
            lon_min, lat_min, lon_max, lat_max = patch_bbox

            if use_synthetic:
                dem = _synthetic_dem((lat_min + lat_max) / 2)
            else:
                try:
                    dem = _extract_patch(sources, lon_min, lat_min, lon_max, lat_max)
                except Exception as e:
                    print(f"  [warn] patch {i} extract failed ({e}) — synthetic fallback.")
                    dem = _synthetic_dem((lat_min + lat_max) / 2)

            out_path = os.path.join(out_dir, f"dem_512_part_{i}_z{zoom}_{x}_{y}.tif")
            save_dem_tif(dem, out_path, bbox=patch_bbox)

    finally:
        if sources:
            for ds in sources:
                ds.close()

    print(f"[dem] Done — {len(coords)} patches saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lat", type=float)
    parser.add_argument("--lon", type=float)
    parser.add_argument("--bbox", type=float, nargs=4, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    parser.add_argument("--zoom", type=int, default=18)
    parser.add_argument("--out_dir", type=str, default="data/input/dem_patches")
    args = parser.parse_args()

    if args.bbox:
        fetch_bbox(args.bbox, args.zoom, args.out_dir)
    elif args.lat and args.lon:
        x, y = latlon_to_tile(args.lat, args.lon, args.zoom)
        patch_bbox = tile_to_bbox(x, y, args.zoom)
        lon_min, lat_min, lon_max, lat_max = patch_bbox
        print(f"[dem] Single patch at ({args.lat}, {args.lon})")
        sources = _open_dem_sources(patch_bbox)
        if sources:
            dem = _extract_patch(sources, lon_min, lat_min, lon_max, lat_max)
            for ds in sources:
                ds.close()
        else:
            dem = _synthetic_dem((lat_min + lat_max) / 2)
        os.makedirs(args.out_dir, exist_ok=True)
        out_path = os.path.join(args.out_dir, f"dem_512_center_z{args.zoom}_{x}_{y}.tif")
        save_dem_tif(dem, out_path, bbox=patch_bbox)
        print(f"  Saved: {out_path}")
    else:
        parser.error("Provide --bbox or --lat/--lon.")
