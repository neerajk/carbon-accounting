"""
scripts/fetch_esri_patches.py
==============================
Fetch 512x512 PNG patches from ESRI World Imagery (zoom 18, ~0.6m/px).
Each patch is a 2x2 stitch of 256px XYZ tiles.

Usage:
    python scripts/fetch_esri_patches.py --bbox 78.05 30.44 78.09 30.47 --zoom 18
    python scripts/fetch_esri_patches.py --lat 30.455 --lon 78.075 --zoom 18
"""

from __future__ import annotations
import argparse
import io
import math
import os

import requests
from PIL import Image
from tqdm import tqdm

ESRI_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; CarbonTool/1.0)"}


def latlon_to_tile(lat: float, lon: float, zoom: int):
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def tile_to_bbox(x: int, y: int, zoom: int):
    """Return (lon_min, lat_min, lon_max, lat_max) for a 2×2 tile grid starting at (x, y)."""
    n = 2 ** zoom
    lon_min = x / n * 360.0 - 180.0
    lon_max = (x + 2) / n * 360.0 - 180.0
    lat_max = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    lat_min = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 2) / n))))
    return lon_min, lat_min, lon_max, lat_max


def fetch_tile(zoom: int, x: int, y: int) -> Image.Image | None:
    url = ESRI_URL.format(z=zoom, x=x, y=y)
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception as e:
        print(f"  [warn] Tile {zoom}/{x}/{y} failed: {e}")
    return None


def stitch_512(zoom: int, start_x: int, start_y: int, out_dir: str, identifier: str) -> str:
    """Fetch 2×2 tiles and stitch into one 512×512 PNG."""
    canvas = Image.new("RGB", (512, 512))
    positions = [(0, 0), (256, 0), (0, 256), (256, 256)]
    offsets = [(start_x, start_y), (start_x + 1, start_y), (start_x, start_y + 1), (start_x + 1, start_y + 1)]

    for (px, py), (tx, ty) in zip(positions, offsets):
        tile = fetch_tile(zoom, tx, ty)
        if tile:
            canvas.paste(tile, (px, py))

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"esri_512_{identifier}_z{zoom}_{start_x}_{start_y}.png")
    canvas.save(path, format="PNG")
    return path


def fetch_bbox(bbox, zoom: int, out_dir: str):
    min_lon, min_lat, max_lon, max_lat = bbox
    min_x, min_y = latlon_to_tile(max_lat, min_lon, zoom)
    max_x, max_y = latlon_to_tile(min_lat, max_lon, zoom)

    coords = [(x, y) for x in range(min_x, max_x + 1, 2) for y in range(min_y, max_y + 1, 2)]
    print(f"[esri] Fetching {len(coords)} patch(es) for bbox {bbox} @ zoom {zoom}")

    for i, (x, y) in enumerate(tqdm(coords, desc="Fetching ESRI patches")):
        path = stitch_512(zoom, x, y, out_dir, f"part_{i}")
        print(f"  Saved: {path}")

    print(f"[esri] Done — {len(coords)} patches in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lat", type=float)
    parser.add_argument("--lon", type=float)
    parser.add_argument("--bbox", type=float, nargs=4, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    parser.add_argument("--zoom", type=int, default=18)
    parser.add_argument("--out_dir", type=str, default="data/input/esri_patches")
    args = parser.parse_args()

    if args.bbox:
        fetch_bbox(args.bbox, args.zoom, args.out_dir)
    elif args.lat and args.lon:
        x, y = latlon_to_tile(args.lat, args.lon, args.zoom)
        print(f"[esri] Single patch at ({args.lat}, {args.lon}) → tile ({x}, {y})")
        path = stitch_512(args.zoom, x, y, args.out_dir, "center")
        print(f"  Saved: {path}")
    else:
        parser.error("Provide --bbox or --lat/--lon.")
