#!/usr/bin/env python3
"""
Carbon Accounting Tool — Main Entry Point
==========================================
Flow:
  1. Match ESRI patches (PNG) with DEM patches (TIF) by tile coordinates
  2. Run CHMv2 inference → Canopy Height Model
  3. Classify forest type from DEM altitude zones
  4. Calculate AGB + Carbon Density (Chave 2014 allometry)
  5. Save per-layer TIFs + 6-panel visualization PNG

Usage:
    python run.py
    python run.py --esri_dir data/input/esri_patches --dem_dir data/input/dem_patches --n 2
    python run.py --config config.yaml --n 1
"""

from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.model import load_model_and_processor
from pipeline.tiling import Patch
from pipeline.inference import run_inference
from pipeline.carbon import classify_forest, calculate_carbon, patch_stats
from pipeline.visualise import visualize_patch, save_tif


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_tile_key(filename: str) -> str | None:
    """Extract 'part_N' or 'zZOOM_X_Y' from filename for matching."""
    m = re.search(r"(part_\d+)", filename)
    return m.group(1) if m else None


def match_patches(esri_dir: Path, dem_dir: Path):
    """
    Return list of (esri_path, dem_path) pairs matched by part index.
    """
    esri_files = sorted(esri_dir.glob("*.png"))
    dem_files = sorted(dem_dir.glob("*.tif"))

    dem_by_key = {}
    for d in dem_files:
        key = parse_tile_key(d.name)
        if key:
            dem_by_key[key] = d

    pairs = []
    for e in esri_files:
        key = parse_tile_key(e.name)
        if key and key in dem_by_key:
            pairs.append((e, dem_by_key[key]))

    return pairs


def process_patch(
    esri_path: Path,
    dem_path: Path,
    patch_idx: int,
    model,
    processor,
    device,
    cfg: dict,
    out_dir: Path,
):
    patch_name = esri_path.stem

    # Load ESRI patch
    esri_rgb = np.array(Image.open(esri_path).convert("RGB"), dtype=np.uint8)

    # Load DEM
    import rasterio
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)

    # Ensure DEM matches ESRI patch size (512×512)
    if dem.shape != (512, 512):
        from scipy.ndimage import zoom as ndimage_zoom
        sy = 512 / dem.shape[0]
        sx = 512 / dem.shape[1]
        dem = ndimage_zoom(dem, (sy, sx), order=1).astype(np.float32)

    # Run CHMv2 inference
    patch = Patch(array=esri_rgb, patch_idx=patch_idx, name=patch_name)
    chm_list = run_inference([patch], model, processor, device, cfg)
    chm = chm_list[0]

    # Forest classification from DEM
    zones = cfg.get("altitude_zones", [])
    forest_class = classify_forest(dem, zones)

    # Carbon calculation
    allometry_csv = cfg["allometry"]["csv_path"]
    carbon_fraction = cfg["allometry"].get("carbon_fraction", 0.47)
    carbon_result = calculate_carbon(chm, forest_class, allometry_csv, carbon_fraction)
    agb = carbon_result["agb"]
    carbon_density = carbon_result["carbon_density"]

    # Statistics
    stats = patch_stats(chm, dem, forest_class, agb, carbon_density)

    # Save TIFs
    tif_dir = out_dir / "tifs" / patch_name
    tif_dir.mkdir(parents=True, exist_ok=True)
    save_tif(chm,           tif_dir / "chm.tif")
    save_tif(dem,           tif_dir / "dem.tif")
    save_tif(forest_class.astype(np.float32), tif_dir / "forest_class.tif")
    save_tif(agb,           tif_dir / "agb.tif")
    save_tif(carbon_density, tif_dir / "carbon_density.tif")

    # Visualization
    viz_dir = out_dir / "visualizations"
    visualize_patch(
        esri_rgb=esri_rgb,
        dem=dem,
        chm=chm,
        forest_class=forest_class,
        agb=agb,
        carbon_density=carbon_density,
        patch_name=patch_name,
        out_dir=viz_dir,
        stats=stats,
        cfg=cfg,
    )

    return stats


def print_stats(patch_name: str, stats: dict):
    from pipeline.carbon import CLASS_NAMES
    print(f"\n{'='*60}")
    print(f"  Patch: {patch_name}")
    print(f"{'='*60}")
    print(f"  CHM mean/max   : {stats['chm_mean_m']}m  /  {stats['chm_max_m']}m")
    print(f"  DEM mean       : {stats['dem_mean_m']}m")
    print(f"  AGB mean       : {stats['agb_mean_mgha']} Mg/ha")
    print(f"  Carbon mean    : {stats['carbon_mean_mgcha']} MgC/ha")
    print(f"  Carbon total   : {stats['carbon_total_mgc']} MgC")
    print(f"  Forest area (ha):")
    for name, ha in stats["forest_area_ha"].items():
        if ha > 0:
            print(f"    {name:15s}: {ha:.3f} ha")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Carbon Accounting Tool")
    parser.add_argument("--config",   default="config.yaml")
    parser.add_argument("--esri_dir", default="data/input/esri_patches")
    parser.add_argument("--dem_dir",  default="data/input/dem_patches")
    parser.add_argument("--out_dir",  default="data/output")
    parser.add_argument("--n",        type=int, default=2, help="Number of patches to process")
    args = parser.parse_args()

    cfg = load_config(args.config)
    esri_dir = Path(args.esri_dir)
    dem_dir  = Path(args.dem_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("  Carbon Accounting Tool")
    print("  CHMv2 + DEM → Forest Class → AGB → Carbon")
    print("="*60)
    print(f"  ESRI patches : {esri_dir}")
    print(f"  DEM patches  : {dem_dir}")
    print(f"  Output       : {out_dir}")
    print(f"  Device       : {cfg['model']['device']}")
    print(f"  Patches      : {args.n}")
    print("="*60 + "\n")

    # Match ESRI ↔ DEM pairs
    pairs = match_patches(esri_dir, dem_dir)
    if not pairs:
        print(f"[error] No matched ESRI/DEM pairs found.")
        print(f"  Run fetch scripts first:")
        print(f"    python scripts/fetch_esri_patches.py --bbox <MIN_LON MIN_LAT MAX_LON MAX_LAT>")
        print(f"    python scripts/fetch_dem_patches.py  --bbox <MIN_LON MIN_LAT MAX_LON MAX_LAT>")
        sys.exit(1)

    pairs = pairs[: args.n]
    print(f"[run] Processing {len(pairs)} patch pair(s).\n")

    # Load model once
    model, processor, device = load_model_and_processor(cfg)

    # Process each pair
    all_stats = []
    for idx, (esri_path, dem_path) in enumerate(pairs):
        print(f"\n[patch {idx+1}/{len(pairs)}] {esri_path.name} + {dem_path.name}")
        stats = process_patch(esri_path, dem_path, idx, model, processor, device, cfg, out_dir)
        print_stats(esri_path.stem, stats)
        all_stats.append((esri_path.stem, stats))

    # Summary
    print("\n" + "="*60)
    print("  COMPLETE")
    print("="*60)
    print(f"  TIFs         : {out_dir / 'tifs'}/")
    print(f"  Visuals      : {out_dir / 'visualizations'}/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
