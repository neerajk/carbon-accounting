#!/usr/bin/env python3
"""
CHMv2 Carbon Accounting Platform for Uttarakhand
================================================
Entry point for state-scale carbon estimation using CHMv2 + DINOv3.

CORRECT FLOW:
1. Fetch ESRI + DEM data → keep patches paired
2. Run CHMv2 model → Canopy Height Model (CHM)
3. Use DEM on CHM output → classify forest type
4. Visualize: RGB, DEM, CHM, Forest Class
5. Use allometry_params.csv → calculate AGB from CHM + forest_class
6. Calculate Carbon = 47% of AGB
7. Save all to GeoTIFF + unified visualization (single PNG with all steps)

Usage:
    python run_carbon_accounting.py --region dehradun --config config.yaml
    python run_carbon_accounting.py --region mussoorie --visualize-only
    python run_carbon_accounting.py --region dehradun --patch-viz

Author: Lead Geospatial AI Engineer
"""

from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import yaml
from pathlib import Path
import sys
import torch
import numpy as np
from dask.distributed import Client, LocalCluster

# Add pipeline to path
sys.path.insert(0, str(Path(__file__).parent))

from carbon_platform.inference_engine import CarbonInferenceEngine
from carbon_platform.stream_loader import StreamLoader
from carbon_platform.visualizer import CarbonVisualizer
from carbon_platform.dem_classifier import DEMClassifier
from carbon_platform.data_analysis import CarbonDataAnalyzer
from carbon_platform.geotiff_manager import GeoTIFFManager


def load_config(path: str) -> dict:
    """Load YAML configuration."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="CHMv2 Carbon Accounting Platform for Uttarakhand"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--region",
        choices=["dehradun", "mussoorie"],
        default="dehradun",
        help="Region to process",
    )
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Only initialize DataCube, don't run inference",
    )
    parser.add_argument(
        "--phase",
        choices=["a", "b", "all"],
        default="all",
        help="Run specific phase (a=ABA, b=ITC hotspots, all=both)",
    )
    parser.add_argument(
        "--visualize-only",
        action="store_true",
        help="Only generate visualizations from existing DataCube",
    )
    parser.add_argument(
        "--patch-viz",
        action="store_true",
        help="Generate detailed single-patch visualizations",
    )
    parser.add_argument(
        "--dem-analysis",
        action="store_true",
        help="Generate DEM analysis visualizations",
    )
    parser.add_argument(
        "--forest-map",
        action="store_true",
        help="Generate forest classification map",
    )
    parser.add_argument(
        "--data-charts",
        action="store_true",
        help="Generate data analysis charts",
    )
    parser.add_argument(
        "--export-report",
        action="store_true",
        help="Export analysis report",
    )
    parser.add_argument(
        "--carbon-threshold",
        type=float,
        default=50.0,
        help="Carbon threshold for Phase B hotspot detection (MgC/ha)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit processing to N chunks (for testing)",
    )
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)
    cfg["region"] = args.region
    if args.limit is not None:
        cfg["limit"] = args.limit

    # Paths
    project_root = Path(__file__).parent
    config_dir = project_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    geotiff_dir = outputs_dir / "geotiffs"
    allometry_path = config_dir / "allometry_params.csv"
    viz_dir = outputs_dir / "visualizations"

    print("=" * 70)
    print("  CHMv2 Carbon Accounting Platform")
    print("  Uttarakhand State-Scale Forest Carbon Estimation")
    print("=" * 70)
    print(f"  Region: {args.region.upper()}")
    print(f"  Device: {cfg['model']['device']}")
    print(f"  Resolution: {cfg.get('resolution', 2.0)}m")
    if cfg.get("limit"):
        print(f"  Limit: {cfg['limit']} chunks (testing mode)")
    print("=" * 70)
    print()

    # --- DASK MULTI-PROCESSING SETUP ---
    torch.set_num_threads(1)

    print("[Setup] Initializing Dask Multi-Processing Cluster...")
    cluster = LocalCluster(
        n_workers=4,
        threads_per_worker=1,
        processes=True,
        dashboard_address=':8787'
    )

    with Client(cluster) as client:
        print(f"Dask Dashboard: {client.dashboard_link}\n")

        # Initialize GeoTIFF Manager
        print("[Setup] Initializing GeoTIFF Manager...")
        geotiff_manager = GeoTIFFManager(
            output_dir=str(geotiff_dir),
            region=args.region,
            resolution=cfg.get("resolution", 2.0),
        )

        geotiff_manager.initialize_geotiffs()

        # Visualization-only mode
        if args.visualize_only:
            print("[Visualize] Loading existing GeoTIFF files...")
            data_dict = geotiff_manager.load_all_layers()
            run_comprehensive_visualization(data_dict, viz_dir, cfg, args, str(allometry_path))
            return

        if args.init_only:
            print("\n[Complete] GeoTIFF files initialized. Ready for inference.")
            print(f"  Store: {geotiff_dir}")
            print(f"  Dimensions: {geotiff_manager.height_px} x {geotiff_manager.width_px} pixels")
            return

        # Initialize inference engine
        print("[Setup] Loading inference engine...")
        engine = CarbonInferenceEngine(cfg, str(allometry_path))

        # Initialize stream loader
        print("[Setup] Initializing stream loader...")
        stream_loader = StreamLoader(
            bounds_3857=geotiff_manager.bounds_3857,
            zoom=cfg.get("zoom", 18),
            resolution=cfg.get("resolution", 2.0),
            use_real_data=cfg.get("use_real_data", True),
        )

        total_tiles = stream_loader.estimate_total_tiles()
        print(f"[Setup] Estimated tiles to process: {total_tiles}")

        # ================================================================
        # PHASE A: CORRECT FLOW
        # ================================================================
        if args.phase in ["a", "all"]:
            print("\n" + "=" * 70)
            print("  PHASE A: Area-Based Approach (ABA)")
            print("  Flow:")
            print("    1. Fetch ESRI + DEM → keep patches paired")
            print("    2. Run CHMv2 model → Canopy Height")
            print("    3. Use DEM on CHM output → classify forest")
            print("    4. Use allometry_params.csv → calculate AGB")
            print("    5. Calculate Carbon = 47% of AGB")
            print("    6. Save GeoTIFF + unified visualization (single PNG)")
            print("=" * 70)

            # Run Phase A - returns first chunk for visualization
            first_chunk_data = engine.run_phase_a(geotiff_manager, stream_loader)

            # Generate unified visualization (ALL STEPS IN ONE PNG)
            if first_chunk_data and first_chunk_data.get("chm") is not None:
                print("\n[Visualize] Generating unified flow visualization...")
                print("  This single PNG shows all steps:")
                print("    - RGB Imagery (ESRI input)")
                print("    - DEM Elevation (input)")
                print("    - Canopy Height (CHMv2 output)")
                print("    - Forest Classification (DEM-based)")
                print("    - Above-Ground Biomass (allometry)")
                print("    - Carbon Density (47% of AGB)")
                print("    - DEM vs CHM scatter")
                print("    - Statistics summary")

                visualizer = CarbonVisualizer(
                    str(viz_dir),
                    allometry_path=str(allometry_path),
                    carbon_fraction=0.47
                )

                rgb_stack = np.stack([
                    first_chunk_data["red"],
                    first_chunk_data["green"],
                    first_chunk_data["blue"]
                ], axis=-1)

                visualizer.visualize_single_patch(
                    rgb=rgb_stack,
                    dem=first_chunk_data["dem"],
                    chm=first_chunk_data["chm"],
                    forest_class=first_chunk_data["forest_class"],
                    carbon_density=first_chunk_data["carbon_density"],
                    agb=first_chunk_data["agb"],
                    patch_idx=0,
                    save_path=str(viz_dir / "complete_flow_visualization.png")
                )
                print("\n  SAVED: complete_flow_visualization.png (8-panel unified)")

            # Get statistics
            print("\n[Analysis] Calculating forest statistics...")
            data_dict = geotiff_manager.load_all_layers()
            dem_classifier = DEMClassifier()
            stats = dem_classifier.get_zone_statistics(data_dict["dem"])

            print("\nForest Zone Statistics:")
            for forest_name, info in stats.items():
                if info["area_ha"] > 0:
                    print(f"  {forest_name}: {info['area_ha']:.1f} ha")

        # ================================================================
        # ADDITIONAL VISUALIZATIONS (on request)
        # ================================================================
        print("\n" + "=" * 70)
        print("  GENERATING ADDITIONAL VISUALIZATIONS")
        print("=" * 70)

        data_dict = geotiff_manager.load_all_layers()
        run_comprehensive_visualization(data_dict, viz_dir, cfg, args, str(allometry_path))

        print("\n" + "=" * 70)
        print("  CARBON ACCOUNTING COMPLETE")
        print("=" * 70)
        print(f"  GeoTIFF Store: {geotiff_dir}")
        print(f"  Visualizations: {viz_dir}")
        print(f"  Reports: {outputs_dir}")
        print("=" * 70)


def run_comprehensive_visualization(data_dict, viz_dir, cfg, args, allometry_path: str | None = None):
    """Run comprehensive visualization suite with unified flow visualization."""
    viz_dir = Path(viz_dir)
    viz_dir.mkdir(parents=True, exist_ok=True)

    carbon_fraction = cfg.get("carbon", {}).get("carbon_fraction", 0.47)
    visualizer = CarbonVisualizer(str(viz_dir), allometry_path=allometry_path, carbon_fraction=carbon_fraction)

    altitude_zones = cfg.get("altitude_zones", [
        {"name": "Sal_Forest", "min_alt": 0, "max_alt": 1000, "class_code": 1, "wood_density": 0.82},
        {"name": "Chir_Pine", "min_alt": 1000, "max_alt": 1800, "class_code": 2, "wood_density": 0.49},
        {"name": "Oak_Banj", "min_alt": 1800, "max_alt": 2800, "class_code": 3, "wood_density": 0.72},
    ])

    print("\n[Visualize] Loading full arrays into memory...")
    dem_full = data_dict["dem"]
    forest_class_full = data_dict["forest_class"]
    chm_full = data_dict["chm"]
    agb_full = data_dict.get("agb")
    carbon_full = data_dict.get("carbon_density")

    # Find first valid patch for unified visualization
    h, w = chm_full.shape

    for i in range(min(3, h // 512)):
        y, x = i * 512, i * 512
        if y + 512 <= h and x + 512 <= w:
            rgb_patch = np.stack([
                data_dict["red"][y:y+512, x:x+512],
                data_dict["green"][y:y+512, x:x+512],
                data_dict["blue"][y:y+512, x:x+512],
            ], axis=-1)
            dem_patch = dem_full[y:y+512, x:x+512]
            chm_patch = chm_full[y:y+512, x:x+512]
            forest_patch = forest_class_full[y:y+512, x:x+512]
            agb_patch = agb_full[y:y+512, x:x+512] if agb_full is not None else None
            carbon_patch = carbon_full[y:y+512, x:x+512] if carbon_full is not None else None

            if np.any(chm_patch > 0):
                visualizer.visualize_single_patch(
                    rgb_patch, dem_patch, chm_patch, forest_patch,
                    carbon_patch, agb_patch,
                    patch_idx=i,
                    save_path=str(viz_dir / "complete_flow_visualization.png")
                )
                print(f"  Saved: complete_flow_visualization.png (8-panel unified)")
                break

    # Additional visualizations on request
    if args.dem_analysis or args.visualize_only:
        print("\n[Visualize] Generating DEM analysis...")
        visualizer.create_dem_analysis(dem_full, forest_class_full, altitude_zones,
                                       save_path=str(viz_dir / "dem_analysis.png"))
        print("  Saved: dem_analysis.png")

    if args.forest_map or args.visualize_only:
        print("\n[Visualize] Generating forest classification map...")
        visualizer.create_forest_classification_map(forest_class_full, dem_full, altitude_zones,
                                                    save_path=str(viz_dir / "forest_classification_map.png"))
        print("  Saved: forest_classification_map.png")

    if args.data_charts or args.visualize_only:
        print("\n[Visualize] Generating data analysis charts...")
        visualizer.create_data_analysis_charts(data_dict, altitude_zones,
                                               save_path=str(viz_dir / "data_analysis_charts.png"))
        print("  Saved: data_analysis_charts.png")

    if args.export_report or args.visualize_only:
        print("\n[Analysis] Running data analysis...")
        analyzer = CarbonDataAnalyzer(data_dict)
        results = analyzer.run_full_analysis(altitude_zones)
        analyzer.export_report(viz_dir / "analysis_report.txt")
        analyzer.export_csv(viz_dir / "analysis_results.csv")
        print("  Saved: analysis_report.txt")
        print("  Saved: analysis_results.csv")

    # Export sample slices
    print("\n[Visualize] Exporting sample slices...")
    for i in range(min(3, h // 512)):
        y = i * 512
        x = i * 512
        if y + 512 <= h and x + 512 <= w:
            visualizer.export_slice(data_dict, y, y + 512, x, x + 512, f"slice_{i:02d}.png")
            print(f"  Saved: slice_{i:02d}.png")

    # Summary
    print("\n[Visualize] Generating summary...")
    dem_classifier = DEMClassifier()
    stats = dem_classifier.get_zone_statistics(dem_full)
    visualizer.export_summary(data_dict, stats)
    print("  Saved: carbon_summary.png")


if __name__ == "__main__":
    main()