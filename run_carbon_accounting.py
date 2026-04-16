#!/usr/bin/env python3
"""
CHMv2 Carbon Accounting Platform for Uttarakhand
================================================
Entry point for state-scale carbon estimation using CHMv2 + DINOv3.

Usage:
    python run_carbon_accounting.py --region dehradun --config config.yaml
    python run_carbon_accounting.py --region mussoorie --visualize-only
    python run_carbon_accounting.py --region dehradun --patch-viz

Author: Lead Geospatial AI Engineer
"""

from __future__ import annotations
import argparse
import yaml
from pathlib import Path
import sys

# Add pipeline to path
sys.path.insert(0, str(Path(__file__).parent))

from carbon_platform.datacube import DataCubeManager
from carbon_platform.inference_engine import CarbonInferenceEngine
from carbon_platform.stream_loader import StreamLoader
from carbon_platform.visualizer import CarbonVisualizer
from carbon_platform.dem_classifier import DEMClassifier
from carbon_platform.data_analysis import CarbonDataAnalyzer


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
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)
    cfg["region"] = args.region

    # Paths
    data_dir = Path(cfg["output"]["output_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)
    allometry_path = data_dir / "allometry_params.csv"
    viz_dir = data_dir / "visualizations"

    print("=" * 70)
    print("  CHMv2 Carbon Accounting Platform")
    print("  Uttarakhand State-Scale Forest Carbon Estimation")
    print("=" * 70)
    print(f"  Region: {args.region.upper()}")
    print(f"  Device: {cfg['model']['device']}")
    print(f"  Resolution: {cfg.get('resolution', 2.0)}m")
    print("=" * 70)
    print()

    # Initialize DataCube
    print("[Setup] Initializing DataCube...")
    datacube = DataCubeManager(
        store_path=str(data_dir),
        region=args.region,
        resolution=cfg.get("resolution", 2.0),
    )

    # Visualization-only mode
    if args.visualize_only:
        print("[Visualize] Loading existing DataCube...")
        ds = datacube.open_store()

        # Run comprehensive visualization
        run_comprehensive_visualization(ds, viz_dir, cfg, args)
        return

    # Initialize or open DataCube
    try:
        ds = datacube.open_store()
        print(f"[Setup] Opened existing DataCube: {data_dir / f'{args.region}_carbon.zarr'}")
    except FileNotFoundError:
        print("[Setup] Creating new DataCube...")
        ds = datacube.initialize_store(chunk_size=512)

    if args.init_only:
        print("\n[Complete] DataCube initialized. Ready for inference.")
        print(f"  Store: {data_dir / f'{args.region}_carbon.zarr'}")
        print(f"  Dimensions: {datacube.height_px} x {datacube.width_px} pixels")
        return

    # Initialize inference engine
    print("[Setup] Loading inference engine...")
    engine = CarbonInferenceEngine(cfg, str(allometry_path))

    # Initialize stream loader
    print("[Setup] Initializing stream loader...")
    stream_loader = StreamLoader(
        bounds_3857=datacube.bounds_3857,
        zoom=cfg.get("zoom", 18),
        resolution=cfg.get("resolution", 2.0),
        use_real_data=cfg.get("use_real_data", True),
    )

    total_tiles = stream_loader.estimate_total_tiles()
    print(f"[Setup] Estimated tiles to process: {total_tiles}")

    # Phase A: Full district ABA processing
    if args.phase in ["a", "all"]:
        print("\n" + "=" * 70)
        print("  PHASE A: Area-Based Approach (ABA)")
        print("  Full-district pixel-wise canopy height & carbon estimation")
        print("=" * 70)

        ds = engine.run_phase_a(ds, stream_loader)

        # Get statistics
        print("\n[Analysis] Calculating forest statistics...")
        dem_classifier = DEMClassifier()
        stats = dem_classifier.get_zone_statistics(ds["dem"].values)

        print("\nForest Zone Statistics:")
        for forest_name, info in stats.items():
            if info["area_ha"] > 0:
                print(f"  {forest_name}: {info['area_ha']:.1f} ha")

    # Phase B: ITC hotspot detection
    if args.phase in ["b", "all"]:
        print("\n" + "=" * 70)
        print("  PHASE B: Individual Tree Crown (ITC) Detection")
        print(f"  High-carbon hotspot detection (>{args.carbon_threshold} MgC/ha)")
        print("=" * 70)

        phase_b_results = engine.run_phase_b(ds, carbon_threshold=args.carbon_threshold)

        print(f"\n[Phase B] Hotspots detected: {phase_b_results['hotspot_count']}")
        print(f"[Phase B] Total carbon in hotspots: {phase_b_results['total_carbon_in_hotspots']:.1f} MgC")

    # Visualization
    print("\n" + "=" * 70)
    print("  GENERATING VISUALIZATIONS")
    print("=" * 70)

    run_comprehensive_visualization(ds, viz_dir, cfg, args)

    print("\n" + "=" * 70)
    print("  CARBON ACCOUNTING COMPLETE")
    print("=" * 70)
    print(f"  DataCube: {data_dir / f'{args.region}_carbon.zarr'}")
    print(f"  Visualizations: {viz_dir}")
    print(f"  Reports: {data_dir}")
    print("=" * 70)


def run_comprehensive_visualization(ds, viz_dir, cfg, args):
    """Run comprehensive visualization suite."""
    import numpy as np
    viz_dir = Path(viz_dir)
    viz_dir.mkdir(parents=True, exist_ok=True)

    visualizer = CarbonVisualizer(str(viz_dir))

    # Get altitude zones from config
    altitude_zones = cfg.get("altitude_zones", [
        {"name": "Sal_Forest", "min_alt": 0, "max_alt": 1000, "class_code": 1, "wood_density": 0.82},
        {"name": "Chir_Pine", "min_alt": 1000, "max_alt": 1800, "class_code": 2, "wood_density": 0.49},
        {"name": "Oak_Banj", "min_alt": 1800, "max_alt": 2800, "class_code": 3, "wood_density": 0.72},
    ])

    # 1. Single Patch Visualization (first patch)
    if args.patch_viz or args.visualize_only:
        print("\n[Visualize] Generating single-patch analysis...")
        # Get first valid patch
        h, w = ds.chm.shape

        # Find first valid 512x512 patch
        for i in range(min(3, h // 512)):
            y, x = i * 512, i * 512
            if y + 512 <= h and x + 512 <= w:
                rgb = np.stack([
                    ds.red[y:y+512, x:x+512].values,
                    ds.green[y:y+512, x:x+512].values,
                    ds.blue[y:y+512, x:x+512].values,
                ], axis=-1)

                dem = ds.dem[y:y+512, x:x+512].values
                chm = ds.chm[y:y+512, x:x+512].values
                forest_class = ds.forest_class[y:y+512, x:x+512].values
                carbon = ds.carbon_density[y:y+512, x:x+512].values

                # Check if valid
                if np.any(chm > 0):
                    visualizer.visualize_single_patch(
                        rgb, dem, chm, forest_class, carbon,
                        patch_idx=i, save_path=str(viz_dir / f"patch_{i:04d}_full_analysis.png")
                    )
                    print(f"  Saved: patch_{i:04d}_full_analysis.png (6-panel)")
                    break

    # 2. DEM Analysis
    if args.dem_analysis or args.visualize_only:
        print("\n[Visualize] Generating DEM analysis...")
        dem_classifier = DEMClassifier()
        dem_full = ds.dem.values
        forest_class_full = ds.forest_class.values

        visualizer.create_dem_analysis(
            dem_full, forest_class_full, altitude_zones,
            save_path=str(viz_dir / "dem_analysis.png")
        )
        print("  Saved: dem_analysis.png (DEM + slope + aspect + histograms)")

    # 3. Forest Classification Map
    if args.forest_map or args.visualize_only:
        print("\n[Visualize] Generating forest classification map...")
        visualizer.create_forest_classification_map(
            forest_class_full, dem_full, altitude_zones,
            save_path=str(viz_dir / "forest_classification_map.png")
        )
        print("  Saved: forest_classification_map.png")

    # 4. Data Analysis Charts
    if args.data_charts or args.visualize_only:
        print("\n[Visualize] Generating data analysis charts...")
        visualizer.create_data_analysis_charts(
            ds, altitude_zones,
            save_path=str(viz_dir / "data_analysis_charts.png")
        )
        print("  Saved: data_analysis_charts.png (10-panel analysis)")

    # 5. Data Analysis and Reporting
    if args.export_report or args.visualize_only:
        print("\n[Analysis] Running data analysis...")
        analyzer = CarbonDataAnalyzer(ds)
        results = analyzer.run_full_analysis(altitude_zones)

        # Export reports
        analyzer.export_report(viz_dir / "analysis_report.txt")
        analyzer.export_csv(viz_dir / "analysis_results.csv")
        print("  Saved: analysis_report.txt")
        print("  Saved: analysis_results.csv")

    # Always export sample slices
    print("\n[Visualize] Exporting sample slices...")
    h, w = ds.chm.shape
    for i in range(min(3, h // 512)):
        y = i * 512
        x = i * 512
        if y + 512 <= h and x + 512 <= w:
            visualizer.export_slice(ds, y, y + 512, x, x + 512, f"slice_{i:02d}.png")
            print(f"  Saved: slice_{i:02d}.png")

    # Summary visualization
    print("\n[Visualize] Generating summary...")
    dem_classifier = DEMClassifier()
    stats = dem_classifier.get_zone_statistics(ds["dem"].values)
    visualizer.export_summary(ds, stats)
    print("  Saved: carbon_summary.png")


if __name__ == "__main__":
    main()
