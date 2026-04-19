"""
Quick test script to validate the CarbonVisualizer output without running the whole pipeline.
Generates synthetic RGB/DEM/CHM/forest arrays and produces a single PNG.
"""
import numpy as np
from pathlib import Path
import sys
# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Import visualizer module directly to avoid running package __init__ (which may require heavy deps)
from importlib.util import spec_from_file_location, module_from_spec
viz_path = Path(__file__).resolve().parents[1] / "carbon_platform" / "visualizer.py"
spec = spec_from_file_location("carbon_platform.visualizer", str(viz_path))
viz_mod = module_from_spec(spec)
spec.loader.exec_module(viz_mod)
CarbonVisualizer = viz_mod.CarbonVisualizer

out_dir = Path("outputs/visualizations")
out_dir.mkdir(parents=True, exist_ok=True)

# Synthetic data
H = 512
W = 512
rgb = (np.random.rand(H, W, 3) * 255).astype(np.uint8)
# DEM: smooth gradient from 800m to 2200m
dem = np.linspace(800, 2200, W)[None, :].repeat(H, axis=0).astype(float)
# CHM: random canopy heights between 0 and 40m, with some zeros
chm = (np.abs(np.random.randn(H, W)) * 7).astype(float)
chm[np.random.rand(H, W) < 0.2] = 0.0
# Forest classes: 1,2,3
forest_class = np.random.choice([1, 2, 3], size=(H, W), p=[0.5, 0.3, 0.2])

# Do not provide carbon_density to force visualizer to compute AGB and carbon
carbon_density = None

# Path to allometry params (existing in repository)
allometry_path = "config/allometry_params.csv"

viz = CarbonVisualizer(str(out_dir), allometry_path=allometry_path, carbon_fraction=0.47)
viz.visualize_single_patch(rgb, dem, chm, forest_class, carbon_density, patch_idx=0)
print("Test visualization created in outputs/visualizations")
