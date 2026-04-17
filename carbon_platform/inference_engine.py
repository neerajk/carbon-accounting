"""
carbon_platform/inference_engine.py
======================================
Adaptive inference engine for CHMv2 with Dask parallelization.
Two-phase approach: Phase A (full district) and Phase B (ITC hotspots).
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
import torch
import dask.array as da
from dask import delayed
import xarray as xr
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.model import load_model_and_processor
from pipeline.inference import run_patch_inference
from pipeline.tiling import Patch
from .allometry import AllometryCalculator
from .dem_classifier import DEMClassifier


class CarbonInferenceEngine:
    """
    Dask-compatible inference engine for carbon accounting.

    Implements two-phase approach:
    - Phase A: Full district pixel-wise processing (ABA method)
    - Phase B: ITC detection in high-carbon hotspots (placeholder for YOLOv11)
    """

    def __init__(self, cfg: dict, allometry_path: str):
        """
        Initialize inference engine.

        Parameters
        ----------
        cfg : dict
            Configuration dictionary with model, device, batch_size settings
        allometry_path : str
            Path to allometry params CSV
        """
        self.cfg = cfg
        self.allometry = AllometryCalculator(allometry_path)
        self.dem_classifier = DEMClassifier()
        self.model = None
        self.processor = None
        self.device = None
        self._model_loaded = False

    def _load_model(self):
        """Lazy model loading for Dask workers."""
        if not self._model_loaded:
            self.model, self.processor, self.device = load_model_and_processor(self.cfg)
            self._model_loaded = True

    def process_chunk(
        self,
        rgb_chunk: np.ndarray,
        dem_chunk: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Process a single 512x512 chunk.

        Parameters
        ----------
        rgb_chunk : np.ndarray
            RGB data (H, W, 3) uint8
        dem_chunk : np.ndarray
            DEM elevation (H, W) float32

        Returns
        -------
        dict
            chm, forest_class, carbon_density, agb arrays
        """
        self._load_model()

        # Validate inputs
        if rgb_chunk.size == 0 or dem_chunk.size == 0:
            return self._empty_output(rgb_chunk.shape[:2])

        # Ensure correct shape
        if rgb_chunk.ndim == 2 and rgb_chunk.shape[2] != 3:
            raise ValueError(f"RGB chunk must be (H,W,3), got {rgb_chunk.shape}")

        # Create Patch wrapper
        class NativePatch:
            def __init__(self, arr):
                self.array = arr

        patch = NativePatch(rgb_chunk)

        # Run CHMv2 inference
        predictions, embeddings = run_patch_inference(
            [patch], self.model, self.processor, self.device, self.cfg
        )

        chm = predictions[0]  # Canopy height in meters

        # Classify forest type from DEM
        forest_class = self.dem_classifier.classify(dem_chunk)

        # Calculate carbon using allometry
        carbon_results = self.allometry.process_array(chm, forest_class)

        return {
            "chm": chm.astype(np.float32),
            "forest_class": forest_class.astype(np.uint8),
            "carbon_density": carbon_results["carbon_density"].astype(np.float32),
            "agb": carbon_results["agb"].astype(np.float32),
        }

    def _empty_output(self, shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """Return empty arrays for invalid chunks."""
        h, w = shape
        return {
            "chm": np.full((h, w), -9999.0, dtype=np.float32),
            "forest_class": np.zeros((h, w), dtype=np.uint8),
            "carbon_density": np.full((h, w), -9999.0, dtype=np.float32),
            "agb": np.full((h, w), -9999.0, dtype=np.float32),
        }

    def create_dask_graph(
        self,
        rgb_da: da.Array,
        dem_da: da.Array,
    ) -> Dict[str, da.Array]:
        """
        Create Dask computation graph for parallel processing.

        Parameters
        ----------
        rgb_da : da.Array
            Dask array of RGB (H, W, 3) - can be lazy
        dem_da : da.Array
            Dask array of DEM (H, W) - can be lazy

        Returns
        -------
        dict
            Dask arrays for chm, forest_class, carbon_density, agb
        """
        # Ensure chunks align
        chunk_size = self.cfg.get("tiling", {}).get("patch_size", 512)

        # Rechunk if necessary
        if rgb_da.chunksize[0] != chunk_size or rgb_da.chunksize[1] != chunk_size:
            rgb_da = rgb_da.rechunk((chunk_size, chunk_size, 3))
            dem_da = dem_da.rechunk((chunk_size, chunk_size))

        # Create delayed function
        @delayed
        def process_block(rgb_block, dem_block):
            return self.process_chunk(rgb_block, dem_block)

        # Map over blocks
        results = {}
        for key in ["chm", "forest_class", "carbon_density", "agb"]:
            results[key] = da.map_blocks(
                lambda rgb, dem: self.process_chunk(rgb, dem)[key],
                rgb_da,
                dem_da,
                dtype=np.float32 if key != "forest_class" else np.uint8,
                drop_axis=2 if key != "forest_class" else None,
            )

        return results

    def run_phase_a(
        self,
        ds: xr.Dataset,
        stream_loader,
    ) -> xr.Dataset:
        """
        Phase A: Full district pixel-wise processing (ABA method).

        Parameters
        ----------
        ds : xr.Dataset
            DataCube dataset
        stream_loader : StreamLoader
            Streaming data loader for tiles

        Returns
        -------
        xr.Dataset
            Updated dataset with computed variables
        """
        print("[Phase A] Starting full-district ABA processing...")
        print(f"[Phase A] Region: {self.cfg.get('region', 'unknown')}")

        # Get chunks from stream loader
        chunk_indices = stream_loader.get_chunk_indices()

        # Apply limit if specified
        limit = self.cfg.get("limit")
        if limit:
            chunk_indices = chunk_indices[:limit]
            print(f"[Phase A] Limit applied: processing first {limit} of {len(stream_loader.get_chunk_indices())} chunks")

        print(f"[Phase A] Total chunks to process: {len(chunk_indices)}")

        # Process each chunk
        for idx, (row, col) in enumerate(tqdm(chunk_indices, desc="Phase A Progress")):
            # Fetch data
            rgb_tile, dem_tile, transform = stream_loader.fetch_chunk(row, col)

            if rgb_tile is None or dem_tile is None:
                continue

            # Process
            results = self.process_chunk(rgb_tile, dem_tile)

            # --- THE FIX: Check target view shape before assignment to prevent crashing on edge fragments ---
            target_view = ds["red"][row * 512:(row + 1) * 512, col * 512:(col + 1) * 512]
            
            # Ensure both the available dataset space and the fetched tile are a perfect 512x512 fit
            if target_view.shape == (512, 512) and rgb_tile.shape[:2] == (512, 512):
                ds["red"][row * 512:(row + 1) * 512, col * 512:(col + 1) * 512] = rgb_tile[:, :, 0]
                ds["green"][row * 512:(row + 1) * 512, col * 512:(col + 1) * 512] = rgb_tile[:, :, 1]
                ds["blue"][row * 512:(row + 1) * 512, col * 512:(col + 1) * 512] = rgb_tile[:, :, 2]
                ds["dem"][row * 512:(row + 1) * 512, col * 512:(col + 1) * 512] = dem_tile
                ds["chm"][row * 512:(row + 1) * 512, col * 512:(col + 1) * 512] = results["chm"]
                ds["forest_class"][row * 512:(row + 1) * 512, col * 512:(col + 1) * 512] = results["forest_class"]
                ds["carbon_density"][row * 512:(row + 1) * 512, col * 512:(col + 1) * 512] = results["carbon_density"]
                ds["agb"][row * 512:(row + 1) * 512, col * 512:(col + 1) * 512] = results["agb"]
            else:
                # Option 1: Skip and warn
                print(f"\n⚠️ Skipping edge fragment at row {row}, col {col} (Target Shape: {target_view.shape}, Tile Shape: {rgb_tile.shape[:2]})")
                pass

            # Periodic save
            # Periodic save
            if (idx + 1) % 10 == 0:
                ds.to_zarr(self.cfg["output"]["output_dir"], mode="a", consolidated=False)
                print(f"[Phase A] Saved checkpoint after {idx + 1} chunks")

        # Final save
        # Final save
        ds.to_zarr(self.cfg["output"]["output_dir"], mode="a", consolidated=False)
        print("[Phase A] Complete. Data saved to Zarr store.")

        return ds

    def run_phase_b(
        self,
        ds: xr.Dataset,
        carbon_threshold: float = 50.0,
    ) -> Dict:
        """
        Phase B: ITC detection in high-carbon hotspots.

        Placeholder for YOLOv11 individual tree crown detection.

        Parameters
        ----------
        ds : xr.Dataset
            DataCube with Phase A results
        carbon_threshold : float
            Threshold for hotspot detection (MgC/ha)

        Returns
        -------
        dict
            Hotspot coordinates and statistics
        """
        print("[Phase B] Identifying high-carbon hotspots...")
        print(f"[Phase B] Carbon threshold: {carbon_threshold} MgC/ha")

        # Find hotspots
        carbon = ds["carbon_density"].values
        hotspots = carbon > carbon_threshold

        hotspot_coords = np.argwhere(hotspots)
        print(f"[Phase B] Found {len(hotspot_coords)} hotspot pixels")

        # TODO: Integrate YOLOv11 for ITC detection
        # For now, return placeholder
        results = {
            "hotspot_count": len(hotspot_coords),
            "hotspot_coords": hotspot_coords,
            "total_carbon_in_hotspots": float(np.sum(carbon[hotspots])),
            "yolo_ready": False,  # Flag for future YOLO integration
        }

        print("[Phase B] Placeholder complete. YOLOv11 integration pending.")
        return results