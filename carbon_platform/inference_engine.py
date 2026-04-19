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
        geotiff_manager,
        stream_loader,
    ) -> dict:
        """
        Phase A: Full district pixel-wise processing (ABA method).

        CORRECT FLOW:
        1. Fetch ESRI + DEM together → keep patches paired
        2. Run CHMv2 model → Canopy Height Model (CHM)
        3. Use DEM on CHM output → classify forest type
        4. Visualize: RGB, DEM, CHM, Forest Class
        5. Use allometry_params.csv → calculate AGB from CHM + forest_class
        6. Calculate Carbon = 47% of AGB
        7. Save all to GeoTIFF + unified visualization

        Parameters
        ----------
        geotiff_manager : GeoTIFFManager
            Manager for GeoTIFF storage
        stream_loader : StreamLoader
            Streaming data loader for tiles

        Returns
        -------
        dict
            Data dictionary with all layers for visualization (first chunk)
        """
        print("[Phase A] Starting full-district ABA processing...")
        print(f"[Phase A] Region: {self.cfg.get('region', 'unknown')}")

        chunk_indices = stream_loader.get_chunk_indices()
        limit = self.cfg.get("limit")
        if limit:
            chunk_indices = chunk_indices[:limit]
            print(f"[Phase A] Limit applied: processing first {limit} chunks")

        print(f"[Phase A] Total chunks to process: {len(chunk_indices)}")

        # Storage for full arrays
        all_data = {
            "red": [],
            "green": [],
            "blue": [],
            "dem": [],
            "chm": [],
            "forest_class": [],
            "agb": [],
            "carbon_density": [],
        }

        first_chunk_data = None

        for idx, (row, col) in enumerate(tqdm(chunk_indices, desc="Phase A Progress")):
            y_start = row * 512
            x_start = col * 512
            y_end = min(y_start + 512, geotiff_manager.height_px)
            x_end = min(x_start + 512, geotiff_manager.width_px)

            # ============================================================
            # STEP 1: Fetch ESRI + DEM together (keep patches paired)
            # ============================================================
            rgb_tile, dem_tile = stream_loader.fetch_chunk(row, col)

            if rgb_tile is None or dem_tile is None:
                print(f"  Skipping chunk ({row}, {col}) - no data")
                continue

            # ============================================================
            # STEP 2: Run CHMv2 model → Canopy Height
            # ============================================================
            self._load_model()

            class NativePatch:
                def __init__(self, arr):
                    self.array = arr

            patch = NativePatch(rgb_tile)

            from pipeline.inference import run_patch_inference
            predictions, _ = run_patch_inference(
                [patch], self.model, self.processor, self.device, self.cfg
            )
            chm = predictions[0].astype(np.float32)

            # ============================================================
            # STEP 3: Use DEM on CHM output → classify forest type
            # ============================================================
            forest_class = self.dem_classifier.classify(dem_tile)

            # ============================================================
            # STEP 4: Calculate AGB using allometry_params.csv
            #         DBH = a × H^b (regional coefficients)
            #         AGB = 0.0673 × (ρ × DBH² × H)^0.976 (Chave 2014)
            # ============================================================
            carbon_results = self.allometry.process_array(chm, forest_class)
            agb = carbon_results["agb"].astype(np.float32)

            # ============================================================
            # STEP 5: Calculate Carbon = 47% of AGB
            # ============================================================
            carbon_density = carbon_results["carbon_density"].astype(np.float32)

            # ============================================================
            # STEP 6: Write to GeoTIFF
            # ============================================================
            geotiff_manager.write_layer_chunk("chm", chm, y_start, x_start)
            geotiff_manager.write_layer_chunk("dem", dem_tile.astype(np.float32), y_start, x_start)
            geotiff_manager.write_layer_chunk("forest_class", forest_class.astype(np.uint8), y_start, x_start)
            geotiff_manager.write_layer_chunk("agb", agb, y_start, x_start)
            geotiff_manager.write_layer_chunk("carbon_density", carbon_density, y_start, x_start)
            geotiff_manager.write_layer_chunk("red", rgb_tile[:, :, 0].astype(np.uint8), y_start, x_start)
            geotiff_manager.write_layer_chunk("green", rgb_tile[:, :, 1].astype(np.uint8), y_start, x_start)
            geotiff_manager.write_layer_chunk("blue", rgb_tile[:, :, 2].astype(np.uint8), y_start, x_start)

            # Store for return
            all_data["red"].append(rgb_tile[:, :, 0].astype(np.uint8))
            all_data["green"].append(rgb_tile[:, :, 1].astype(np.uint8))
            all_data["blue"].append(rgb_tile[:, :, 2].astype(np.uint8))
            all_data["dem"].append(dem_tile.astype(np.float32))
            all_data["chm"].append(chm)
            all_data["forest_class"].append(forest_class.astype(np.uint8))
            all_data["agb"].append(agb)
            all_data["carbon_density"].append(carbon_density)

            # Capture first complete chunk for visualization
            if first_chunk_data is None:
                first_chunk_data = {
                    "red": rgb_tile[:, :, 0].astype(np.uint8),
                    "green": rgb_tile[:, :, 1].astype(np.uint8),
                    "blue": rgb_tile[:, :, 2].astype(np.uint8),
                    "dem": dem_tile.astype(np.float32),
                    "chm": chm,
                    "forest_class": forest_class.astype(np.uint8),
                    "agb": agb,
                    "carbon_density": carbon_density,
                }
                print(f"\n[Phase A] Captured first chunk (row={row}, col={col}) for visualization")

            if (idx + 1) % 10 == 0:
                print(f"[Phase A] Processed {idx + 1} chunks")

        print("[Phase A] Complete. Data saved to GeoTIFF files.")

        return first_chunk_data

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