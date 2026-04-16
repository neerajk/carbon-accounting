"""
carbon_platform/datacube.py
===========================
Zarr DataCube initialization and management.
Handles spatial bounds, chunking, and CRS for Uttarakhand regions.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import xarray as xr
import zarr
from affine import Affine
import pyproj


# Dehradun-Mussoorie bounds in EPSG:4326 (WGS84)
REGION_BOUNDS = {
    "dehradun": {
        "min_lon": 77.8,
        "max_lon": 78.5,
        "min_lat": 30.2,
        "max_lat": 30.5,
    },
    "mussoorie": {
        "min_lon": 78.0,
        "max_lon": 78.2,
        "min_lat": 30.35,
        "max_lat": 30.55,
    },
}

# Forest type codes
FOREST_TYPES = {
    0: "Unknown",
    1: "Sal_Forest",
    2: "Chir_Pine",
    3: "Oak_Banj",
}


class DataCubeManager:
    """
    Manages Zarr DataCube for carbon accounting data.

    Dimensions: (y, x, time) with chunked storage for parallel processing.
    Variables: rgb, chm, dem, forest_class, carbon_density
    """

    def __init__(self, store_path: str, region: str = "dehradun", resolution: float = 2.0):
        """
        Initialize DataCube manager.

        Parameters
        ----------
        store_path : str
            Path to Zarr store directory
        region : str
            Region name ('dehradun' or 'mussoorie')
        resolution : float
            Pixel resolution in meters (default 2m for CHMv2)
        """
        self.store_path = Path(store_path)
        self.region = region
        self.resolution = resolution
        self.crs = "EPSG:3857"  # Web Mercator

        # Get bounds
        bounds = REGION_BOUNDS.get(region, REGION_BOUNDS["dehradun"])
        self.bounds_4326 = bounds

        # Convert to Web Mercator (EPSG:3857)
        self.bounds_3857 = self._wgs84_to_webmercator(bounds)

        # Calculate dimensions
        self.width_m = self.bounds_3857["max_x"] - self.bounds_3857["min_x"]
        self.height_m = self.bounds_3857["max_y"] - self.bounds_3857["min_y"]

        self.width_px = int(self.width_m / resolution)
        self.height_px = int(self.height_m / resolution)

        # Affine transform for georeferencing
        self.transform = Affine.identity()
        self.transform *= Affine.translation(self.bounds_3857["min_x"], self.bounds_3857["min_y"])
        self.transform *= Affine.scale(resolution, resolution)

    def _wgs84_to_webmercator(self, bounds: Dict) -> Dict:
        """Convert WGS84 bounds to Web Mercator (EPSG:3857)."""
        transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

        min_x, min_y = transformer.transform(bounds["min_lon"], bounds["min_lat"])
        max_x, max_y = transformer.transform(bounds["max_lon"], bounds["max_lat"])

        return {"min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y}

    def initialize_store(self, chunk_size: int = 512) -> xr.Dataset:
        """
        Create new Zarr DataCube with proper structure.

        Parameters
        ----------
        chunk_size : int
            Chunk size in pixels (default 512 matches DINOv3 input)

        Returns
        -------
        xr.Dataset
            Empty DataCube with all variables initialized
        """
        self.store_path.mkdir(parents=True, exist_ok=True)
        zarr_path = self.store_path / f"{self.region}_carbon.zarr"

        # Define coordinates
        x_coords = np.arange(self.width_px) * self.resolution + self.bounds_3857["min_x"]
        y_coords = np.arange(self.height_px) * self.resolution + self.bounds_3857["min_y"]

        # Create dataset with dask-backed arrays
        ds = xr.Dataset(
            {
                # RGB bands (uint8) - store as separate bands for efficiency
                "red": (
                    ["y", "x"],
                    zarr.zeros(
                        (self.height_px, self.width_px),
                        chunks=(chunk_size, chunk_size),
                        dtype="uint8",
                        store=str(zarr_path / "red"),
                    ),
                ),
                "green": (
                    ["y", "x"],
                    zarr.zeros(
                        (self.height_px, self.width_px),
                        chunks=(chunk_size, chunk_size),
                        dtype="uint8",
                        store=str(zarr_path / "green"),
                    ),
                ),
                "blue": (
                    ["y", "x"],
                    zarr.zeros(
                        (self.height_px, self.width_px),
                        chunks=(chunk_size, chunk_size),
                        dtype="uint8",
                        store=str(zarr_path / "blue"),
                    ),
                ),
                # Canopy Height Model (float32)
                "chm": (
                    ["y", "x"],
                    zarr.full(
                        (self.height_px, self.width_px),
                        fill_value=-9999.0,
                        chunks=(chunk_size, chunk_size),
                        dtype="float32",
                        store=str(zarr_path / "chm"),
                    ),
                ),
                # DEM elevation (float32)
                "dem": (
                    ["y", "x"],
                    zarr.full(
                        (self.height_px, self.width_px),
                        fill_value=-9999.0,
                        chunks=(chunk_size, chunk_size),
                        dtype="float32",
                        store=str(zarr_path / "dem"),
                    ),
                ),
                # Forest class (uint8) - 1=Sal, 2=Pine, 3=Oak
                "forest_class": (
                    ["y", "x"],
                    zarr.zeros(
                        (self.height_px, self.width_px),
                        chunks=(chunk_size, chunk_size),
                        dtype="uint8",
                        store=str(zarr_path / "forest_class"),
                    ),
                ),
                # Carbon density MgC/ha (float32)
                "carbon_density": (
                    ["y", "x"],
                    zarr.full(
                        (self.height_px, self.width_px),
                        fill_value=-9999.0,
                        chunks=(chunk_size, chunk_size),
                        dtype="float32",
                        store=str(zarr_path / "carbon_density"),
                    ),
                ),
                # AGB Mg/ha (float32)
                "agb": (
                    ["y", "x"],
                    zarr.full(
                        (self.height_px, self.width_px),
                        fill_value=-9999.0,
                        chunks=(chunk_size, chunk_size),
                        dtype="float32",
                        store=str(zarr_path / "agb"),
                    ),
                ),
            },
            coords={
                "x": (["x"], x_coords),
                "y": (["y"], y_coords),
            },
        )

        # Add CRS and spatial metadata
        ds.attrs["crs"] = self.crs
        ds.attrs["resolution"] = self.resolution
        ds.attrs["region"] = self.region
        ds.attrs["transform"] = list(self.transform)
        ds.attrs["bounds_3857"] = self.bounds_3857

        # Save to disk
        ds.to_zarr(zarr_path, mode="w")

        print(f"[DataCube] Initialized store at {zarr_path}")
        print(f"[DataCube] Dimensions: {self.height_px} x {self.width_px} pixels")
        print(f"[DataCube] Chunks: {chunk_size} x {chunk_size}")
        print(f"[DataCube] CRS: {self.crs}")

        return ds

    def open_store(self) -> xr.Dataset:
        """Open existing Zarr DataCube."""
        zarr_path = self.store_path / f"{self.region}_carbon.zarr"
        if not zarr_path.exists():
            raise FileNotFoundError(f"DataCube not found at {zarr_path}. Run initialize_store() first.")
        return xr.open_zarr(zarr_path)

    def get_chunk_bounds(self, chunk_idx: Tuple[int, int]) -> Dict:
        """
        Get spatial bounds for a specific chunk.

        Parameters
        ----------
        chunk_idx : tuple
            (row, col) chunk index

        Returns
        -------
        dict
            Bounds in Web Mercator coordinates
        """
        row, col = chunk_idx
        chunk_size = 512  # Default chunk size

        x_start = self.bounds_3857["min_x"] + col * chunk_size * self.resolution
        x_end = min(x_start + chunk_size * self.resolution, self.bounds_3857["max_x"])

        y_start = self.bounds_3857["min_y"] + row * chunk_size * self.resolution
        y_end = min(y_start + chunk_size * self.resolution, self.bounds_3857["max_y"])

        return {"min_x": x_start, "max_x": x_end, "min_y": y_start, "max_y": y_end}

    def write_chunk(self, ds: xr.Dataset, chunk_idx: Tuple[int, int], data: Dict[str, np.ndarray]):
        """Write data to a specific chunk in the DataCube."""
        row, col = chunk_idx
        chunk_size = 512

        y_slice = slice(row * chunk_size, (row + 1) * chunk_size)
        x_slice = slice(col * chunk_size, (col + 1) * chunk_size)

        for var_name, arr in data.items():
            if var_name in ds:
                ds[var_name][y_slice, x_slice] = arr
