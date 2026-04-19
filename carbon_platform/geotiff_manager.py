"""
carbon_platform/geotiff_manager.py
===================================
GeoTIFF-based storage manager for carbon accounting data.

Replaces Zarr-based DataCubeManager with simple individual GeoTIFF files.
Each layer (CHM, carbon density, forest class, DEM, AGB) is saved as a separate GeoTIFF.

Advantages over Zarr:
- Compatible with QGIS, ArcGIS, standard GIS tools
- Simpler file I/O (no Zarr chunking complexity)
- Easier debugging (inspect individual .tif files)
- Standard format with proper georeferencing
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import rasterio
from rasterio.transform import Affine
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

# Layers to save
LAYERS = {
    "chm": {"dtype": "float32", "nodata": -9999.0, "description": "Canopy Height Model (m)"},
    "carbon_density": {"dtype": "float32", "nodata": -9999.0, "description": "Carbon Density (MgC/ha)"},
    "forest_class": {"dtype": "uint8", "nodata": 0, "description": "Forest Type (1=Sal, 2=Pine, 3=Oak)"},
    "dem": {"dtype": "float32", "nodata": -9999.0, "description": "Digital Elevation Model (m)"},
    "agb": {"dtype": "float32", "nodata": -9999.0, "description": "Above-Ground Biomass (Mg/ha)"},
    "red": {"dtype": "uint8", "nodata": 0, "description": "Red band (ESRI imagery)"},
    "green": {"dtype": "uint8", "nodata": 0, "description": "Green band (ESRI imagery)"},
    "blue": {"dtype": "uint8", "nodata": 0, "description": "Blue band (ESRI imagery)"},
}


class GeoTIFFManager:
    """
    Manage geospatial carbon accounting data using individual GeoTIFF files.

    Each variable (CHM, carbon, forest class, etc.) is stored as a separate GeoTIFF
    with proper georeferencing (Affine transform, CRS).

    Parameters
    ----------
    output_dir : str
        Directory for output GeoTIFF files
    region : str
        Region name ('dehradun' or 'mussoorie')
    resolution : float
        Pixel resolution in meters (default 2m for CHMv2)
    """

    def __init__(self, output_dir: str, region: str = "dehradun", resolution: float = 2.0):
        """Initialize GeoTIFF manager."""
        self.output_dir = Path(output_dir)
        self.region = region
        self.resolution = resolution
        self.crs = "EPSG:3857"  # Web Mercator

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

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

        # Rasterio profile template
        self.profile = {
            "driver": "GTiff",
            "width": self.width_px,
            "height": self.height_px,
            "count": 1,
            "dtype": "float32",
            "crs": self.crs,
            "transform": self.transform,
            "compress": "lzw",
        }

        print(f"[GeoTIFFManager] Initialized for {region.upper()}")
        print(f"  Resolution: {resolution}m")
        print(f"  Dimensions: {self.height_px} x {self.width_px} pixels")
        print(f"  Output dir: {self.output_dir}")

    def _wgs84_to_webmercator(self, bounds: Dict) -> Dict:
        """Convert WGS84 bounds to Web Mercator (EPSG:3857)."""
        transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

        min_x, min_y = transformer.transform(bounds["min_lon"], bounds["min_lat"])
        max_x, max_y = transformer.transform(bounds["max_lon"], bounds["max_lat"])

        return {"min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y}

    def get_geotiff_path(self, layer: str) -> Path:
        """Get path to GeoTIFF file for a layer."""
        return self.output_dir / f"{layer}.tif"

    def get_geotiff_paths(self) -> Dict[str, Path]:
        """Get paths to all GeoTIFF files."""
        return {layer: self.get_geotiff_path(layer) for layer in LAYERS.keys()}

    def initialize_geotiffs(self) -> None:
        """
        Create empty GeoTIFF files for all layers.

        This initializes all output files with correct geospatial metadata
        but no data (will be filled incrementally).
        """
        print(f"\n[GeoTIFFManager] Initializing empty GeoTIFF files...")

        for layer_name, layer_info in LAYERS.items():
            out_path = self.get_geotiff_path(layer_name)

            # Update profile for this layer
            layer_profile = self.profile.copy()
            layer_profile["dtype"] = layer_info["dtype"]
            layer_profile["nodata"] = layer_info["nodata"]

            # Create empty file
            with rasterio.open(out_path, "w", **layer_profile) as dst:
                # Write empty array
                if layer_info["dtype"] == "uint8":
                    empty = np.zeros((self.height_px, self.width_px), dtype=np.uint8)
                else:
                    empty = np.full((self.height_px, self.width_px), layer_info["nodata"], dtype=np.float32)

                dst.write(empty, 1)
                dst.set_band_description(1, layer_info["description"])

            print(f"  Created: {out_path}")

    def write_layer_chunk(
        self,
        layer: str,
        data: np.ndarray,
        y_start: int,
        x_start: int,
    ) -> None:
        """
        Write a chunk of data to a GeoTIFF layer.

        Parameters
        ----------
        layer : str
            Layer name ('chm', 'carbon_density', 'forest_class', 'dem', 'agb')
        data : np.ndarray
            Data to write (H, W)
        y_start : int
            Starting row index
        x_start : int
            Starting column index
        """
        out_path = self.get_geotiff_path(layer)
        layer_info = LAYERS[layer]

        # Create window for this chunk
        y_end = min(y_start + data.shape[0], self.height_px)
        x_end = min(x_start + data.shape[1], self.width_px)

        actual_h = y_end - y_start
        actual_w = x_end - x_start

        # Trim data if necessary
        if data.shape[0] > actual_h or data.shape[1] > actual_w:
            data = data[:actual_h, :actual_w]

        # Convert NaN to nodata value
        if layer_info["dtype"] == "float32":
            data_to_write = np.copy(data).astype(np.float32)
            data_to_write[np.isnan(data_to_write)] = layer_info["nodata"]
        else:
            data_to_write = data.astype(layer_info["dtype"])

        # Write to file using rasterio window
        from rasterio.windows import Window

        window = Window(x_start, y_start, actual_w, actual_h)

        with rasterio.open(out_path, "r+") as dst:
            dst.write(data_to_write, 1, window=window)

    def read_layer(self, layer: str) -> np.ndarray:
        """
        Read entire layer into memory.

        Parameters
        ----------
        layer : str
            Layer name

        Returns
        -------
        np.ndarray
            Full layer data (H, W)
        """
        out_path = self.get_geotiff_path(layer)

        with rasterio.open(out_path) as src:
            return src.read(1)

    def read_layer_slice(self, layer: str, y_slice: slice, x_slice: slice) -> np.ndarray:
        """
        Read a slice of a layer (for visualization).

        Parameters
        ----------
        layer : str
            Layer name
        y_slice : slice
            Row slice
        x_slice : slice
            Column slice

        Returns
        -------
        np.ndarray
            Sliced layer data
        """
        out_path = self.get_geotiff_path(layer)

        y_start = y_slice.start or 0
        y_stop = y_slice.stop or self.height_px
        x_start = x_slice.start or 0
        x_stop = x_slice.stop or self.width_px

        from rasterio.windows import Window

        window = Window(x_start, y_start, x_stop - x_start, y_stop - y_start)

        with rasterio.open(out_path) as src:
            return src.read(1, window=window)

    def load_all_layers(self) -> Dict[str, np.ndarray]:
        """
        Load all layers into memory as a dictionary.

        Returns
        -------
        dict
            {layer_name: np.ndarray} for all layers
        """
        print(f"\n[GeoTIFFManager] Loading all layers into memory...")

        data_dict = {}
        for layer in LAYERS.keys():
            print(f"  Loading {layer}...")
            data_dict[layer] = self.read_layer(layer)

        return data_dict

    def get_profile(self) -> dict:
        """Get rasterio profile for reference."""
        return self.profile.copy()

    def get_metadata(self) -> dict:
        """Get spatial metadata."""
        return {
            "region": self.region,
            "resolution": self.resolution,
            "crs": self.crs,
            "transform": str(self.transform),
            "bounds_3857": self.bounds_3857,
            "bounds_4326": self.bounds_4326,
            "dimensions": {
                "height_px": self.height_px,
                "width_px": self.width_px,
                "height_m": self.height_m,
                "width_m": self.width_m,
            },
        }
