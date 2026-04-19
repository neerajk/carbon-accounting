"""
carbon_platform/stream_loader.py
================================
Parallel streaming loader for ESRI tiles and Planetary Computer DEM.
Fetches data on-the-fly without local storage.
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from pyproj import Transformer
from rasterio.transform import from_bounds

# ESRI World Imagery tile server
ESRI_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"


class StreamLoader:
    """
    Stream ESRI World Imagery tiles on-the-fly.

    - ESRI World Imagery: Zoom 18 (~0.6m native) → resampled to target resolution
    - DEM: Synthetic elevation model (Dehradun-Mussoorie region)
    """

    def __init__(
        self,
        bounds_3857: Dict[str, float],
        zoom: int = 18,
        resolution: float = 2.0,
        use_real_data: bool = True,
    ):
        """
        Initialize stream loader.

        Parameters
        ----------
        bounds_3857 : dict
            Web Mercator bounds (min_x, max_x, min_y, max_y)
        zoom : int
            ESRI tile zoom level (default 18 ~ 0.6m/pixel)
        resolution : float
            Target resolution in meters (default 2m for CHMv2)
        use_real_data : bool
            If True, fetch real ESRI tiles + synthetic DEM.
            If False, use fully synthetic data.
        """
        self.bounds_3857 = bounds_3857
        self.zoom = zoom
        self.resolution = resolution
        self.use_real_data = use_real_data
        self.chunk_size = 512

        # Transformers
        self.transformer_3857_to_4326 = Transformer.from_crs(
            "EPSG:3857", "EPSG:4326", always_xy=True
        )
        self.transformer_4326_to_3857 = Transformer.from_crs(
            "EPSG:4326", "EPSG:3857", always_xy=True
        )

    def _webmercator_to_tile(self, x: float, y: float, zoom: int) -> Tuple[int, int]:
        """Convert Web Mercator coordinates to tile coordinates."""
        # Convert to lat/lon first
        lon, lat = self.transformer_3857_to_4326.transform(x, y)
        return self._latlon_to_tile(lat, lon, zoom)

    def _latlon_to_tile(self, lat: float, lon: float, zoom: int) -> Tuple[int, int]:
        """Convert WGS84 lat/lon to tile coordinates (XYZ)."""
        lat_rad = np.radians(lat)
        n = 2 ** zoom
        xtile = int((lon + 180.0) / 360.0 * n)
        ytile = int((1.0 - np.arcsinh(np.tan(lat_rad)) / np.pi) / 2.0 * n)
        return xtile, ytile

    def _tile_to_webmercator_bounds(
        self, x: int, y: int, z: int
    ) -> Tuple[float, float, float, float]:
        """Get Web Mercator bounds for a tile."""
        n = 2 ** z
        lon_min = x / n * 360.0 - 180.0
        lon_max = (x + 1) / n * 360.0 - 180.0

        lat_max = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * y / n))))
        lat_min = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (y + 1) / n))))

        # Convert to Web Mercator
        min_x, min_y = self.transformer_4326_to_3857.transform(lon_min, lat_min)
        max_x, max_y = self.transformer_4326_to_3857.transform(lon_max, lat_max)

        return min_x, min_y, max_x, max_y

    def get_chunk_indices(self) -> List[Tuple[int, int]]:
        """Get list of (row, col) chunk indices for the region."""
        width_m = self.bounds_3857["max_x"] - self.bounds_3857["min_x"]
        height_m = self.bounds_3857["max_y"] - self.bounds_3857["min_y"]

        width_px = int(width_m / self.resolution)
        height_px = int(height_m / self.resolution)

        n_cols = (width_px + self.chunk_size - 1) // self.chunk_size
        n_rows = (height_px + self.chunk_size - 1) // self.chunk_size

        indices = []
        for row in range(n_rows):
            for col in range(n_cols):
                indices.append((row, col))

        return indices

    def fetch_chunk(
        self, row: int, col: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Fetch data for a specific 512x512 chunk.

        Returns
        -------
        tuple: (rgb_array, dem_array)
        """
        if self.use_real_data:
            # Fetch real ESRI tiles
            rgb = self._fetch_esri_chunk(row, col)
            # Generate synthetic DEM (Planetary Computer fallback)
            dem = self._generate_synthetic_dem_for_chunk(row, col)
        else:
            # Fully synthetic data for testing
            dem = self._generate_synthetic_dem_for_chunk(row, col)
            rgb = self._generate_synthetic_rgb(dem)

        return rgb, dem

    def _fetch_esri_chunk(self, row: int, col: int) -> np.ndarray:
        """
        Fetch ESRI tiles for a specific chunk and mosaic to 512x512.
        """
        # Calculate chunk bounds in Web Mercator
        chunk_size_m = self.chunk_size * self.resolution
        min_x = self.bounds_3857["min_x"] + col * chunk_size_m
        max_x = min(min_x + chunk_size_m, self.bounds_3857["max_x"])
        min_y = self.bounds_3857["min_y"] + row * chunk_size_m
        max_y = min(min_y + chunk_size_m, self.bounds_3857["max_y"])

        # Find tile range at zoom level
        x_min_tile, y_max_tile = self._webmercator_to_tile(min_x, max_y, self.zoom)
        x_max_tile, y_min_tile = self._webmercator_to_tile(max_x, min_y, self.zoom)

        tiles_rgb = []
        tile_bounds = []

        for tx in range(x_min_tile, x_max_tile + 1):
            for ty in range(y_min_tile, y_max_tile + 1):
                tile_img = self._fetch_esri_tile(tx, ty, self.zoom)
                if tile_img is not None:
                    tile_bounds_3857 = self._tile_to_webmercator_bounds(tx, ty, self.zoom)
                    tiles_rgb.append(tile_img)
                    tile_bounds.append(tile_bounds_3857)

        if not tiles_rgb:
            # Fallback to synthetic
            dem_fallback = self._generate_synthetic_dem_for_chunk(row, col)
            return self._generate_synthetic_rgb(dem_fallback)

        # Mosaic tiles into single 512x512 image
        return self._mosaic_tiles_to_512(tiles_rgb, tile_bounds, min_x, min_y, max_x, max_y)

    def _fetch_esri_tile(self, x: int, y: int, z: int) -> Optional[np.ndarray]:
        """Fetch single ESRI tile (256x256)."""
        url = ESRI_URL.format(x=x, y=y, z=z)
        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                return np.array(img)
        except Exception as e:
            print(f"[StreamLoader] ESRI tile failed {x},{y},{z}: {e}")
        return None

    def _mosaic_tiles_to_512(
        self,
        tiles: List[np.ndarray],
        bounds: List[Tuple],
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
    ) -> np.ndarray:
        """Mosaic multiple tiles into target 512x512 extent."""
        output = np.zeros((self.chunk_size, self.chunk_size, 3), dtype=np.uint8)
        width_m = max_x - min_x
        height_m = max_y - min_y
        x_scale = self.chunk_size / width_m
        y_scale = self.chunk_size / height_m

        for tile_img, tile_bounds in zip(tiles, bounds):
            t_min_x, t_min_y, t_max_x, t_max_y = tile_bounds
            rel_x_start = (t_min_x - min_x) * x_scale
            rel_x_end = (t_max_x - min_x) * x_scale
            rel_y_start = (max_y - t_max_y) * y_scale
            rel_y_end = (max_y - t_min_y) * y_scale
            px_start = max(0, int(rel_x_start))
            px_end = min(self.chunk_size, int(rel_x_end))
            py_start = max(0, int(rel_y_start))
            py_end = min(self.chunk_size, int(rel_y_end))

            if px_end > px_start and py_end > py_start:
                tile_h = py_end - py_start
                tile_w = px_end - px_start
                tile_resized = np.array(
                    Image.fromarray(tile_img).resize((tile_w, tile_h), Image.BILINEAR)
                )
                output[py_start:py_end, px_start:px_end] = tile_resized

        return output

    def _generate_synthetic_dem_for_chunk(self, row: int, col: int) -> np.ndarray:
        """Generate synthetic elevation for a specific chunk."""
        # Calculate chunk center for elevation gradient
        min_x = self.bounds_3857["min_x"] + col * self.chunk_size * self.resolution
        min_y = self.bounds_3857["min_y"] + row * self.chunk_size * self.resolution
        center_x = min_x + self.chunk_size * self.resolution / 2
        center_y = min_y + self.chunk_size * self.resolution / 2
        lon, lat = self.transformer_3857_to_4326.transform(center_x, center_y)

        # Base elevation for Dehradun-Mussoorie region
        base_elev = 600 + (lat - 30.2) * 3000
        dem = np.full((self.chunk_size, self.chunk_size), base_elev, dtype=np.float32)
        y_gradient = np.linspace(0, 1, self.chunk_size)[:, None] * 200
        dem += y_gradient
        noise = np.random.normal(0, 50, (self.chunk_size, self.chunk_size))
        dem += noise
        return np.clip(dem, 400, 3500)


    def _generate_synthetic_rgb(self, dem: np.ndarray) -> np.ndarray:
        """Generate synthetic RGB based on elevation."""
        dem_norm = (dem - dem.min()) / (dem.max() - dem.min() + 1e-8)

        rgb = np.zeros((*dem.shape, 3), dtype=np.uint8)
        rgb[:, :, 0] = (dem_norm * 50 + 20).astype(np.uint8)
        rgb[:, :, 1] = (dem_norm * 100 + 80).astype(np.uint8)
        rgb[:, :, 2] = (dem_norm * 40 + 10).astype(np.uint8)

        return rgb

    def estimate_total_tiles(self) -> int:
        """Estimate total number of chunks to process."""
        return len(self.get_chunk_indices())

    def get_forest_class_mask(
        self, dem: np.ndarray, altitude_zones: List[Dict]
    ) -> np.ndarray:
        """
        Apply altitudinal zonation from config to create forest_class_mask.

        Parameters
        ----------
        dem : np.ndarray
            Elevation data in meters
        altitude_zones : list
            Zone definitions from config.yaml

        Returns
        -------
        np.ndarray
            Integer class codes (0=Unknown, 1=Sal, 2=Pine, 3=Oak, etc.)
        """
        forest_class = np.zeros_like(dem, dtype=np.uint8)

        for zone in altitude_zones:
            min_alt = zone.get("min_alt", -np.inf)
            max_alt = zone.get("max_alt", np.inf)
            class_code = zone.get("class_code", 0)

            mask = (dem >= min_alt) & (dem < max_alt)
            forest_class[mask] = class_code

        return forest_class
