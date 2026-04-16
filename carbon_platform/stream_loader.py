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
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.transform import from_bounds
from pyproj import Transformer
import warnings

# Planetary Computer imports
try:
    import pystac_client
    import planetary_computer
    import stackstac
    PC_AVAILABLE = True
except ImportError:
    PC_AVAILABLE = False
    warnings.warn("Planetary Computer libraries not available. Using synthetic DEM.")


# ESRI World Imagery tile server
ESRI_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

# Planetary Computer STAC API
STAC_API = "https://planetarycomputer.microsoft.com/api/stac/v1"
COP_DEM_COLLECTION = "cop-dem-glo-30"


class StreamLoader:
    """
    Stream ESRI tiles and COP DEM GLO-30 on-the-fly.

    - ESRI World Imagery: Zoom 18 (~0.6m native) → resampled to target resolution
    - Planetary Computer DEM: 30m native → resampled to match RGB grid (512x512)
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
            If True, fetch real ESRI tiles and Planetary Computer DEM.
            If False, use synthetic data.
        """
        self.bounds_3857 = bounds_3857
        self.zoom = zoom
        self.resolution = resolution
        self.use_real_data = use_real_data and PC_AVAILABLE
        self.chunk_size = 512

        # Transformers
        self.transformer_3857_to_4326 = Transformer.from_crs(
            "EPSG:3857", "EPSG:4326", always_xy=True
        )
        self.transformer_4326_to_3857 = Transformer.from_crs(
            "EPSG:4326", "EPSG:3857", always_xy=True
        )

        # Initialize Planetary Computer client
        self.pc_catalog = None
        if self.use_real_data and PC_AVAILABLE:
            try:
                self.pc_catalog = pystac_client.Client.open(
                    STAC_API,
                    modifier=planetary_computer.sign_inplace,
                )
                print("[StreamLoader] Connected to Planetary Computer STAC API")
            except Exception as e:
                print(f"[StreamLoader] Failed to connect to Planetary Computer: {e}")
                self.use_real_data = False

        # Pre-fetch DEM search (we'll reuse this)
        self.dem_item = None
        if self.use_real_data:
            self._prefetch_dem()

    def _prefetch_dem(self):
        """Search for DEM coverage over the region."""
        if not self.pc_catalog:
            return

        try:
            # Convert bounds to WGS84 for STAC search
            min_lon, min_lat = self.transformer_3857_to_4326.transform(
                self.bounds_3857["min_x"], self.bounds_3857["min_y"]
            )
            max_lon, max_lat = self.transformer_3857_to_4326.transform(
                self.bounds_3857["max_x"], self.bounds_3857["max_y"]
            )

            search = self.pc_catalog.search(
                collections=[COP_DEM_COLLECTION],
                bbox=[min_lon, min_lat, max_lon, max_lat],
            )
            items = list(search.items())

            if items:
                self.dem_item = items[0]  # Use first item that covers the area
                print(f"[StreamLoader] Found DEM item: {self.dem_item.id}")
            else:
                print("[StreamLoader] No DEM coverage found, will use synthetic")
                self.use_real_data = False

        except Exception as e:
            print(f"[StreamLoader] DEM prefetch failed: {e}")
            self.use_real_data = False

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
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], rasterio.Affine]:
        """
        Fetch data for a specific 512x512 chunk.

        Returns
        -------
        tuple: (rgb_array, dem_array, affine_transform)
        """
        # Calculate chunk bounds in Web Mercator
        chunk_size_m = self.chunk_size * self.resolution
        min_x = self.bounds_3857["min_x"] + col * chunk_size_m
        max_x = min(min_x + chunk_size_m, self.bounds_3857["max_x"])
        min_y = self.bounds_3857["min_y"] + row * chunk_size_m
        max_y = min(min_y + chunk_size_m, self.bounds_3857["max_y"])

        # Create affine transform for this chunk
        transform = from_bounds(min_x, min_y, max_x, max_y, self.chunk_size, self.chunk_size)

        if self.use_real_data:
            rgb = self._fetch_esri_for_bounds(min_x, min_y, max_x, max_y)
            dem = self._fetch_dem_for_bounds(min_x, min_y, max_x, max_y)
        else:
            # Synthetic data
            dem = self._generate_synthetic_dem((min_x, min_y, max_x, max_y))
            rgb = self._generate_synthetic_rgb(dem)

        return rgb, dem, transform

    def _fetch_esri_for_bounds(
        self, min_x: float, min_y: float, max_x: float, max_y: float
    ) -> np.ndarray:
        """
        Fetch and mosaic ESRI tiles to cover the given bounds.
        """
        # Find tile range
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
            return self._generate_synthetic_rgb(
                self._generate_synthetic_dem((min_x, min_y, max_x, max_y))
            )

        # Mosaic tiles into single image at target resolution
        return self._mosaic_tiles(tiles_rgb, tile_bounds, min_x, min_y, max_x, max_y)

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

    def _mosaic_tiles(
        self,
        tiles: List[np.ndarray],
        bounds: List[Tuple],
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
    ) -> np.ndarray:
        """Mosaic multiple tiles into target extent at 512x512."""
        # Create output canvas
        output = np.zeros((self.chunk_size, self.chunk_size, 3), dtype=np.uint8)

        # Calculate pixel scale
        width_m = max_x - min_x
        height_m = max_y - min_y
        x_scale = self.chunk_size / width_m
        y_scale = self.chunk_size / height_m

        for tile_img, tile_bounds in zip(tiles, bounds):
            # Calculate pixel overlap
            t_min_x, t_min_y, t_max_x, t_max_y = tile_bounds

            # Relative coordinates
            rel_x_start = (t_min_x - min_x) * x_scale
            rel_x_end = (t_max_x - min_x) * x_scale
            rel_y_start = (max_y - t_max_y) * y_scale  # Flip Y
            rel_y_end = (max_y - t_min_y) * y_scale

            # Convert to pixel coords
            px_start = max(0, int(rel_x_start))
            px_end = min(self.chunk_size, int(rel_x_end))
            py_start = max(0, int(rel_y_start))
            py_end = min(self.chunk_size, int(rel_y_end))

            if px_end > px_start and py_end > py_start:
                # Resize tile to fit
                tile_h = py_end - py_start
                tile_w = px_end - px_start
                tile_resized = np.array(
                    Image.fromarray(tile_img).resize((tile_w, tile_h), Image.BILINEAR)
                )

                output[py_start:py_end, px_start:px_end] = tile_resized

        return output

    def _fetch_dem_for_bounds(
        self, min_x: float, min_y: float, max_x: float, max_y: float
    ) -> np.ndarray:
        """
        Fetch COP DEM GLO-30 from Planetary Computer and resample to 512x512.
        """
        if not self.dem_item or not PC_AVAILABLE:
            return self._generate_synthetic_dem((min_x, min_y, max_x, max_y))

        try:
            # Convert bounds to WGS84 for STAC
            min_lon, min_lat = self.transformer_3857_to_4326.transform(min_x, min_y)
            max_lon, max_lat = self.transformer_3857_to_4326.transform(max_x, max_y)
            bbox = [min_lon, min_lat, max_lon, max_lat]

            # Load DEM data using stackstac
            dem_stack = stackstac.stack(
                [self.dem_item],
                assets=["data"],
                bounds=bbox,
                epsg=3857,
                resolution=self.resolution,
                dtype="float32",
            )

            # Compute (this triggers the actual read)
            dem_data = dem_stack.compute()

            if dem_data.size == 0:
                raise ValueError("DEM data empty")

            # Extract array and resample to 512x512
            dem_arr = dem_data.values[0, 0]  # First time, first band

            # Resize to 512x512 if needed
            if dem_arr.shape != (self.chunk_size, self.chunk_size):
                dem_arr = np.array(
                    Image.fromarray(dem_arr).resize(
                        (self.chunk_size, self.chunk_size), Image.BILINEAR
                    )
                )

            return dem_arr.astype(np.float32)

        except Exception as e:
            print(f"[StreamLoader] DEM fetch failed: {e}, using synthetic")
            return self._generate_synthetic_dem((min_x, min_y, max_x, max_y))

    def _generate_synthetic_dem(
        self, bounds_3857: Tuple[float, float, float, float]
    ) -> np.ndarray:
        """Generate synthetic elevation data for Dehradun-Mussoorie."""
        min_x, min_y, max_x, max_y = bounds_3857

        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        lon, lat = self.transformer_3857_to_4326.transform(center_x, center_y)

        # Rough altitude model
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
