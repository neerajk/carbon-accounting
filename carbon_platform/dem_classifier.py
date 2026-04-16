"""
carbon_platform/dem_classifier.py
=================================
Classify forest types based on Digital Elevation Model (DEM) altitude.
Uttarakhand Himalayan forest zonation.
"""

from __future__ import annotations
from typing import Dict, Tuple
import numpy as np


# Altitude-based forest classification for Uttarakhand
FOREST_ZONES = [
    {"name": "Unknown", "min_alt": -np.inf, "max_alt": 0, "class_code": 0, "wood_density": 0.0},
    {"name": "Sal_Forest", "min_alt": 0, "max_alt": 1000, "class_code": 1, "wood_density": 0.82},
    {"name": "Chir_Pine", "min_alt": 1000, "max_alt": 1800, "class_code": 2, "wood_density": 0.49},
    {"name": "Oak_Banj", "min_alt": 1800, "max_alt": 2800, "class_code": 3, "wood_density": 0.72},
    {"name": "High_Alpine", "min_alt": 2800, "max_alt": np.inf, "class_code": 4, "wood_density": 0.60},
]


class DEMClassifier:
    """
    Classify forest types from DEM elevation data.

    Uttarakhand forest zones:
    - 0-1000m: Sal Forest (Shorea robusta), ρ=0.82
    - 1000-1800m: Chir Pine (Pinus roxburghii), ρ=0.49
    - 1800-2800m: Oak/Banj (Quercus), ρ=0.72
    - 2800m+: High Alpine, ρ=0.60
    """

    def __init__(self):
        """Initialize classifier with Uttarakhand forest zones."""
        self.zones = FOREST_ZONES
        self.class_map = {z["class_code"]: z["name"] for z in self.zones}

    def classify(self, dem: np.ndarray) -> np.ndarray:
        """
        Classify forest type from DEM elevation.

        Parameters
        ----------
        dem : np.ndarray
            Elevation in meters above sea level

        Returns
        -------
        np.ndarray
            Integer class codes (0=Unknown, 1=Sal, 2=Pine, 3=Oak, 4=Alpine)
        """
        classes = np.zeros_like(dem, dtype=np.uint8)

        for zone in self.zones:
            mask = (dem >= zone["min_alt"]) & (dem < zone["max_alt"])
            classes[mask] = zone["class_code"]

        return classes

    def get_forest_name(self, class_code: int) -> str:
        """Get forest type name from class code."""
        return self.class_map.get(class_code, "Unknown")

    def get_wood_density(self, class_code: int) -> float:
        """Get wood density for class code."""
        for zone in self.zones:
            if zone["class_code"] == class_code:
                return zone["wood_density"]
        return 0.0

    def classify_with_info(self, dem: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Classify and return detailed information.

        Returns
        -------
        dict
            Contains 'class_codes', 'forest_names', 'wood_densities'
        """
        class_codes = self.classify(dem)

        # Create arrays for additional info
        wood_densities = np.zeros_like(dem, dtype=np.float32)
        for zone in self.zones:
            mask = class_codes == zone["class_code"]
            wood_densities[mask] = zone["wood_density"]

        return {
            "class_codes": class_codes,
            "wood_densities": wood_densities,
        }

    def get_zone_statistics(self, dem: np.ndarray) -> Dict[str, Dict]:
        """
        Calculate statistics for each elevation zone.

        Parameters
        ----------
        dem : np.ndarray
            Elevation data

        Returns
        -------
        dict
            Statistics per zone: pixel count, area (ha), elevation range
        """
        classes = self.classify(dem)
        stats = {}

        for zone in self.zones:
            code = zone["class_code"]
            mask = classes == code
            count = np.sum(mask)

            if count > 0:
                # Assume 2m resolution = 4 m² per pixel
                area_ha = count * 4 / 10000
                elevation_min = np.min(dem[mask]) if count > 0 else 0
                elevation_max = np.max(dem[mask]) if count > 0 else 0
            else:
                area_ha = 0
                elevation_min = 0
                elevation_max = 0

            stats[zone["name"]] = {
                "class_code": code,
                "pixel_count": int(count),
                "area_ha": float(area_ha),
                "elevation_min_m": float(elevation_min),
                "elevation_max_m": float(elevation_max),
            }

        return stats
