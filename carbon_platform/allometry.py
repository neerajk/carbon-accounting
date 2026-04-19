"""
carbon_platform/allometry.py
============================
Chave 2014 allometric equation for Above Ground Biomass (AGB) calculation.
Includes regional DBH-height relationships.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd


class AllometryCalculator:
    """
    Calculate carbon stocks using Chave 2014 allometry.

    Equations:
    - DBH = a * H^b  (from regional params)
    - AGB = 0.0673 * (ρ * DBH^2 * H)^0.976  (Chave 2014)
    - Carbon = AGB * 0.47  (IPCC default carbon fraction)
    """

    def __init__(self, params_path: str):
        """
        Initialize with allometry parameters CSV.

        Parameters
        ----------
        params_path : str
            Path to CSV with columns: forest_type, a, b, wood_density
        """
        self.params_path = Path(params_path)
        self.params = self._load_params()

    def _load_params(self) -> pd.DataFrame:
        """Load regional allometry parameters."""
        if not self.params_path.exists():
            # Create default params if file doesn't exist
            default_params = pd.DataFrame({
                "forest_type": ["Sal_Forest", "Chir_Pine", "Oak_Banj"],
                "a": [0.396, 0.307, 0.235],
                "b": [1.089, 1.138, 1.246],
                "wood_density": [0.82, 0.49, 0.72],
            })
            default_params.to_csv(self.params_path, index=False)
            print(f"[Allometry] Created default params at {self.params_path}")
            return default_params

        return pd.read_csv(self.params_path)

    def get_params(self, forest_type: str) -> Dict[str, float]:
        """Get a, b, and wood density for a forest type."""
        row = self.params[self.params["forest_type"] == forest_type]
        if row.empty:
            raise ValueError(f"Unknown forest type: {forest_type}")
        return {
            "a": row["a"].values[0],
            "b": row["b"].values[0],
            "wood_density": row["wood_density"].values[0],
        }

    def calculate_dbh(self, height: np.ndarray, forest_type: str) -> np.ndarray:
        """
        Calculate Diameter at Breast Height (DBH) from canopy height.

        Parameters
        ----------
        height : np.ndarray
            Canopy height in meters (CHM output)
        forest_type : str
            Forest type name (Sal_Forest, Chir_Pine, Oak_Banj)

        Returns
        -------
        np.ndarray
            DBH in centimeters
        """
        params = self.get_params(forest_type)
        a = params["a"]
        b = params["b"]

        # DBH = a * H^b
        dbh = a * np.power(height, b)
        return dbh

    def calculate_agb(
        self,
        height: np.ndarray,
        forest_type: str,
        dbh: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Calculate Above Ground Biomass (AGB) using Chave 2014.

        AGB = 0.0673 * (ρ * DBH^2 * H)^0.976

        where:
        - ρ = wood density (g/cm³)
        - DBH = diameter at breast height (cm)
        - H = height (m)
        - AGB = Above Ground Biomass (Mg/ha)

        Parameters
        ----------
        height : np.ndarray
            Canopy height in meters
        forest_type : str
            Forest type for wood density lookup
        dbh : np.ndarray, optional
            Pre-calculated DBH in cm. If None, calculated from height.

        Returns
        -------
        np.ndarray
            AGB in Mg/ha (megagrams per hectare)
        """
        params = self.get_params(forest_type)
        rho = params["wood_density"]  # g/cm³

        if dbh is None:
            dbh = self.calculate_dbh(height, forest_type)

        # Chave 2014 equation
        # AGB = 0.0673 * (ρ * DBH² * H)^0.976
        agb = 0.0673 * np.power(rho * np.square(dbh) * height, 0.976)

        # Handle invalid values
        agb = np.where((height <= 0) | np.isnan(height), 0.0, agb)
        agb = np.where(np.isinf(agb), 0.0, agb)

        return agb

    def calculate_carbon_density(
        self,
        height: np.ndarray,
        forest_type: str,
        carbon_fraction: float = 0.47,
    ) -> np.ndarray:
        """
        Calculate carbon density from AGB.

        Parameters
        ----------
        height : np.ndarray
            Canopy height in meters
        forest_type : str
            Forest type name
        carbon_fraction : float
            Carbon fraction of biomass (default 0.47 from IPCC)

        Returns
        -------
        np.ndarray
            Carbon density in MgC/ha (megagrams of carbon per hectare)
        """
        agb = self.calculate_agb(height, forest_type)
        carbon = agb * carbon_fraction
        return carbon

    def process_array(
        self,
        height: np.ndarray,
        forest_class: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Process full arrays with mixed forest types.

        Parameters
        ----------
        height : np.ndarray
            Canopy height (CHM)
        forest_class : np.ndarray
            Integer class codes (1=Sal, 2=Pine, 3=Oak)

        Returns
        -------
        dict
            Dictionary with 'agb', 'carbon_density', 'dbh' arrays
        """
        forest_type_map = {1: "Sal_Forest", 2: "Chir_Pine", 3: "Oak_Banj"}

        agb = np.zeros_like(height, dtype=np.float32)
        carbon = np.zeros_like(height, dtype=np.float32)
        dbh = np.zeros_like(height, dtype=np.float32)

        for class_code, forest_type in forest_type_map.items():
            mask = forest_class == class_code
            if np.any(mask):
                h_subset = height[mask]
                dbh_subset = self.calculate_dbh(h_subset, forest_type)
                agb_subset = self.calculate_agb(h_subset, forest_type, dbh_subset)
                carbon_subset = agb_subset * 0.47

                dbh[mask] = dbh_subset
                agb[mask] = agb_subset
                carbon[mask] = carbon_subset

        return {
            "agb": agb,
            "carbon_density": carbon,
            "dbh": dbh,
        }
