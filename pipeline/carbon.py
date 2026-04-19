"""
pipeline/carbon.py
==================
Forest classification from DEM + carbon stock calculation via Chave 2014.

Flow:
  classify_forest(dem, zones)  →  forest_class (uint8)
  calculate_carbon(chm, forest_class, csv, fraction)  →  {agb, carbon_density}
"""

from __future__ import annotations
from typing import Dict, List
import numpy as np
import pandas as pd

CLASS_NAMES = {0: "Non-forest", 1: "Sal_Forest", 2: "Chir_Pine", 3: "Oak_Banj", 4: "High_Alpine"}


def classify_forest(dem: np.ndarray, zones: List[dict]) -> np.ndarray:
    """
    Classify forest type from DEM elevation using altitude zone config.

    Parameters
    ----------
    dem   : float32 (H, W) elevation in metres
    zones : list of dicts with keys: min_alt, max_alt, class_code

    Returns
    -------
    forest_class : uint8 (H, W)  0=non-forest, 1=Sal, 2=Pine, 3=Oak, 4=Alpine
    """
    classes = np.zeros_like(dem, dtype=np.uint8)
    for zone in zones:
        mask = (dem >= zone["min_alt"]) & (dem < zone["max_alt"])
        classes[mask] = zone["class_code"]
    return classes


def calculate_carbon(
    chm: np.ndarray,
    forest_class: np.ndarray,
    allometry_csv: str,
    carbon_fraction: float = 0.47,
) -> Dict[str, np.ndarray]:
    """
    Compute AGB (Mg/ha) and carbon density (MgC/ha) per pixel.

    Equations (Chave 2014):
      DBH = a * H^b
      AGB = 0.0673 * (rho * DBH^2 * H)^0.976
      Carbon = AGB * carbon_fraction

    Returns
    -------
    dict with keys 'agb' and 'carbon_density', both float32 (H, W)
    """
    params = pd.read_csv(allometry_csv)
    params = params.set_index("forest_type")

    agb = np.zeros_like(chm, dtype=np.float32)
    carbon = np.zeros_like(chm, dtype=np.float32)

    for code, name in CLASS_NAMES.items():
        if code == 0:
            continue
        mask = forest_class == code
        if not np.any(mask):
            continue
        if name not in params.index:
            continue

        a = params.loc[name, "a"]
        b = params.loc[name, "b"]
        rho = params.loc[name, "wood_density"]

        h = np.maximum(chm[mask], 0.0)
        dbh = a * np.power(h, b)
        agb_vals = 0.0673 * np.power(rho * np.square(dbh) * h, 0.976)
        agb_vals = np.where((h <= 0) | ~np.isfinite(agb_vals), 0.0, agb_vals)

        agb[mask] = agb_vals.astype(np.float32)
        carbon[mask] = (agb_vals * carbon_fraction).astype(np.float32)

    return {"agb": agb, "carbon_density": carbon}


def patch_stats(
    chm: np.ndarray,
    dem: np.ndarray,
    forest_class: np.ndarray,
    agb: np.ndarray,
    carbon: np.ndarray,
) -> dict:
    """Return summary statistics for a processed patch."""
    valid_chm = chm[chm > 0]
    valid_carbon = carbon[carbon > 0]

    zone_ha = {}
    for code, name in CLASS_NAMES.items():
        px = int(np.sum(forest_class == code))
        zone_ha[name] = round(px * 4 / 10000, 3)  # 2m res → 4 m²/px → ha

    return {
        "chm_mean_m": round(float(np.mean(valid_chm)), 2) if len(valid_chm) else 0.0,
        "chm_max_m": round(float(np.max(valid_chm)), 2) if len(valid_chm) else 0.0,
        "dem_mean_m": round(float(np.mean(dem)), 1),
        "agb_mean_mgha": round(float(np.mean(agb[agb > 0])), 2) if np.any(agb > 0) else 0.0,
        "carbon_mean_mgcha": round(float(np.mean(valid_carbon)), 2) if len(valid_carbon) else 0.0,
        "carbon_total_mgc": round(float(np.sum(carbon)) * 4 / 10000, 2),
        "forest_area_ha": zone_ha,
    }
