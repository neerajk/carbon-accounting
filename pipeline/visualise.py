"""pipeline/visualise.py — 6-panel carbon accounting visualization."""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize, BoundaryNorm, ListedColormap
import matplotlib.cm as cm
import rasterio

from .carbon import CLASS_NAMES

FOREST_COLORS = {
    0: "#888888",   # Non-forest / unknown
    1: "#1a7a1a",   # Sal Forest — dark green
    2: "#c8b400",   # Chir Pine — yellow
    3: "#8B4513",   # Oak Banj  — brown
    4: "#add8e6",   # High Alpine — light blue
}


def _colorbar(ax, cmap, vmin, vmax, label):
    sm = cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(label, fontsize=8)


def _apply_cmap(arr, cmap, vmin=None, vmax=None):
    vmin = vmin if vmin is not None else float(np.nanmin(arr))
    vmax = vmax if vmax is not None else float(np.nanmax(arr))
    norm = Normalize(vmin=vmin, vmax=vmax)
    rgba = cm.get_cmap(cmap)(norm(arr))
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def _forest_class_rgb(forest_class: np.ndarray) -> np.ndarray:
    h, w = forest_class.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for code, hex_color in FOREST_COLORS.items():
        r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
        mask = forest_class == code
        rgb[mask] = [r, g, b]
    return rgb


def visualize_patch(
    esri_rgb: np.ndarray,
    dem: np.ndarray,
    chm: np.ndarray,
    forest_class: np.ndarray,
    agb: np.ndarray,
    carbon_density: np.ndarray,
    patch_name: str,
    out_dir: Path,
    stats: Optional[dict] = None,
    cfg: dict = None,
):
    """
    Save a 6-panel figure:
      [ESRI RGB] [Canopy Height] [DEM]
      [Forest Class] [AGB] [Carbon Density]
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmap_chm = cfg["output"].get("colormap_chm", "viridis") if cfg else "viridis"
    cmap_carbon = cfg["output"].get("colormap_carbon", "hot_r") if cfg else "hot_r"

    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=130)
    fig.suptitle(f"Carbon Accounting — {patch_name}", fontsize=14, fontweight="bold")
    axes = axes.flatten()

    # Panel 1: ESRI RGB
    axes[0].imshow(esri_rgb)
    axes[0].set_title("ESRI Satellite (RGB)", fontsize=11)
    axes[0].axis("off")

    # Panel 2: Canopy Height
    chm_vmax = max(float(np.nanmax(chm)), 1.0)
    axes[1].imshow(_apply_cmap(chm, cmap_chm, 0, chm_vmax))
    axes[1].set_title("Canopy Height — CHMv2 (m)", fontsize=11)
    axes[1].axis("off")
    _colorbar(axes[1], cmap_chm, 0, chm_vmax, "Height (m)")

    # Panel 3: DEM
    dem_min, dem_max = float(np.nanmin(dem)), float(np.nanmax(dem))
    axes[2].imshow(_apply_cmap(dem, "terrain", dem_min, dem_max))
    axes[2].set_title("DEM — Elevation (m)", fontsize=11)
    axes[2].axis("off")
    _colorbar(axes[2], "terrain", dem_min, dem_max, "Elevation (m)")

    # Panel 4: Forest Classification
    axes[3].imshow(_forest_class_rgb(forest_class))
    axes[3].set_title("Forest Classification (DEM-based)", fontsize=11)
    axes[3].axis("off")
    legend_patches = [
        mpatches.Patch(color=FOREST_COLORS[code], label=name)
        for code, name in CLASS_NAMES.items()
        if np.any(forest_class == code)
    ]
    axes[3].legend(handles=legend_patches, loc="lower right", fontsize=7, framealpha=0.8)

    # Panel 5: AGB
    agb_vmax = max(float(np.nanmax(agb)), 1.0)
    axes[4].imshow(_apply_cmap(agb, "YlGn", 0, agb_vmax))
    axes[4].set_title("Above-Ground Biomass (Mg/ha)", fontsize=11)
    axes[4].axis("off")
    _colorbar(axes[4], "YlGn", 0, agb_vmax, "AGB (Mg/ha)")

    # Panel 6: Carbon Density
    c_vmax = max(float(np.nanmax(carbon_density)), 1.0)
    axes[5].imshow(_apply_cmap(carbon_density, cmap_carbon, 0, c_vmax))
    axes[5].set_title("Carbon Density (MgC/ha)", fontsize=11)
    axes[5].axis("off")
    _colorbar(axes[5], cmap_carbon, 0, c_vmax, "Carbon (MgC/ha)")

    # Stats annotation
    if stats:
        summary = (
            f"CHM mean: {stats['chm_mean_m']}m  max: {stats['chm_max_m']}m\n"
            f"DEM mean: {stats['dem_mean_m']}m\n"
            f"AGB mean: {stats['agb_mean_mgha']} Mg/ha\n"
            f"Carbon mean: {stats['carbon_mean_mgcha']} MgC/ha\n"
            f"Carbon total: {stats['carbon_total_mgc']} MgC"
        )
        fig.text(0.01, 0.01, summary, fontsize=8, va="bottom",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    out_path = out_dir / f"{patch_name}_carbon.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[vis] Saved → {out_path}")
    return out_path


def save_tif(array: np.ndarray, out_path: Path, nodata: float = -9999.0):
    """Save a float32 array as a single-band GeoTIFF (no CRS, pixel coords)."""
    out_path = Path(out_path)
    arr = np.where(np.isnan(array), nodata, array).astype(np.float32)
    with rasterio.open(
        out_path, "w", driver="GTiff",
        height=arr.shape[0], width=arr.shape[1],
        count=1, dtype=rasterio.float32,
        nodata=nodata,
    ) as dst:
        dst.write(arr, 1)
    print(f"[tif] Saved → {out_path}")
