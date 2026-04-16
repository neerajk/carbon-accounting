"""
carbon_platform/visualizer.py
==============================
Comprehensive visualization suite for carbon accounting.
Includes 4-panel diagnostics, DEM analysis, forest classification, and data charts.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize, ListedColormap, BoundaryNorm
import xarray as xr


# Forest class colormap
FOREST_COLORS = {
    0: [0.5, 0.5, 0.5],    # Unknown - gray
    1: [0.2, 0.6, 0.2],    # Sal Forest - green
    2: [0.4, 0.3, 0.2],    # Chir Pine - brown
    3: [0.0, 0.4, 0.2],    # Oak/Banj - dark green
    4: [0.8, 0.8, 0.9],    # Alpine - light gray
}

FOREST_NAMES = {
    0: "Unknown",
    1: "Sal Forest",
    2: "Chir Pine",
    3: "Oak/Banj",
    4: "High Alpine",
}


class CarbonVisualizer:
    """
    Comprehensive visualization suite for carbon accounting DataCube.

    Includes:
    - 4-panel diagnostic visualizations
    - DEM elevation maps
    - Forest classification maps
    - Data analysis charts
    """

    def __init__(self, output_dir: str = "outputs/visualizations"):
        """
        Initialize visualizer.

        Parameters
        ----------
        output_dir : str
            Directory for output PNGs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def visualize_single_patch(
        self,
        rgb: np.ndarray,
        dem: np.ndarray,
        chm: np.ndarray,
        forest_class: np.ndarray,
        carbon_density: np.ndarray,
        patch_idx: int = 0,
        save_path: Optional[str] = None,
    ) -> Path:
        """
        Create comprehensive 6-panel visualization for a single patch.

        Panels:
        1. RGB imagery
        2. DEM elevation
        3. Forest classification (DEM-based)
        4. Canopy Height (CHM)
        5. Carbon density
        6. Histogram analysis

        Parameters
        ----------
        rgb : np.ndarray
            RGB image (H, W, 3)
        dem : np.ndarray
            DEM elevation (H, W)
        chm : np.ndarray
            Canopy height (H, W)
        forest_class : np.ndarray
            Forest classification (H, W)
        carbon_density : np.ndarray
            Carbon density (H, W)
        patch_idx : int
            Patch index for filename
        save_path : str, optional
            Custom save path

        Returns
        -------
        Path
            Path to saved PNG
        """
        fig = plt.figure(figsize=(20, 12), dpi=150)
        gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)

        # Panel 1: RGB
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(np.clip(rgb, 0, 255).astype(np.uint8))
        ax1.set_title(f"Patch {patch_idx:04d}: RGB Imagery", fontsize=11, fontweight="bold")
        ax1.axis("off")

        # Panel 2: DEM Elevation
        ax2 = fig.add_subplot(gs[0, 1])
        dem_vmin, dem_vmax = dem.min(), dem.max()
        im2 = ax2.imshow(dem, cmap="terrain", vmin=dem_vmin, vmax=dem_vmax)
        ax2.set_title("DEM Elevation (m)", fontsize=11, fontweight="bold")
        ax2.axis("off")
        cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label("Elevation (m)", fontsize=9)

        # Add altitude zone lines
        self._add_altitude_lines(ax2, dem)

        # Panel 3: Forest Classification
        ax3 = fig.add_subplot(gs[0, 2])
        forest_rgb = self._forest_class_to_rgb(forest_class)
        ax3.imshow(forest_rgb)
        ax3.set_title("Forest Type (DEM-based)", fontsize=11, fontweight="bold")
        ax3.axis("off")
        self._add_forest_legend(ax3, loc="upper left")

        # Panel 4: Canopy Height
        ax4 = fig.add_subplot(gs[0, 3])
        chm_valid = chm[chm > 0]
        chm_vmax = np.percentile(chm_valid, 99) if len(chm_valid) > 0 else 40
        im4 = ax4.imshow(chm, cmap="YlGn", vmin=0, vmax=chm_vmax)
        ax4.set_title("Canopy Height (m)", fontsize=11, fontweight="bold")
        ax4.axis("off")
        cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
        cbar4.set_label("Height (m)", fontsize=9)

        # Panel 5: Carbon Density
        ax5 = fig.add_subplot(gs[1, 0])
        carbon_valid = carbon_density[carbon_density > 0]
        carbon_vmax = np.percentile(carbon_valid, 99) if len(carbon_valid) > 0 else 100
        im5 = ax5.imshow(carbon_density, cmap="YlOrRd", vmin=0, vmax=carbon_vmax)
        ax5.set_title("Carbon Density (MgC/ha)", fontsize=11, fontweight="bold")
        ax5.axis("off")
        cbar5 = plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
        cbar5.set_label("MgC/ha", fontsize=9)

        # Panel 6: Combined overlay
        ax6 = fig.add_subplot(gs[1, 1])
        self._plot_combined_overlay(ax6, rgb, forest_class, carbon_density)
        ax6.set_title("Carbon over Forest Types", fontsize=11, fontweight="bold")
        ax6.axis("off")

        # Panel 7: DEM vs CHM scatter
        ax7 = fig.add_subplot(gs[1, 2])
        self._plot_elevation_height_scatter(ax7, dem, chm, forest_class)

        # Panel 8: Statistics table
        ax8 = fig.add_subplot(gs[1, 3])
        self._plot_statistics_table(ax8, dem, chm, forest_class, carbon_density)

        if save_path is None:
            save_path = self.output_dir / f"patch_{patch_idx:04d}_full_analysis.png"
        else:
            save_path = Path(save_path)

        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"[Visualizer] Full patch analysis saved: {save_path}")
        return save_path

    def create_dem_analysis(
        self,
        dem: np.ndarray,
        forest_class: np.ndarray,
        altitude_zones: List[Dict],
        save_path: Optional[str] = None,
    ) -> Path:
        """
        Create comprehensive DEM analysis visualization.

        Parameters
        ----------
        dem : np.ndarray
            DEM elevation data
        forest_class : np.ndarray
            Forest classification
        altitude_zones : list
            Zone definitions from config
        save_path : str, optional
            Custom save path

        Returns
        -------
        Path
            Path to saved PNG
        """
        fig = plt.figure(figsize=(18, 10), dpi=150)
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # Panel 1: DEM with hillshade effect
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_hillshade_dem(ax1, dem)
        ax1.set_title("DEM with Hillshade", fontsize=12, fontweight="bold")

        # Panel 2: Slope
        ax2 = fig.add_subplot(gs[0, 1])
        slope = self._calculate_slope(dem)
        im2 = ax2.imshow(slope, cmap="RdYlGn_r", vmin=0, vmax=45)
        ax2.set_title("Slope (degrees)", fontsize=12, fontweight="bold")
        ax2.axis("off")
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # Panel 3: Aspect
        ax3 = fig.add_subplot(gs[0, 2])
        aspect = self._calculate_aspect(dem)
        im3 = ax3.imshow(aspect, cmap="hsv", vmin=0, vmax=360)
        ax3.set_title("Aspect (degrees)", fontsize=12, fontweight="bold")
        ax3.axis("off")
        cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        cbar3.set_label("Degrees from North", fontsize=9)

        # Panel 4: Elevation histogram by forest type
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_elevation_histogram(ax4, dem, forest_class)

        # Panel 5: Zone distribution pie chart
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_zone_distribution(ax5, forest_class)

        # Panel 6: Altitude profile (if applicable)
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_altitude_profile(ax6, dem, altitude_zones)

        if save_path is None:
            save_path = self.output_dir / "dem_analysis.png"
        else:
            save_path = Path(save_path)

        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"[Visualizer] DEM analysis saved: {save_path}")
        return save_path

    def create_forest_classification_map(
        self,
        forest_class: np.ndarray,
        dem: np.ndarray,
        altitude_zones: List[Dict],
        save_path: Optional[str] = None,
    ) -> Path:
        """
        Create detailed forest classification map with altitude zones.

        Parameters
        ----------
        forest_class : np.ndarray
            Forest classification array
        dem : np.ndarray
            DEM for altitude reference
        altitude_zones : list
            Zone definitions
        save_path : str, optional
            Custom save path

        Returns
        -------
        Path
            Path to saved PNG
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=150)

        # Left: Forest classification with DEM contours
        ax1 = axes[0]
        forest_rgb = self._forest_class_to_rgb(forest_class)
        ax1.imshow(forest_rgb)

        # Add DEM contours
        levels = [z["min_alt"] for z in altitude_zones[1:]]  # Skip unknown
        cs = ax1.contour(dem, levels=levels, colors="white", alpha=0.5, linewidths=0.8)
        ax1.clabel(cs, inline=True, fontsize=8, fmt="%dm")

        ax1.set_title("Forest Classification with Altitude Zones", fontsize=12, fontweight="bold")
        ax1.axis("off")
        self._add_forest_legend(ax1, loc="lower left")

        # Right: Altitude zone bands
        ax2 = axes[1]
        self._plot_altitude_bands(ax2, dem, altitude_zones)
        ax2.set_title("Altitude Zone Classification", fontsize=12, fontweight="bold")
        ax2.axis("off")

        if save_path is None:
            save_path = self.output_dir / "forest_classification_map.png"
        else:
            save_path = Path(save_path)

        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"[Visualizer] Forest classification map saved: {save_path}")
        return save_path

    def create_data_analysis_charts(
        self,
        ds: xr.Dataset,
        altitude_zones: List[Dict],
        save_path: Optional[str] = None,
    ) -> Path:
        """
        Create comprehensive data analysis charts.

        Parameters
        ----------
        ds : xr.Dataset
            DataCube dataset
        altitude_zones : list
            Zone definitions
        save_path : str, optional
            Custom save path

        Returns
        -------
        Path
            Path to saved PNG
        """
        fig = plt.figure(figsize=(20, 14), dpi=150)
        gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)

        # Extract data
        dem = ds.dem.values
        chm = ds.chm.values
        forest_class = ds.forest_class.values
        carbon = ds.carbon_density.values

        # Mask invalid values
        valid_mask = (chm > 0) & (carbon > 0) & (dem > 0)

        # Chart 1: Carbon by Forest Type (Box plot)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_carbon_by_forest_type(ax1, carbon, forest_class)

        # Chart 2: Height Distribution by Forest Type
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_height_distribution(ax2, chm, forest_class)

        # Chart 3: Carbon vs Elevation Scatter
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_carbon_elevation_scatter(ax3, carbon, dem, forest_class)

        # Chart 4: Forest Area Statistics
        ax4 = fig.add_subplot(gs[0, 3])
        self._plot_forest_area_stats(ax4, forest_class, altitude_zones)

        # Chart 5: Carbon Stock by Zone (Bar chart)
        ax5 = fig.add_subplot(gs[1, 0])
        self._plot_carbon_stock_by_zone(ax5, carbon, forest_class)

        # Chart 6: Canopy Height vs Carbon Density
        ax6 = fig.add_subplot(gs[1, 1])
        self._plot_height_carbon_correlation(ax6, chm, carbon)

        # Chart 7: DEM Histogram
        ax7 = fig.add_subplot(gs[1, 2])
        self._plot_dem_histogram(ax7, dem)

        # Chart 8: Summary Statistics Table
        ax8 = fig.add_subplot(gs[1, 3])
        self._plot_summary_stats_table(ax8, ds, altitude_zones)

        # Chart 9: Time series placeholder (if multiple timestamps)
        ax9 = fig.add_subplot(gs[2, :2])
        self._plot_spatial_distribution(ax9, carbon, forest_class)

        # Chart 10: Insights text
        ax10 = fig.add_subplot(gs[2, 2:])
        self._plot_insights_text(ax10, ds, altitude_zones)

        if save_path is None:
            save_path = self.output_dir / "data_analysis_charts.png"
        else:
            save_path = Path(save_path)

        fig.suptitle("Carbon Accounting Data Analysis - Uttarakhand", fontsize=16, fontweight="bold", y=0.98)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"[Visualizer] Data analysis charts saved: {save_path}")
        return save_path

    # Helper methods
    def _forest_class_to_rgb(self, forest_class: np.ndarray) -> np.ndarray:
        """Convert forest class codes to RGB image."""
        rgb = np.zeros((*forest_class.shape, 3), dtype=np.float32)
        for code, color in FOREST_COLORS.items():
            mask = forest_class == code
            rgb[mask] = color
        return (rgb * 255).astype(np.uint8)

    def _add_forest_legend(self, ax: plt.Axes, loc: str = "upper left"):
        """Add forest type legend."""
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=FOREST_COLORS[1], label="Sal Forest (<1000m)"),
            Patch(facecolor=FOREST_COLORS[2], label="Chir Pine (1000-1800m)"),
            Patch(facecolor=FOREST_COLORS[3], label="Oak/Banj (1800-2800m)"),
            Patch(facecolor=FOREST_COLORS[4], label="High Alpine (>2800m)"),
        ]
        ax.legend(handles=legend_elements, loc=loc, fontsize=8, framealpha=0.9)

    def _add_altitude_lines(self, ax: plt.Axes, dem: np.ndarray):
        """Add altitude zone boundary lines."""
        levels = [1000, 1800, 2800]
        cs = ax.contour(dem, levels=levels, colors="black", alpha=0.4, linewidths=0.5)

    def _calculate_slope(self, dem: np.ndarray) -> np.ndarray:
        """Calculate slope from DEM."""
        from numpy import gradient
        gy, gx = gradient(dem)
        slope = np.degrees(np.arctan(np.sqrt(gx**2 + gy**2)))
        return slope

    def _calculate_aspect(self, dem: np.ndarray) -> np.ndarray:
        """Calculate aspect from DEM."""
        from numpy import gradient
        gy, gx = gradient(dem)
        aspect = np.degrees(np.arctan2(-gy, gx))
        aspect = np.where(aspect < 0, 90.0 - aspect, 360.0 - aspect + 90.0)
        return aspect

    def _plot_hillshade_dem(self, ax: plt.Axes, dem: np.ndarray):
        """Plot DEM with hillshade effect."""
        from matplotlib.colors import LightSource
        ls = LightSource(azdeg=315, altdeg=45)
        hillshade = ls.hillshade(dem, vert_exag=0.5)
        ax.imshow(hillshade, cmap="gray", alpha=0.8)
        im = ax.imshow(dem, cmap="terrain", alpha=0.6)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Elevation (m)")

    def _plot_elevation_histogram(self, ax: plt.Axes, dem: np.ndarray, forest_class: np.ndarray):
        """Plot elevation histogram by forest type."""
        colors = [FOREST_COLORS[i] for i in range(1, 5)]
        labels = [FOREST_NAMES[i] for i in range(1, 5)]

        for code in range(1, 5):
            mask = forest_class == code
            if np.any(mask):
                elevations = dem[mask]
                ax.hist(elevations.flatten(), bins=30, alpha=0.6,
                       color=FOREST_COLORS[code], label=FOREST_NAMES[code])

        ax.set_xlabel("Elevation (m)", fontsize=10)
        ax.set_ylabel("Pixel Count", fontsize=10)
        ax.set_title("Elevation Distribution by Forest Type", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_zone_distribution(self, ax: plt.Axes, forest_class: np.ndarray):
        """Plot forest zone distribution as pie chart."""
        counts = []
        labels = []
        colors = []

        for code in range(1, 5):
            count = np.sum(forest_class == code)
            if count > 0:
                counts.append(count)
                labels.append(FOREST_NAMES[code])
                colors.append(FOREST_COLORS[code])

        if counts:
            ax.pie(counts, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
            ax.set_title("Forest Type Distribution", fontsize=11, fontweight="bold")

    def _plot_altitude_profile(self, ax: plt.Axes, dem: np.ndarray, altitude_zones: List[Dict]):
        """Plot altitude profile (mean elevation across rows)."""
        profile = np.mean(dem, axis=1)
        ax.plot(profile, range(len(profile)), "b-", linewidth=1.5)
        ax.set_xlabel("Mean Elevation (m)", fontsize=10)
        ax.set_ylabel("Row (North-South)", fontsize=10)
        ax.set_title("North-South Altitude Profile", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Add zone lines
        for zone in altitude_zones[1:]:  # Skip unknown
            ax.axvline(x=zone["min_alt"], color="red", linestyle="--", alpha=0.5)
            ax.axvline(x=zone["max_alt"], color="red", linestyle="--", alpha=0.5)

    def _plot_combined_overlay(self, ax: plt.Axes, rgb: np.ndarray,
                               forest_class: np.ndarray, carbon: np.ndarray):
        """Plot carbon density overlaid on forest types."""
        # Base: forest class
        base = self._forest_class_to_rgb(forest_class)

        # Overlay: carbon as heatmap
        carbon_norm = np.clip(carbon / (np.percentile(carbon[carbon > 0], 99) + 1e-8), 0, 1)
        heatmap = plt.cm.YlOrRd(carbon_norm)[:, :, :3]

        # Blend
        blended = (0.6 * base / 255.0 + 0.4 * heatmap)
        ax.imshow(np.clip(blended, 0, 1))

    def _plot_elevation_height_scatter(self, ax: plt.Axes, dem: np.ndarray,
                                       chm: np.ndarray, forest_class: np.ndarray):
        """Plot elevation vs canopy height scatter."""
        valid = (dem > 0) & (chm > 0)
        if np.any(valid):
            for code in range(1, 5):
                mask = valid & (forest_class == code)
                if np.any(mask):
                    ax.scatter(dem[mask], chm[mask], alpha=0.3, s=5,
                             color=FOREST_COLORS[code], label=FOREST_NAMES[code])

        ax.set_xlabel("Elevation (m)", fontsize=10)
        ax.set_ylabel("Canopy Height (m)", fontsize=10)
        ax.set_title("Elevation vs Canopy Height", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_statistics_table(self, ax: plt.Axes, dem: np.ndarray, chm: np.ndarray,
                              forest_class: np.ndarray, carbon: np.ndarray):
        """Plot statistics table."""
        stats_text = []
        stats_text.append(f"{'Metric':<25} {'Value':>15}")
        stats_text.append("-" * 45)

        # Overall stats
        valid_dem = dem[dem > 0]
        valid_chm = chm[chm > 0]
        valid_carbon = carbon[carbon > 0]

        if len(valid_dem) > 0:
            stats_text.append(f"{'Mean Elevation':<25} {np.mean(valid_dem):>12.1f} m")
            stats_text.append(f"{'Elevation Range':<25} {np.min(valid_dem):>6.0f}-{np.max(valid_dem):>6.0f} m")

        if len(valid_chm) > 0:
            stats_text.append(f"{'Mean Canopy Height':<25} {np.mean(valid_chm):>12.1f} m")
            stats_text.append(f"{'Max Canopy Height':<25} {np.max(valid_chm):>12.1f} m")

        if len(valid_carbon) > 0:
            stats_text.append(f"{'Mean Carbon Density':<25} {np.mean(valid_carbon):>12.1f} MgC/ha")
            stats_text.append(f"{'Total Carbon Stock':<25} {np.sum(valid_carbon) * 4 / 10000:>12.1f} MgC")

        # Per forest type
        stats_text.append("")
        stats_text.append("By Forest Type:")
        for code in range(1, 5):
            mask = forest_class == code
            if np.any(mask):
                count = np.sum(mask)
                pct = count / forest_class.size * 100
                stats_text.append(f"  {FOREST_NAMES[code]:<20} {count:>8} px ({pct:>5.1f}%)")

        ax.text(0.05, 0.95, "\n".join(stats_text), transform=ax.transAxes,
               fontsize=9, verticalalignment="top", fontfamily="monospace",
               bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        ax.axis("off")
        ax.set_title("Patch Statistics", fontsize=11, fontweight="bold")

    def _plot_altitude_bands(self, ax: plt.Axes, dem: np.ndarray, altitude_zones: List[Dict]):
        """Plot altitude bands with discrete colors."""
        bands = np.zeros((*dem.shape, 3))
        for zone in altitude_zones:
            if zone["class_code"] == 0:
                continue
            mask = (dem >= zone["min_alt"]) & (dem < zone["max_alt"])
            color = FOREST_COLORS.get(zone["class_code"], [0.5, 0.5, 0.5])
            bands[mask] = color

        ax.imshow(bands)
        ax.set_title("Altitude Zone Bands", fontsize=11, fontweight="bold")

    def _plot_carbon_by_forest_type(self, ax: plt.Axes, carbon: np.ndarray, forest_class: np.ndarray):
        """Box plot of carbon by forest type."""
        data = []
        labels = []
        colors = []

        for code in range(1, 5):
            mask = (forest_class == code) & (carbon > 0)
            if np.any(mask):
                data.append(carbon[mask].flatten())
                labels.append(FOREST_NAMES[code])
                colors.append(FOREST_COLORS[code])

        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

        ax.set_ylabel("Carbon Density (MgC/ha)", fontsize=10)
        ax.set_title("Carbon by Forest Type", fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", rotation=45)

    def _plot_height_distribution(self, ax: plt.Axes, chm: np.ndarray, forest_class: np.ndarray):
        """Plot canopy height distribution."""
        for code in range(1, 5):
            mask = (forest_class == code) & (chm > 0)
            if np.any(mask):
                ax.hist(chm[mask].flatten(), bins=30, alpha=0.6,
                       color=FOREST_COLORS[code], label=FOREST_NAMES[code])

        ax.set_xlabel("Canopy Height (m)", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.set_title("Height Distribution", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)

    def _plot_carbon_elevation_scatter(self, ax: plt.Axes, carbon: np.ndarray,
                                       dem: np.ndarray, forest_class: np.ndarray):
        """Scatter plot of carbon vs elevation."""
        valid = (carbon > 0) & (dem > 0)
        if np.any(valid):
            for code in range(1, 5):
                mask = valid & (forest_class == code)
                if np.any(mask):
                    ax.scatter(dem[mask], carbon[mask], alpha=0.3, s=5,
                             color=FOREST_COLORS[code], label=FOREST_NAMES[code])

        ax.set_xlabel("Elevation (m)", fontsize=10)
        ax.set_ylabel("Carbon Density (MgC/ha)", fontsize=10)
        ax.set_title("Carbon vs Elevation", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)

    def _plot_forest_area_stats(self, ax: plt.Axes, forest_class: np.ndarray,
                               altitude_zones: List[Dict]):
        """Plot forest area statistics."""
        areas = []
        labels = []
        colors = []

        pixel_area_ha = 4 / 10000  # 2m resolution = 4 m2 = 0.0004 ha

        for code in range(1, 5):
            count = np.sum(forest_class == code)
            if count > 0:
                areas.append(count * pixel_area_ha)
                labels.append(FOREST_NAMES[code])
                colors.append(FOREST_COLORS[code])

        if areas:
            bars = ax.bar(range(len(areas)), areas, color=colors, alpha=0.7)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_ylabel("Area (ha)", fontsize=10)
            ax.set_title("Forest Area by Type", fontsize=11, fontweight="bold")

            # Add value labels
            for bar, val in zip(bars, areas):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    def _plot_carbon_stock_by_zone(self, ax: plt.Axes, carbon: np.ndarray, forest_class: np.ndarray):
        """Plot total carbon stock by zone."""
        stocks = []
        labels = []
        colors = []

        pixel_area_ha = 4 / 10000

        for code in range(1, 5):
            mask = (forest_class == code) & (carbon > 0)
            if np.any(mask):
                total_carbon = np.sum(carbon[mask]) * pixel_area_ha
                stocks.append(total_carbon)
                labels.append(FOREST_NAMES[code])
                colors.append(FOREST_COLORS[code])

        if stocks:
            bars = ax.bar(range(len(stocks)), stocks, color=colors, alpha=0.7)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_ylabel("Total Carbon (MgC)", fontsize=10)
            ax.set_title("Carbon Stock by Zone", fontsize=11, fontweight="bold")

    def _plot_height_carbon_correlation(self, ax: plt.Axes, chm: np.ndarray, carbon: np.ndarray):
        """Plot canopy height vs carbon correlation."""
        valid = (chm > 0) & (carbon > 0)
        if np.any(valid):
            ax.scatter(chm[valid], carbon[valid], alpha=0.3, s=5)

            # Add trend line
            z = np.polyfit(chm[valid].flatten(), carbon[valid].flatten(), 1)
            p = np.poly1d(z)
            x_line = np.linspace(chm[valid].min(), chm[valid].max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, label="Trend")

            # Calculate R²
            y_pred = p(chm[valid].flatten())
            ss_res = np.sum((carbon[valid].flatten() - y_pred) ** 2)
            ss_tot = np.sum((carbon[valid].flatten() - np.mean(carbon[valid])) ** 2)
            r2 = 1 - (ss_res / ss_tot)

            ax.text(0.05, 0.95, f"R² = {r2:.3f}", transform=ax.transAxes,
                   fontsize=10, verticalalignment="top",
                   bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        ax.set_xlabel("Canopy Height (m)", fontsize=10)
        ax.set_ylabel("Carbon Density (MgC/ha)", fontsize=10)
        ax.set_title("Height-Carbon Correlation", fontsize=11, fontweight="bold")

    def _plot_dem_histogram(self, ax: plt.Axes, dem: np.ndarray):
        """Plot DEM elevation histogram."""
        valid_dem = dem[dem > 0]
        if len(valid_dem) > 0:
            ax.hist(valid_dem.flatten(), bins=50, color="steelblue", alpha=0.7, edgecolor="black")
            ax.axvline(np.mean(valid_dem), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(valid_dem):.0f}m")
            ax.axvline(np.median(valid_dem), color="green", linestyle="--", linewidth=2, label=f"Median: {np.median(valid_dem):.0f}m")

        ax.set_xlabel("Elevation (m)", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.set_title("Elevation Histogram", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)

    def _plot_summary_stats_table(self, ax: plt.Axes, ds: xr.Dataset, altitude_zones: List[Dict]):
        """Plot comprehensive summary statistics table."""
        stats_text = []
        stats_text.append("REGIONAL SUMMARY STATISTICS")
        stats_text.append("=" * 50)

        dem = ds.dem.values
        chm = ds.chm.values
        carbon = ds.carbon_density.values
        forest_class = ds.forest_class.values

        valid_dem = dem[dem > 0]
        valid_chm = chm[chm > 0]
        valid_carbon = carbon[carbon > 0]

        stats_text.append(f"Total Area: {forest_class.size * 4 / 10000:.1f} ha")
        stats_text.append(f"Valid Pixels: {np.sum(forest_class > 0) / forest_class.size * 100:.1f}%")
        stats_text.append("")

        if len(valid_dem) > 0:
            stats_text.append(f"Elevation: {np.min(valid_dem):.0f} - {np.max(valid_dem):.0f} m (mean: {np.mean(valid_dem):.0f}m)")

        if len(valid_chm) > 0:
            stats_text.append(f"Canopy Height: {np.min(valid_chm):.1f} - {np.max(valid_chm):.1f} m (mean: {np.mean(valid_chm):.1f}m)")

        if len(valid_carbon) > 0:
            stats_text.append(f"Carbon Density: {np.min(valid_carbon):.1f} - {np.max(valid_carbon):.1f} MgC/ha")
            stats_text.append(f"Mean Carbon: {np.mean(valid_carbon):.1f} MgC/ha")
            total_carbon = np.sum(valid_carbon) * 4 / 10000
            stats_text.append(f"Total Carbon Stock: {total_carbon:.1f} MgC")

        ax.text(0.05, 0.95, "\n".join(stats_text), transform=ax.transAxes,
               fontsize=9, verticalalignment="top", fontfamily="monospace",
               bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7))
        ax.axis("off")
        ax.set_title("Summary Statistics", fontsize=11, fontweight="bold")

    def _plot_spatial_distribution(self, ax: plt.Axes, carbon: np.ndarray, forest_class: np.ndarray):
        """Plot spatial distribution of carbon."""
        im = ax.imshow(carbon, cmap="YlOrRd", vmin=0, vmax=np.percentile(carbon[carbon > 0], 99) if np.any(carbon > 0) else 100)
        ax.set_title("Spatial Carbon Distribution", fontsize=12, fontweight="bold")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="MgC/ha")

    def _plot_insights_text(self, ax: plt.Axes, ds: xr.Dataset, altitude_zones: List[Dict]):
        """Generate and plot insights text."""
        insights = self._generate_insights(ds, altitude_zones)
        ax.text(0.05, 0.95, insights, transform=ax.transAxes,
               fontsize=10, verticalalignment="top", fontfamily="sans-serif",
               bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
        ax.axis("off")
        ax.set_title("Key Insights & Findings", fontsize=12, fontweight="bold")

    def _generate_insights(self, ds: xr.Dataset, altitude_zones: List[Dict]) -> str:
        """Generate insights from data analysis."""
        insights = []
        insights.append("KEY INSIGHTS")
        insights.append("=" * 40)

        dem = ds.dem.values
        chm = ds.chm.values
        carbon = ds.carbon_density.values
        forest_class = ds.forest_class.values

        valid_mask = (chm > 0) & (carbon > 0) & (forest_class > 0)

        if np.any(valid_mask):
            # Find dominant forest type
            forest_counts = []
            for code in range(1, 5):
                count = np.sum(forest_class == code)
                forest_counts.append((count, FOREST_NAMES[code]))
            forest_counts.sort(reverse=True)
            dominant = forest_counts[0][1]
            insights.append(f"1. Dominant Forest Type: {dominant}")

            # Carbon hotspots
            high_carbon = carbon > np.percentile(carbon[carbon > 0], 90)
            hotspot_pct = np.sum(high_carbon) / carbon.size * 100
            insights.append(f"2. Carbon Hotspots: {hotspot_pct:.1f}% of area")

            # Elevation-carbon relationship
            mean_carbon_by_zone = []
            for code in range(1, 5):
                mask = (forest_class == code) & (carbon > 0)
                if np.any(mask):
                    mean_carbon_by_zone.append((np.mean(carbon[mask]), FOREST_NAMES[code]))

            if mean_carbon_by_zone:
                mean_carbon_by_zone.sort(reverse=True)
                insights.append(f"3. Highest Carbon: {mean_carbon_by_zone[0][1]}")

            # Height insights
            mean_height = np.mean(chm[valid_mask])
            insights.append(f"4. Mean Canopy Height: {mean_height:.1f}m")

            # Total carbon
            total_carbon = np.sum(carbon[carbon > 0]) * 4 / 10000
            insights.append(f"5. Total Carbon Stock: {total_carbon:.1f} MgC")

        return "\n".join(insights)

    # Legacy methods for backwards compatibility
    def create_4panel(
        self,
        ds: xr.Dataset,
        slice_y: Optional[slice] = None,
        slice_x: Optional[slice] = None,
        title: str = "Carbon Accounting Analysis",
    ) -> plt.Figure:
        """Create 4-panel figure (legacy method)."""
        if slice_y is None:
            slice_y = slice(0, min(1024, len(ds.y)))
        if slice_x is None:
            slice_x = slice(0, min(1024, len(ds.x)))

        rgb = np.stack([
            ds.red[slice_y, slice_x].values,
            ds.green[slice_y, slice_x].values,
            ds.blue[slice_y, slice_x].values,
        ], axis=-1)

        chm = ds.chm[slice_y, slice_x].values
        forest_class = ds.forest_class[slice_y, slice_x].values
        carbon = ds.carbon_density[slice_y, slice_x].values

        dem = ds.dem[slice_y, slice_x].values if "dem" in ds else None

        return self.visualize_single_patch(rgb, dem, chm, forest_class, carbon,
                                          save_path=str(self.output_dir / "4panel_legacy.png"))

    def export_slice(
        self,
        ds: xr.Dataset,
        y_start: int,
        y_end: int,
        x_start: int,
        x_end: int,
        filename: str,
    ) -> Path:
        """Export a specific slice (legacy method)."""
        slice_y = slice(y_start, y_end)
        slice_x = slice(x_start, x_end)

        rgb = np.stack([
            ds.red[slice_y, slice_x].values,
            ds.green[slice_y, slice_x].values,
            ds.blue[slice_y, slice_x].values,
        ], axis=-1)

        chm = ds.chm[slice_y, slice_x].values
        forest_class = ds.forest_class[slice_y, slice_x].values
        carbon = ds.carbon_density[slice_y, slice_x].values
        dem = ds.dem[slice_y, slice_x].values

        return self.visualize_single_patch(rgb, dem, chm, forest_class, carbon,
                                          save_path=str(self.output_dir / filename))

    def export_summary(self, ds: xr.Dataset, stats: Dict) -> Path:
        """Export summary visualization (legacy method)."""
        return self.create_data_analysis_charts(ds, stats)
