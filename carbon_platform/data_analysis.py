"""
carbon_platform/data_analysis.py
=================================
Data analysis module for carbon accounting insights.
Generates statistics, trends, and analytical reports.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats


class CarbonDataAnalyzer:
    """
    Analyze carbon accounting data for insights and trends.
    """

    def __init__(self, data_dict):
        """
        Initialize analyzer with carbon accounting data.

        Parameters
        ----------
        data_dict : dict
            Dictionary with numpy arrays: 'dem', 'chm', 'carbon_density', 'forest_class', 'agb'
        """
        self.dem = data_dict['dem']
        self.chm = data_dict['chm']
        self.carbon_density = data_dict['carbon_density']
        self.forest_class = data_dict['forest_class']
        self.agb = data_dict.get('agb')
        self.results = {}

    def run_full_analysis(self, altitude_zones: List[Dict]) -> Dict:
        """
        Run comprehensive data analysis.

        Parameters
        ----------
        altitude_zones : list
            Zone definitions from config

        Returns
        -------
        dict
            Complete analysis results
        """
        print("[DataAnalysis] Running full analysis...")

        self.results["overview"] = self._analyze_overview()
        self.results["by_forest_type"] = self._analyze_by_forest_type(altitude_zones)
        self.results["elevation_analysis"] = self._analyze_elevation_patterns()
        self.results["carbon_hotspots"] = self._identify_carbon_hotspots()
        self.results["correlations"] = self._analyze_correlations()
        self.results["carbon_stock"] = self._calculate_carbon_stock(altitude_zones)

        print("[DataAnalysis] Analysis complete.")
        return self.results

    def _analyze_overview(self) -> Dict:
        """Generate overview statistics."""
        dem = self.dem
        chm = self.chm
        carbon = self.carbon_density

        overview = {
            "total_pixels": int(self.forest_class.size),
            "valid_pixels": int(np.sum(self.forest_class > 0)),
            "coverage_percent": float(np.sum(self.forest_class > 0) / self.forest_class.size * 100),
            "elevation_range": {
                "min": float(np.min(dem[dem > 0])) if np.any(dem > 0) else 0,
                "max": float(np.max(dem[dem > 0])) if np.any(dem > 0) else 0,
                "mean": float(np.mean(dem[dem > 0])) if np.any(dem > 0) else 0,
            },
            "canopy_height": {
                "min": float(np.min(chm[chm > 0])) if np.any(chm > 0) else 0,
                "max": float(np.max(chm[chm > 0])) if np.any(chm > 0) else 0,
                "mean": float(np.mean(chm[chm > 0])) if np.any(chm > 0) else 0,
                "std": float(np.std(chm[chm > 0])) if np.any(chm > 0) else 0,
            },
            "carbon_density": {
                "min": float(np.min(carbon[carbon > 0])) if np.any(carbon > 0) else 0,
                "max": float(np.max(carbon[carbon > 0])) if np.any(carbon > 0) else 0,
                "mean": float(np.mean(carbon[carbon > 0])) if np.any(carbon > 0) else 0,
                "std": float(np.std(carbon[carbon > 0])) if np.any(carbon > 0) else 0,
                "median": float(np.median(carbon[carbon > 0])) if np.any(carbon > 0) else 0,
            },
        }

        return overview

    def _analyze_by_forest_type(self, altitude_zones: List[Dict]) -> Dict:
        """Analyze statistics by forest type."""
        chm = self.chm
        carbon = self.carbon_density
        forest_class = self.forest_class
        dem = self.dem

        results = {}

        for zone in altitude_zones:
            code = zone.get("class_code", 0)
            if code == 0:
                continue

            mask = forest_class == code
            if not np.any(mask):
                continue

            area_ha = np.sum(mask) * 4 / 10000  # 2m resolution

            results[zone["name"]] = {
                "class_code": code,
                "pixel_count": int(np.sum(mask)),
                "area_ha": float(area_ha),
                "percent_area": float(np.sum(mask) / forest_class.size * 100),
                "elevation": {
                    "min": float(np.min(dem[mask])),
                    "max": float(np.max(dem[mask])),
                    "mean": float(np.mean(dem[mask])),
                },
                "canopy_height": {
                    "mean": float(np.mean(chm[mask])) if np.any(chm[mask] > 0) else 0,
                    "std": float(np.std(chm[mask])) if np.any(chm[mask] > 0) else 0,
                },
                "carbon_density": {
                    "mean": float(np.mean(carbon[mask])) if np.any(carbon[mask] > 0) else 0,
                    "std": float(np.std(carbon[mask])) if np.any(carbon[mask] > 0) else 0,
                    "max": float(np.max(carbon[mask])) if np.any(carbon[mask] > 0) else 0,
                },
                "total_carbon_stock_mgc": float(np.sum(carbon[mask]) * 4 / 10000) if np.any(carbon[mask] > 0) else 0,
            }

        return results

    def _analyze_elevation_patterns(self) -> Dict:
        """Analyze patterns across elevation gradients."""
        dem = self.dem
        chm = self.chm
        carbon = self.carbon_density

        valid_mask = (dem > 0) & (chm > 0) & (carbon > 0)

        if not np.any(valid_mask):
            return {}

        dem_valid = dem[valid_mask]
        chm_valid = chm[valid_mask]
        carbon_valid = carbon[valid_mask]

        # Bin analysis by elevation
        elevation_bins = np.linspace(dem_valid.min(), dem_valid.max(), 10)
        bin_centers = (elevation_bins[:-1] + elevation_bins[1:]) / 2

        binned_stats = []
        for i in range(len(elevation_bins) - 1):
            mask = (dem_valid >= elevation_bins[i]) & (dem_valid < elevation_bins[i + 1])
            if np.sum(mask) > 10:
                binned_stats.append({
                    "elevation_center": float(bin_centers[i]),
                    "mean_height": float(np.mean(chm_valid[mask])),
                    "mean_carbon": float(np.mean(carbon_valid[mask])),
                    "pixel_count": int(np.sum(mask)),
                })

        # Correlation analysis
        corr_elevation_height, p_val_1 = stats.pearsonr(dem_valid.flatten(), chm_valid.flatten())
        corr_elevation_carbon, p_val_2 = stats.pearsonr(dem_valid.flatten(), carbon_valid.flatten())

        return {
            "elevation_bins": binned_stats,
            "correlation_elevation_height": float(corr_elevation_height),
            "correlation_elevation_carbon": float(corr_elevation_carbon),
            "elevation_height_pvalue": float(p_val_1),
            "elevation_carbon_pvalue": float(p_val_2),
        }

    def _identify_carbon_hotspots(self, threshold_percentile: float = 90.0) -> Dict:
        """Identify high-carbon hotspots."""
        carbon = self.carbon_density
        forest_class = self.forest_class

        valid_mask = carbon > 0
        if not np.any(valid_mask):
            return {}

        threshold = np.percentile(carbon[valid_mask], threshold_percentile)
        hotspot_mask = carbon > threshold

        # Count hotspots by forest type
        hotspot_by_type = {}
        for code in range(1, 5):
            type_mask = forest_class == code
            hotspot_count = np.sum(hotspot_mask & type_mask)
            if hotspot_count > 0:
                from .visualizer import FOREST_NAMES
                hotspot_by_type[FOREST_NAMES[code]] = int(hotspot_count)

        return {
            "threshold_mgc_ha": float(threshold),
            "total_hotspot_pixels": int(np.sum(hotspot_mask)),
            "hotspot_area_ha": float(np.sum(hotspot_mask) * 4 / 10000),
            "hotspot_carbon_stock": float(np.sum(carbon[hotspot_mask]) * 4 / 10000),
            "hotspot_by_forest_type": hotspot_by_type,
        }

    def _analyze_correlations(self) -> Dict:
        """Analyze correlations between variables."""
        dem = self.dem
        chm = self.chm
        carbon = self.carbon_density

        valid_mask = (dem > 0) & (chm > 0) & (carbon > 0)

        if not np.any(valid_mask):
            return {}

        dem_flat = dem[valid_mask].flatten()
        chm_flat = chm[valid_mask].flatten()
        carbon_flat = carbon[valid_mask].flatten()

        correlations = {}

        # Height vs Carbon
        r, p = stats.pearsonr(chm_flat, carbon_flat)
        correlations["height_carbon"] = {"r": float(r), "p_value": float(p), "relationship": "strong" if abs(r) > 0.7 else "moderate" if abs(r) > 0.4 else "weak"}

        # Elevation vs Height
        r, p = stats.pearsonr(dem_flat, chm_flat)
        correlations["elevation_height"] = {"r": float(r), "p_value": float(p), "relationship": "strong" if abs(r) > 0.7 else "moderate" if abs(r) > 0.4 else "weak"}

        # Elevation vs Carbon
        r, p = stats.pearsonr(dem_flat, carbon_flat)
        correlations["elevation_carbon"] = {"r": float(r), "p_value": float(p), "relationship": "strong" if abs(r) > 0.7 else "moderate" if abs(r) > 0.4 else "weak"}

        return correlations

    def _calculate_carbon_stock(self, altitude_zones: List[Dict]) -> Dict:
        """Calculate total carbon stock."""
        carbon = self.carbon_density
        forest_class = self.forest_class

        valid_mask = carbon > 0
        pixel_area_ha = 4 / 10000  # 2m resolution

        total_stock = float(np.sum(carbon[valid_mask]) * pixel_area_ha)
        mean_density = float(np.mean(carbon[valid_mask])) if np.any(valid_mask) else 0

        # By forest type
        stock_by_type = {}
        for zone in altitude_zones:
            code = zone.get("class_code", 0)
            if code == 0:
                continue

            mask = (forest_class == code) & (carbon > 0)
            if np.any(mask):
                stock_by_type[zone["name"]] = float(np.sum(carbon[mask]) * pixel_area_ha)

        return {
            "total_carbon_stock_mgc": total_stock,
            "mean_carbon_density_mgc_ha": mean_density,
            "total_area_ha": float(np.sum(valid_mask) * pixel_area_ha),
            "stock_by_forest_type": stock_by_type,
        }

    def export_report(self, output_path: str) -> Path:
        """
        Export analysis report to text file.

        Parameters
        ----------
        output_path : str
            Path for output report

        Returns
        -------
        Path
            Path to saved report
        """
        output_path = Path(output_path)

        lines = []
        lines.append("=" * 70)
        lines.append("CARBON ACCOUNTING DATA ANALYSIS REPORT")
        lines.append("Uttarakhand Forest Carbon Estimation")
        lines.append("=" * 70)
        lines.append("")

        # Overview
        overview = self.results.get("overview", {})
        lines.append("OVERVIEW STATISTICS")
        lines.append("-" * 40)
        lines.append(f"Total Pixels: {overview.get('total_pixels', 'N/A'):,}")
        lines.append(f"Valid Coverage: {overview.get('coverage_percent', 0):.1f}%")
        lines.append(f"Elevation Range: {overview.get('elevation_range', {}).get('min', 0):.0f} - {overview.get('elevation_range', {}).get('max', 0):.0f} m")
        lines.append(f"Mean Canopy Height: {overview.get('canopy_height', {}).get('mean', 0):.1f} m")
        lines.append(f"Mean Carbon Density: {overview.get('carbon_density', {}).get('mean', 0):.1f} MgC/ha")
        lines.append("")

        # Carbon Stock
        stock = self.results.get("carbon_stock", {})
        lines.append("CARBON STOCK SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Total Carbon Stock: {stock.get('total_carbon_stock_mgc', 0):.1f} MgC")
        lines.append(f"Total Area: {stock.get('total_area_ha', 0):.1f} ha")
        lines.append("")
        lines.append("By Forest Type:")
        for ft, val in stock.get("stock_by_forest_type", {}).items():
            lines.append(f"  {ft}: {val:.1f} MgC")
        lines.append("")

        # Forest Type Analysis
        by_type = self.results.get("by_forest_type", {})
        lines.append("FOREST TYPE ANALYSIS")
        lines.append("-" * 40)
        for ft, data in by_type.items():
            lines.append(f"\n{ft}:")
            lines.append(f"  Area: {data.get('area_ha', 0):.1f} ha ({data.get('percent_area', 0):.1f}%)")
            lines.append(f"  Mean Height: {data.get('canopy_height', {}).get('mean', 0):.1f} m")
            lines.append(f"  Mean Carbon: {data.get('carbon_density', {}).get('mean', 0):.1f} MgC/ha")
            lines.append(f"  Total Carbon: {data.get('total_carbon_stock_mgc', 0):.1f} MgC")
        lines.append("")

        # Hotspots
        hotspots = self.results.get("carbon_hotspots", {})
        lines.append("CARBON HOTSPOTS")
        lines.append("-" * 40)
        lines.append(f"Threshold: {hotspots.get('threshold_mgc_ha', 0):.1f} MgC/ha")
        lines.append(f"Hotspot Area: {hotspots.get('hotspot_area_ha', 0):.1f} ha")
        lines.append(f"Hotspot Carbon: {hotspots.get('hotspot_carbon_stock', 0):.1f} MgC")
        lines.append("")

        # Correlations
        correlations = self.results.get("correlations", {})
        lines.append("CORRELATION ANALYSIS")
        lines.append("-" * 40)
        for var, data in correlations.items():
            lines.append(f"{var}: r={data.get('r', 0):.3f} ({data.get('relationship', 'N/A')})")
        lines.append("")

        # Write to file
        with open(output_path, "w") as f:
            f.write("\n".join(lines))

        print(f"[DataAnalysis] Report saved: {output_path}")
        return output_path

    def export_csv(self, output_path: str) -> Path:
        """Export analysis results to CSV."""
        by_type = self.results.get("by_forest_type", {})

        rows = []
        for ft, data in by_type.items():
            rows.append({
                "forest_type": ft,
                "area_ha": data.get("area_ha", 0),
                "percent_area": data.get("percent_area", 0),
                "mean_elevation": data.get("elevation", {}).get("mean", 0),
                "mean_canopy_height": data.get("canopy_height", {}).get("mean", 0),
                "mean_carbon_density": data.get("carbon_density", {}).get("mean", 0),
                "total_carbon_stock_mgc": data.get("total_carbon_stock_mgc", 0),
            })

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)

        print(f"[DataAnalysis] CSV saved: {output_path}")
        return Path(output_path)
