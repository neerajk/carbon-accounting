"""
Carbon Accounting Platform for Uttarakhand, India.

Stream-Predict-Cube workflow for state-scale canopy height and carbon estimation.
"""

__version__ = "2.5.0"

# Lazy imports to avoid dependency issues when only using subset of modules
def __getattr__(name):
    if name == "DataCubeManager":
        from .datacube import DataCubeManager
        return DataCubeManager
    elif name == "AllometryCalculator":
        from .allometry import AllometryCalculator
        return AllometryCalculator
    elif name == "DEMClassifier":
        from .dem_classifier import DEMClassifier
        return DEMClassifier
    elif name == "CarbonInferenceEngine":
        from .inference_engine import CarbonInferenceEngine
        return CarbonInferenceEngine
    elif name == "GeoTIFFManager":
        from .geotiff_manager import GeoTIFFManager
        return GeoTIFFManager
    raise AttributeError(f"module {__name__!r} has no attribute {__name__!r}")

__all__ = [
    "DataCubeManager",
    "AllometryCalculator",
    "DEMClassifier",
    "CarbonInferenceEngine",
    "GeoTIFFManager",
]
