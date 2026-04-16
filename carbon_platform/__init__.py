"""
Carbon Accounting Platform for Uttarakhand, India.

Stream-Predict-Cube workflow for state-scale canopy height and carbon estimation.
"""

__version__ = "2.5.0"

from .datacube import DataCubeManager
from .allometry import AllometryCalculator
from .dem_classifier import DEMClassifier
from .inference_engine import CarbonInferenceEngine

__all__ = [
    "DataCubeManager",
    "AllometryCalculator",
    "DEMClassifier",
    "CarbonInferenceEngine",
]
