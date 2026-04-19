"""pipeline/tiling.py — Patch dataclass for ESRI 512×512 PNG patches."""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Patch:
    array: np.ndarray   # uint8 RGB (512, 512, 3)
    patch_idx: int
    name: str = field(default="")
