"""
pipeline/tiling.py
==================
Read a multi-band GeoTIFF (RGB), upscale it to match model resolution,
extract patches with overlap, and reconstruct (mosaic) the full scene.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import rasterio
from rasterio import Affine
from PIL import Image


@dataclass
class Patch:
    """One image patch with its location metadata."""
    array: np.ndarray        # uint8 RGB  (H, W, 3)
    row_start: int
    col_start: int
    row_end: int
    col_end: int
    patch_idx: int


def load_rgb_image(cfg: dict) -> Tuple[np.ndarray, rasterio.profiles.Profile]:
    """
    Load a GeoTIFF, apply a smart contrast stretch, and upscale.
    """
    image_path = cfg["input"]["image_path"]
    band_order = cfg["input"].get("band_order", [1, 2, 3])
    scale_factor = cfg["input"].get("upscale_factor", 5)

    with rasterio.open(image_path) as src:
        profile = src.profile.copy()
        
        if scale_factor > 1:
            print(f"[tiling] Upscaling image by {scale_factor}x during read...")
            new_height = int(src.height * scale_factor)
            new_width = int(src.width * scale_factor)
            
            bands = src.read(
                band_order,
                out_shape=(3, new_height, new_width),
                resampling=rasterio.enums.Resampling.cubic
            )
            
            transform = src.transform * src.transform.scale(
                (src.width / bands.shape[-1]),
                (src.height / bands.shape[-2])
            )
            
            profile.update({
                "height": new_height,
                "width": new_width,
                "transform": transform
            })
        else:
            bands = src.read(band_order)

    arr = np.transpose(bands, (1, 2, 0)).astype(np.float32)

    # --- SMART NORMALIZATION ---
    # Instead of raw 0-10000 or dynamic percentiles, we use fixed physical thresholds
    # typical for land cover to ensure the forest isn't crushed to black.
    
    # 1. Define sensible min and max reflectance values for land
    MIN_VAL = 0.0
    MAX_VAL = 3000.0 # Anything brighter than this (clouds/white roofs) gets clipped to max white.
    
    # 2. Clip the data to this range
    arr = np.clip(arr, MIN_VAL, MAX_VAL)
    
    # 3. Scale to 0-255 based on this new stretched range
    arr = ((arr - MIN_VAL) / (MAX_VAL - MIN_VAL)) * 255.0

    rgb = arr.astype(np.uint8)
    print(f"[tiling] Final Image Shape: {rgb.shape}")
    
    return rgb, profile

def extract_patches(
    rgb: np.ndarray,
    patch_size: int,
    overlap: int,
) -> List[Patch]:
    """
    Tile rgb (H, W, 3) into overlapping patches of size patch_size×patch_size.
    Pads the image with reflection padding so every patch is full-size.
    """
    H, W, _ = rgb.shape
    stride = patch_size - overlap

    # Pad so dimensions are multiples of stride
    pad_h = (stride - (H % stride)) % stride
    pad_w = (stride - (W % stride)) % stride
    if pad_h or pad_w:
        rgb_padded = np.pad(
            rgb,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode="reflect",
        )
    else:
        rgb_padded = rgb

    pH, pW, _ = rgb_padded.shape
    patches = []
    idx = 0

    row = 0
    while row + patch_size <= pH:
        col = 0
        while col + patch_size <= pW:
            patch_arr = rgb_padded[row : row + patch_size, col : col + patch_size]
            patches.append(
                Patch(
                    array=patch_arr,
                    row_start=row,
                    col_start=col,
                    row_end=row + patch_size,
                    col_end=col + patch_size,
                    patch_idx=idx,
                )
            )
            idx += 1
            col += stride
        row += stride

    print(
        f"[tiling] Patch size={patch_size}px  overlap={overlap}px  "
        f"→ {len(patches)} patches  (padded image: {rgb_padded.shape[:2]})"
    )
    return patches, rgb_padded


def mosaic_patches(
    patches: List[Patch],
    predictions: List[np.ndarray],
    padded_shape: Tuple[int, int],
    original_shape: Tuple[int, int],
    overlap: int,
    blend_mode: str = "linear",
) -> np.ndarray:
    """
    Reconstruct full-scene canopy height from patch predictions.
    """
    pH, pW = padded_shape
    canvas = np.zeros((pH, pW), dtype=np.float64)
    weight = np.zeros((pH, pW), dtype=np.float64)

    for patch, pred in zip(patches, predictions):
        h, w = pred.shape
        if blend_mode == "linear" and overlap > 0:
            wy = _feather_1d(h, overlap)
            wx = _feather_1d(w, overlap)
            w_mask = np.outer(wy, wx)
        else:
            w_mask = np.ones((h, w), dtype=np.float64)

        canvas[patch.row_start : patch.row_end, patch.col_start : patch.col_end] += (
            pred.astype(np.float64) * w_mask
        )
        weight[patch.row_start : patch.row_end, patch.col_start : patch.col_end] += w_mask

    weight = np.where(weight == 0, 1e-8, weight)
    mosaic_full = (canvas / weight).astype(np.float32)

    H, W = original_shape
    return mosaic_full[:H, :W]


def _feather_1d(length: int, overlap: int) -> np.ndarray:
    """Linear ramp weights for feathered blending."""
    w = np.ones(length, dtype=np.float64)
    ramp = np.linspace(0.0, 1.0, overlap, endpoint=False)
    w[:overlap] = ramp
    w[length - overlap :] = ramp[::-1]
    return w