"""
pipeline/runner.py
==================
Orchestrates the full CHMv2 inference pipeline:

  1. Load model
  2. Load & tile input image
  3. Run inference per patch
  4. Mosaic predictions
  5. Save GeoTIFFs + visualisations

Orchestrates the full CHMv2 inference pipeline:
  - StacInferencePipeline: For full-scene GeoTIFFs (tiling & mosaicing)
  - EsriPatchInferencePipeline: For directories of native-res PNG patches
"""

from __future__ import annotations
from pathlib import Path
from typing import List

import numpy as np
import rasterio
from rasterio.transform import Affine  
from PIL import Image
from tqdm import tqdm

from .model import load_model_and_processor
from .tiling import load_rgb_image, extract_patches, mosaic_patches
from .inference import run_patch_inference
from .visualise import (
    per_patch_visual,
    mosaic_visual,
    save_geotiff,
    _embedding_pca_rgb,  
)


class StacInferencePipeline:
    """Standard pipeline: Upscales a single TIF, tiles it, runs inference, and mosaics."""
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def run(self):
        cfg = self.cfg

        # ── 0. Prepare output directories ──────────────────────────────────
        out_dir = Path(cfg["output"]["output_dir"])
        patches_dir = out_dir / "patches"
        out_dir.mkdir(parents=True, exist_ok=True)
        patches_dir.mkdir(parents=True, exist_ok=True)

        # ── 1. Load model ───────────────────────────────────────────────────
        model, processor, device = load_model_and_processor(cfg)

        # ── 2. Load image ───────────────────────────────────────────────────
        rgb, geo_profile = load_rgb_image(cfg)
        original_shape = rgb.shape[:2]          # (H, W)

        # ── 3. Tile image ───────────────────────────────────────────────────
        patch_size = cfg["tiling"]["patch_size"]
        overlap    = cfg["tiling"]["overlap"]
        blend_mode = cfg["tiling"].get("blend_mode", "linear")

        patches, rgb_padded = extract_patches(rgb, patch_size, overlap)
        padded_shape = rgb_padded.shape[:2]

        # ── 4. Run inference ────────────────────────────────────────────────
        predictions, embeddings = run_patch_inference(
            patches, model, processor, device, cfg
        )

        # ── 5. Per-patch visualisations ─────────────────────────────────────
        if cfg["output"].get("save_patch_visuals", True):
            print("[runner] Saving per-patch visualisations…")
            for patch, pred, emb in zip(patches, predictions, embeddings):
                per_patch_visual(patch, pred, emb, patches_dir, cfg)
            print(f"[runner] Patch visuals → {patches_dir}/")

        # ── 6. Mosaic ───────────────────────────────────────────────────────
        print("[runner] Mosaicking patches…")
        mosaic = mosaic_patches(
            patches,
            predictions,
            padded_shape,
            original_shape,
            overlap,
            blend_mode,
        )
        print(
            f"[runner] Mosaic shape: {mosaic.shape}  "
            f"min={mosaic.min():.2f}m  max={mosaic.max():.2f}m"
        )

        # ── 7. Save mosaic GeoTIFF ──────────────────────────────────────────
        if cfg["output"].get("save_mosaic_tif", True):
            mosaic_tif = out_dir / "canopy_height_mosaic.tif"
            save_geotiff(mosaic, geo_profile, mosaic_tif, band_name="canopy_height_m")

        # ── 8. Full-scene visualisation ─────────────────────────────────────
        if cfg["output"].get("save_mosaic_visual", True):
            print("[runner] Generating full-scene visualisation…")
            mosaic_visual(
                rgb_full=rgb,
                mosaic=mosaic,
                patches=patches,
                predictions=predictions,
                embeddings=embeddings,
                out_dir=out_dir,
                cfg=cfg,
            )

        # ── 9. Summary ──────────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print("  STAC Pipeline complete! Outputs:")
        print(f"  GeoTIFF  : {out_dir / 'canopy_height_mosaic.tif'}")
        print(f"  Mosaic   : {out_dir / 'mosaic_visualisation.png'}")
        print(f"  Patches  : {patches_dir}/")
        print(f"{'='*60}\n")


class EsriPatchInferencePipeline:
    """Direct patch pipeline: Reads native PNGs from a directory, runs inference, saves float32 TIFs."""
    def __init__(self, cfg: dict, input_dir: str):
        self.cfg = cfg
        self.input_dir = Path(input_dir)

    def run(self):
        cfg = self.cfg
        
        # ── 0. Prepare output directories ──────────────────────────────────
        out_dir = Path(cfg["output"]["output_dir"]) / "esri_iirs_results"
        out_dir.mkdir(parents=True, exist_ok=True)

        # ── 1. Load model ───────────────────────────────────────────────────
        print("[runner] Loading model...")
        model, processor, device = load_model_and_processor(cfg)

        # ── 2. Find PNG Patches ─────────────────────────────────────────────
        png_files = sorted(self.input_dir.glob("*.png"))
        if not png_files:
            print(f"[runner] ❌ No .png files found in {self.input_dir}")
            return

        print(f"[runner] Found {len(png_files)} PNG patches in {self.input_dir}. Running direct inference...")

        # ──> Create a simple wrapper class to mimic the STAC patch structure
        class NativePatch:
            def __init__(self, arr):
                self.array = arr

        # ── 3. Run Inference per Patch ──────────────────────────────────────
        for png_path in tqdm(png_files, desc="Processing PNGs"):
            # Load standard RGB
            img = Image.open(png_path).convert("RGB")
            patch_rgb = np.array(img)

            # Pass the wrapped array to the inference function
            predictions, embeddings = run_patch_inference(
                [NativePatch(patch_rgb)], model, processor, device, cfg
            )
            
            # Extract the single result
            height_pred = predictions[0]
            emb_pred = embeddings[0]

            # ── 4. Save Outputs ──────────────────────────────────────────────
            base_name = png_path.stem
            
            # Save Height as a float32 TIF
            chm_out = out_dir / f"{base_name}_CHM.tif"
            with rasterio.open(
                chm_out, 'w', driver='GTiff',
                height=height_pred.shape[0], width=height_pred.shape[1],
                count=1, dtype=rasterio.float32,
                transform=Affine.identity()  # <--- Fix: Explicitly tells rasterio there are no coords
            ) as dst:
                dst.write(height_pred.astype(rasterio.float32), 1)

            # --- Save Embeddings as an RGB PNG using visualise.py! ---
            if emb_pred is not None:
                # Process tokens to RGB using the exact method from the paper
                emb_rgb = _embedding_pca_rgb(
                    emb=emb_pred, 
                    spatial_shape=height_pred.shape, 
                    cmap_name=cfg["output"].get("embedding_colormap", "turbo")
                )
                
                # Save it directly as a standard PNG image
                emb_out = out_dir / f"{base_name}_EMB.png"
                Image.fromarray(emb_rgb).save(emb_out)

        print(f"\n{'='*60}")
        print("  ESRI Pipeline complete! Outputs saved to:")
        print(f"  {out_dir}/")
        print(f"{'='*60}\n")