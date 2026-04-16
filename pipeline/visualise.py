"""
pipeline/visualise.py
=====================
All visualisation utilities:

  1. per_patch_visual()      — side-by-side RGB + canopy height heatmap per patch
  2. embedding_heatmap()     — PCA of DINOv3 tokens → RGB heatmap (like paper Fig.)
  3. mosaic_visual()         — full-scene: RGB | canopy height mosaic | embedding PCA
  4. save_geotiff()          — write georeferenced output GeoTIFF
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
import rasterio
from rasterio.transform import from_bounds

from .tiling import Patch


# ─── helpers ─────────────────────────────────────────────────────────────────

def _apply_colormap(array: np.ndarray, cmap_name: str, vmin=None, vmax=None) -> np.ndarray:
    """Float array → uint8 (H, W, 3) RGB image via colormap."""
    vmin = vmin if vmin is not None else np.nanmin(array)
    vmax = vmax if vmax is not None else np.nanmax(array)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(norm(array))                    # (H, W, 4)
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def _colorbar_legend(
    ax: plt.Axes,
    cmap_name: str,
    vmin: float,
    vmax: float,
    label: str,
):
    """Add a colourbar to a matplotlib axes."""
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(cmap=cmap_name, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label, fontsize=9)


# ─── 1. per-patch side-by-side visual ────────────────────────────────────────

def per_patch_visual(
    patch: Patch,
    pred: np.ndarray,
    emb: Optional[np.ndarray],
    out_dir: Path,
    cfg: dict,
):
    """
    Save a figure with up to 3 panels:
      [RGB] | [Canopy Height Heatmap] | [Embedding PCA Heatmap]
    """
    cmap_height = cfg["output"].get("colormap", "viridis")
    cmap_emb = cfg["output"].get("embedding_colormap", "turbo")
    save_emb = cfg["output"].get("save_embedding_heatmap", True)

    ncols = 3 if (save_emb and emb is not None) else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5.5), dpi=120)
    fig.suptitle(f"Patch {patch.patch_idx:04d}  "
                 f"(rows {patch.row_start}–{patch.row_end}, "
                 f"cols {patch.col_start}–{patch.col_end})",
                 fontsize=11, fontweight="bold")

    # Panel 1: RGB
    axes[0].imshow(patch.array)
    axes[0].set_title("Input RGB", fontsize=10)
    axes[0].axis("off")

    # Panel 2: Canopy height heatmap
    vmin, vmax = 0.0, max(float(np.nanmax(pred)), 1.0)
    height_rgb = _apply_colormap(pred, cmap_height, vmin=vmin, vmax=vmax)
    im = axes[1].imshow(height_rgb)
    axes[1].set_title("Canopy Height (m)", fontsize=10)
    axes[1].axis("off")
    _colorbar_legend(axes[1], cmap_height, vmin, vmax, "Height (m)")

    # Panel 3: Embedding PCA heatmap (paper-style)
    if ncols == 3:
        emb_rgb = _embedding_pca_rgb(emb, pred.shape, cmap_emb)
        axes[2].imshow(emb_rgb)
        axes[2].set_title("DINOv3 Embeddings (PCA)", fontsize=10)
        axes[2].axis("off")

    plt.tight_layout()
    out_path = out_dir / f"patch_{patch.patch_idx:04d}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ─── 2. Embedding → PCA RGB ──────────────────────────────────────────────────

def _embedding_pca_rgb(
    emb: np.ndarray,
    spatial_shape: tuple,
    cmap_name: str = "turbo",
) -> np.ndarray:
    """
    Project (num_tokens, D) embedding to 3 PCA components → RGB image.
    Mimics the cosine-similarity / feature visualisation in DINOv3 paper.
    """
    H, W = spatial_shape

    if emb is None:
        return np.zeros((H, W, 3), dtype=np.uint8)

    n_tokens, D = emb.shape
    # Expect grid of tokens; find closest square root
    grid_side = int(np.sqrt(n_tokens))
    if grid_side * grid_side < n_tokens:
        # Drop CLS and register tokens if present
        emb_spatial = emb[n_tokens - grid_side * grid_side :]
    else:
        emb_spatial = emb

    try:
        n_comp = min(3, emb_spatial.shape[0], emb_spatial.shape[1])
        pca = PCA(n_components=n_comp)
        proj = pca.fit_transform(emb_spatial)        # (grid², n_comp)

        if n_comp == 3:
            # Normalise each component to [0, 1]
            for i in range(3):
                lo, hi = proj[:, i].min(), proj[:, i].max()
                proj[:, i] = (proj[:, i] - lo) / (hi - lo + 1e-8)

            side = int(np.sqrt(proj.shape[0]))
            rgb_tokens = proj[: side * side].reshape(side, side, 3)
            # Resize to patch resolution
            from PIL import Image as PILImage
            pil = PILImage.fromarray((rgb_tokens * 255).astype(np.uint8), "RGB")
            pil = pil.resize((W, H), PILImage.BILINEAR)
            return np.array(pil)
        else:
            # Fall back to single-channel colourmap
            side = int(np.sqrt(proj.shape[0]))
            mono = proj[:side * side, 0].reshape(side, side)
            return _apply_colormap(mono, cmap_name)

    except Exception:
        return np.zeros((H, W, 3), dtype=np.uint8)


# ─── 3. Full-scene mosaic visual ─────────────────────────────────────────────

def mosaic_visual(
    rgb_full: np.ndarray,
    mosaic: np.ndarray,
    patches: List[Patch],
    predictions: List[np.ndarray],
    embeddings: List[Optional[np.ndarray]],
    out_dir: Path,
    cfg: dict,
):
    """
    Save a 3-panel figure:
      [Full RGB] | [Canopy Height Mosaic] | [Embedding Mosaic (PCA)]
    Also saves each panel individually.
    """
    cmap_height = cfg["output"].get("colormap", "viridis")
    cmap_emb = cfg["output"].get("embedding_colormap", "turbo")
    save_emb = cfg["output"].get("save_embedding_heatmap", True)

    H, W = mosaic.shape
    vmin, vmax = 0.0, max(float(np.nanmax(mosaic)), 1.0)

    # Build embedding mosaic
    emb_mosaic = _build_embedding_mosaic(patches, embeddings, (H, W), cmap_emb) \
        if save_emb else None

    ncols = 3 if emb_mosaic is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 7), dpi=130)
    fig.suptitle("CHMv2 Full-Scene Canopy Height — DINOv3", fontsize=13, fontweight="bold")

    # Crop RGB to mosaic size
    rgb_disp = rgb_full[:H, :W]

    axes[0].imshow(rgb_disp)
    axes[0].set_title("Input RGB (full scene)", fontsize=11)
    axes[0].axis("off")

    height_rgb = _apply_colormap(mosaic, cmap_height, vmin=vmin, vmax=vmax)
    axes[1].imshow(height_rgb)
    axes[1].set_title("Canopy Height Mosaic (m)", fontsize=11)
    axes[1].axis("off")
    _colorbar_legend(axes[1], cmap_height, vmin, vmax, "Height (m)")

    if emb_mosaic is not None:
        axes[2].imshow(emb_mosaic)
        axes[2].set_title("DINOv3 Embedding Mosaic (PCA)", fontsize=11)
        axes[2].axis("off")

    plt.tight_layout()
    out_path = out_dir / "mosaic_visualisation.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[vis] Mosaic visual saved → {out_path}")

    # Also save height-only image
    _save_single_panel(height_rgb, out_dir / "mosaic_canopy_height.png",
                       "Canopy Height Mosaic (m)", cmap_height, vmin, vmax)
    if emb_mosaic is not None:
        _save_single_panel(emb_mosaic, out_dir / "mosaic_embeddings_pca.png",
                           "DINOv3 Embeddings (PCA)", None, None, None)


def _save_single_panel(
    img_rgb: np.ndarray,
    path: Path,
    title: str,
    cmap: Optional[str],
    vmin: Optional[float],
    vmax: Optional[float],
):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=130)
    ax.imshow(img_rgb)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")
    if cmap and vmin is not None:
        _colorbar_legend(ax, cmap, vmin, vmax, "Height (m)")
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[vis] Saved → {path}")


def _build_embedding_mosaic(
    patches: List[Patch],
    embeddings: List[Optional[np.ndarray]],
    full_shape: tuple,
    cmap_name: str,
) -> Optional[np.ndarray]:
    """
    Stitch per-patch embedding PCA images into a full-scene embedding mosaic.
    """
    H, W = full_shape
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    count  = np.zeros((H, W, 1), dtype=np.float32)

    for patch, emb in zip(patches, embeddings):
        if emb is None:
            continue
        ph = patch.row_end - patch.row_start
        pw = patch.col_end - patch.col_start
        emb_rgb = _embedding_pca_rgb(emb, (ph, pw), cmap_name).astype(np.float32)

        r0 = patch.row_start
        r1 = min(patch.row_end, H)
        c0 = patch.col_start
        c1 = min(patch.col_end, W)

        canvas[r0:r1, c0:c1] += emb_rgb[: r1 - r0, : c1 - c0]
        count[r0:r1, c0:c1]  += 1.0

    count = np.where(count == 0, 1.0, count)
    mosaic = np.clip(canvas / count, 0, 255).astype(np.uint8)
    return mosaic


# ─── 4. Save georeferenced GeoTIFF ───────────────────────────────────────────

def save_geotiff(
    array: np.ndarray,
    profile: dict,
    out_path: Path,
    band_name: str = "canopy_height_m",
):
    out_profile = profile.copy()
    
    # Force the profile to recognize the specific NoData value
    out_profile.update({
        "dtype": "float32",
        "count": 1,
        "compress": "lzw",
        "nodata": -9999.0, # Tell QGIS what to ignore
    })
    
    for key in ("photometric", "tiled", "blockxsize", "blockysize"):
        out_profile.pop(key, None)

    # Safely convert NaN to the exact NoData float
    arr = np.copy(array)
    arr[np.isnan(arr)] = -9999.0

    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(arr.astype(np.float32), 1)
        dst.set_band_description(1, band_name) # Safer than update_tags for some QGIS versions

    print(f"[vis] GeoTIFF saved   → {out_path}")