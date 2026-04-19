"""pipeline/inference.py — Batched CHMv2 inference on ESRI patches."""

from __future__ import annotations
from typing import List, Tuple, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .tiling import Patch


def run_inference(
    patches: List[Patch],
    model,
    processor,
    device: torch.device,
    cfg: dict,
) -> List[np.ndarray]:
    """
    Run CHMv2 on a list of 512×512 RGB patches.

    Returns
    -------
    predictions : list of float32 (H, W) canopy height arrays, one per patch
    """
    batch_size = cfg["model"].get("batch_size", 2)
    show_bar = cfg["logging"].get("progress_bar", True)

    predictions: List[np.ndarray] = []
    batch_indices = range(0, len(patches), batch_size)
    iterator = tqdm(batch_indices, desc="CHMv2 inference", unit="batch") if show_bar else batch_indices

    with torch.no_grad():
        for i in iterator:
            batch = patches[i : i + batch_size]
            pil_imgs = [Image.fromarray(p.array, mode="RGB") for p in batch]
            target_sizes = [(p.array.shape[0], p.array.shape[1]) for p in batch]

            inputs = processor(images=pil_imgs, return_tensors="pt").to(device)
            outputs = model(**inputs)

            depth_maps = processor.post_process_depth_estimation(outputs, target_sizes=target_sizes)
            for dmap in depth_maps:
                pred = dmap["predicted_depth"].squeeze().cpu().numpy().astype(np.float32)
                predictions.append(pred)

    print(f"[inference] Done — {len(predictions)} patches processed.")
    return predictions
