"""
pipeline/inference.py
=====================
Run CHMv2 on a list of patches (BATCHED).
Collects:
  - per-patch canopy height predictions  (metres, float32)
  - per-patch DINOv3 backbone embeddings (for PCA heatmap visualisation)
"""

from __future__ import annotations
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .tiling import Patch


def run_patch_inference(
    patches: List[Patch],
    model,
    processor,
    device: torch.device,
    cfg: dict,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Parameters
    ----------
    patches   : list of Patch objects
    model     : CHMv2ForDepthEstimation
    processor : CHMv2ImageProcessorFast
    device    : torch.device
    cfg       : full config dict

    Returns
    -------
    predictions  : list of float32 (H, W) canopy height arrays
    embeddings   : list of float32 (num_tokens, embed_dim) patch-level features
    """
    verbose = cfg["logging"].get("verbose", True)
    show_bar = cfg["logging"].get("progress_bar", True)
    
    # Grab batch size from config, default to 4 if not specified
    batch_size = cfg["model"].get("batch_size", 4)

    predictions: List[np.ndarray] = []
    embeddings: List[np.ndarray] = []

    # Create chunked ranges for batching
    batch_indices = list(range(0, len(patches), batch_size))
    iterator = tqdm(batch_indices, desc=f"Running CHMv2 inference (Batch Size: {batch_size})", unit="batch") \
        if show_bar else batch_indices

    with torch.no_grad():
        for i in iterator:
            batch_patches = patches[i : i + batch_size]
            
            # Prepare the batch of images and their target sizes
            pil_imgs = [Image.fromarray(p.array, mode="RGB") for p in batch_patches]
            target_sizes = [(p.array.shape[0], p.array.shape[1]) for p in batch_patches]

            # Preprocess the entire batch at once
            inputs = processor(images=pil_imgs, return_tensors="pt").to(device)

            # Forward pass — batched prediction
            outputs = model(**inputs, output_hidden_states=True)

            # Post-process depth estimation for the whole batch
            depth_maps = processor.post_process_depth_estimation(
                outputs,
                target_sizes=target_sizes,
            )
            
            # Extract heights for each image in the batch
            for dmap in depth_maps:
                pred_np = dmap["predicted_depth"].squeeze().cpu().numpy().astype(np.float32)
                predictions.append(pred_np)

            # Extract DINOv3 backbone embeddings for the batch
            try:
                hs = outputs.hidden_states
                if hs is not None and len(hs) > 0:
                    last_hs = hs[-1] # This might be (B, N, D) or (B, C, H, W)
                    
                    for j in range(len(batch_patches)):
                        emb = last_hs[j].cpu().numpy().astype(np.float32)
                        
                        # --- THE FIX ---
                        # HuggingFace DPT reshapes ViT tokens into spatial maps (C, H, W).
                        # PCA expects a 2D list of tokens (num_tokens, C).
                        # We transpose and flatten it back to the expected shape.
                        if emb.ndim == 3:
                            C, H_feat, W_feat = emb.shape
                            emb = emb.reshape(C, -1).T
                            
                        embeddings.append(emb)
                else:
                    embeddings.extend([None] * len(batch_patches))
            except AttributeError:
                embeddings.extend([None] * len(batch_patches))

    print(f"\n[inference] Completed {len(predictions)} patches.")
    return predictions, embeddings