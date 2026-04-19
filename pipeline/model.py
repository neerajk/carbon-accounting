"""
pipeline/model.py
=================
Load CHMv2 (DINOv3) model and processor from HuggingFace Hub.
"""

import torch
from transformers import CHMv2ForDepthEstimation, CHMv2ImageProcessorFast


def load_model_and_processor(cfg: dict):
    """
    Download (first run) and load CHMv2 model + processor.

    Returns
    -------
    model      : CHMv2ForDepthEstimation  (eval mode)
    processor  : CHMv2ImageProcessorFast
    device     : torch.device
    """
    model_id = cfg["model"]["hf_model_id"]
    device_str = cfg["model"]["device"]
    dtype_str = cfg["model"].get("dtype", "float32")

    device = torch.device(device_str)
    dtype = torch.float32 if dtype_str == "float32" else torch.float16

    print(f"[model] Loading {model_id}")
    print(f"[model] Device={device}  dtype={dtype}")

    processor = CHMv2ImageProcessorFast.from_pretrained(model_id)
    model = CHMv2ForDepthEstimation.from_pretrained(model_id, torch_dtype=dtype)
    model = model.to(device)
    model.eval()

    print("[model] Ready.\n")
    return model, processor, device
