# ==========================================================
# Modified Stable Diffusion Pipeline with Multiple Attention Modes
# Supports: Baseline, Window, Hybrid, Slicing
# ==========================================================
import sys
import os

os.makedirs("outputs/window_images", exist_ok=True)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import torch
from diffusers import StableDiffusionPipeline

from config import (
    MODEL_NAME,
    DEVICE,
    DTYPE,
    USE_SAFETY_CHECKER,
    WINDOW_SIZE,
    HYBRID_SPLIT_DEPTH
)

from src.window_attention.attention_processor import (
    apply_window_attention,
    apply_hybrid_attention,
    apply_attention_slicing
)


# ----------------------------------------------------------
# Load baseline pipeline
# ----------------------------------------------------------

def load_baseline_pipeline():
    """Load standard Stable Diffusion pipeline without modifications."""

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE
    )

    if not USE_SAFETY_CHECKER:
        pipe.safety_checker = None

    pipe = pipe.to(DEVICE)

    return pipe


# ----------------------------------------------------------
# Load pipeline with specific attention mode
# ----------------------------------------------------------

def load_pipeline_with_mode(mode="window", window_size=None, move_to_device=True):
    """
    Load Stable Diffusion pipeline with specified attention mode.
    
    Args:
        mode: "baseline", "window", "hybrid", or "slicing"
        window_size: Window size for window/hybrid modes (uses config default if None)
        move_to_device: If False, skip .to(DEVICE) so the caller can use
                        enable_model_cpu_offload() instead (avoids CUDA OOM).
    
    Returns:
        Configured pipeline (not yet on device if move_to_device=False)
    """
    
    if window_size is None:
        window_size = WINDOW_SIZE
    
    print(f"Loading Stable Diffusion pipeline with mode: {mode}")

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE
    )

    if not USE_SAFETY_CHECKER:
        pipe.safety_checker = None

    # Only move to device if requested.
    # When using enable_model_cpu_offload() the pipeline must NOT be
    # pre-moved to GPU — offload manages device placement itself.
    if move_to_device:
        pipe = pipe.to(DEVICE)

    # Apply attention modification based on mode
    if mode == "baseline":
        print("Using baseline (no modifications).")
    
    elif mode == "window":
        print(f"Applying window attention (size={window_size})...")
        apply_window_attention(pipe.unet, window_size=window_size)
    
    elif mode == "hybrid":
        print(f"Applying hybrid attention (size={window_size}, split={HYBRID_SPLIT_DEPTH})...")
        apply_hybrid_attention(pipe.unet, window_size=window_size, split_depth=HYBRID_SPLIT_DEPTH)
    
    elif mode == "slicing":
        print("Applying attention slicing...")
        apply_attention_slicing(pipe.unet)
    
    else:
        raise ValueError(f"Unknown attention mode: {mode}")

    return pipe


# ----------------------------------------------------------
# Legacy function for backward compatibility
# ----------------------------------------------------------

def load_window_pipeline(window_size=None):
    """Legacy function - use load_pipeline_with_mode instead."""
    return load_pipeline_with_mode("window", window_size)


# ----------------------------------------------------------
# Simple test
# ----------------------------------------------------------

if __name__ == "__main__":

    if DEVICE == "cpu":
        print("CPU detected. Skipping full pipeline load.")
        print("This will be tested later on GPU.")
    else:

        pipe = load_pipeline_with_mode("window")

        prompt = "A futuristic city at sunset"

        image = pipe(prompt).images[0]

        image.save("outputs/window_images/test.png")

        print("Test image saved.")

