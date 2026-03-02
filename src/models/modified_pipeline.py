# ==========================================================
# Modified Stable Diffusion Pipeline with Window Attention
# ==========================================================

import torch
from diffusers import StableDiffusionPipeline

from config import (
    MODEL_NAME,
    DEVICE,
    DTYPE,
    USE_SAFETY_CHECKER
)

from src.window_attention.attention_processor import apply_window_attention


# ----------------------------------------------------------
# Load baseline pipeline
# ----------------------------------------------------------

def load_baseline_pipeline():

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE
    )

    if not USE_SAFETY_CHECKER:
        pipe.safety_checker = None

    pipe = pipe.to(DEVICE)

    return pipe


# ----------------------------------------------------------
# Load modified pipeline with window attention
# ----------------------------------------------------------

def load_window_pipeline():

    print("Loading Stable Diffusion pipeline...")

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE
    )

    if not USE_SAFETY_CHECKER:
        pipe.safety_checker = None

    pipe = pipe.to(DEVICE)

    print("Applying window attention...")

    apply_window_attention(pipe.unet)

    print("Window attention enabled.")

    return pipe


# ----------------------------------------------------------
# Simple test
# ----------------------------------------------------------

if __name__ == "__main__":

    if DEVICE == "cpu":
        print("CPU detected. Skipping full pipeline load.")
        print("This will be tested later on GPU.")
    else:

        pipe = load_window_pipeline()

        prompt = "A futuristic city at sunset"

        image = pipe(prompt).images[0]

        image.save("outputs/window_images/test.png")

        print("Test image saved.")