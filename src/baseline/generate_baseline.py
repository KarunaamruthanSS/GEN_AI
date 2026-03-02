# ==========================================================
# Baseline Image Generation Script
# Efficient Windowed Attention Project
# ==========================================================

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import time
import torch
from diffusers import StableDiffusionPipeline
from config import (
    MODEL_NAME,
    DEVICE,
    DTYPE,
    DEFAULT_RESOLUTION,
    NUM_INFERENCE_STEPS,
    GUIDANCE_SCALE,
    SEED,
    OUTPUT_BASELINE_DIR,
    PROMPTS,
    USE_SAFETY_CHECKER
)

# ----------------------------------------------------------
# Setup Output Directory
# ----------------------------------------------------------

os.makedirs(OUTPUT_BASELINE_DIR, exist_ok=True)


# ----------------------------------------------------------
# Load Model
# ----------------------------------------------------------

def load_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE
    )

    if not USE_SAFETY_CHECKER:
        pipe.safety_checker = None

    pipe = pipe.to(DEVICE)
    pipe.enable_attention_slicing()  # helps reduce VRAM
    return pipe


# ----------------------------------------------------------
# Generate Image + Benchmark
# ----------------------------------------------------------

def generate_and_measure(pipe, prompt, resolution):

    generator = torch.Generator(device=DEVICE).manual_seed(SEED)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    start_time = time.time()

    image = pipe(
        prompt,
        height=resolution,
        width=resolution,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator
    ).images[0]

    end_time = time.time()

    runtime = end_time - start_time
    memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # Convert to GB

    return image, runtime, memory


# ----------------------------------------------------------
# Main Execution
# ----------------------------------------------------------

def main():

    print("Loading Stable Diffusion pipeline...")
    pipe = load_pipeline()

    print(f"Using device: {DEVICE}")
    print(f"Resolution: {DEFAULT_RESOLUTION}x{DEFAULT_RESOLUTION}")
    print("-" * 50)

    for idx, prompt in enumerate(PROMPTS):

        print(f"\nGenerating image for prompt: {prompt}")

        image, runtime, memory = generate_and_measure(
            pipe,
            prompt,
            DEFAULT_RESOLUTION
        )

        # Save image
        filename = os.path.join(
            OUTPUT_BASELINE_DIR,
            f"baseline_{idx+1}.png"
        )
        image.save(filename)

        print(f"Saved: {filename}")
        print(f"Runtime: {runtime:.2f} seconds")
        print(f"Peak VRAM: {memory:.2f} GB")
        print("-" * 50)


if __name__ == "__main__":
    main()
