# ==========================================================
# Scaling Experiment Script
# Compares Baseline vs Window Attention
# ==========================================================

import os
import time
import csv
import torch

from config import (
    DEVICE,
    RESOLUTIONS,
    PROMPTS,
    NUM_INFERENCE_STEPS,
    GUIDANCE_SCALE,
    SEED,
    OUTPUT_BASELINE_DIR,
    OUTPUT_WINDOW_DIR,
    RESULTS_CSV_PATH
)

from src.models.modified_pipeline import (
    load_baseline_pipeline,
    load_window_pipeline
)


# ----------------------------------------------------------
# Create folders
# ----------------------------------------------------------

os.makedirs(OUTPUT_BASELINE_DIR, exist_ok=True)
os.makedirs(OUTPUT_WINDOW_DIR, exist_ok=True)
os.makedirs("experiments", exist_ok=True)


# ----------------------------------------------------------
# Measure runtime and memory
# ----------------------------------------------------------

def generate_and_measure(pipe, prompt, resolution):

    generator = torch.Generator(device=DEVICE).manual_seed(SEED)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    start = time.time()

    image = pipe(
        prompt,
        height=resolution,
        width=resolution,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator
    ).images[0]

    end = time.time()

    runtime = end - start
    memory = torch.cuda.max_memory_allocated() / (1024 ** 3)

    return image, runtime, memory


# ----------------------------------------------------------
# Save CSV header
# ----------------------------------------------------------

def init_csv():

    with open(RESULTS_CSV_PATH, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "method",
            "resolution",
            "prompt",
            "runtime_sec",
            "memory_gb"
        ])


# ----------------------------------------------------------
# Append result
# ----------------------------------------------------------

def save_result(method, resolution, prompt, runtime, memory):

    with open(RESULTS_CSV_PATH, "a", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            method,
            resolution,
            prompt,
            round(runtime, 3),
            round(memory, 3)
        ])


# ----------------------------------------------------------
# Main Experiment
# ----------------------------------------------------------

def main():

    if DEVICE == "cpu":
        print("⚠ CPU detected.")
        print("Run this script later on rented GPU.")
        return


    print("Initializing experiment...")

    init_csv()

    print("\nLoading baseline pipeline...")
    baseline_pipe = load_baseline_pipeline()

    print("\nLoading window attention pipeline...")
    window_pipe = load_window_pipeline()


    for resolution in RESOLUTIONS:

        print(f"\n===== Resolution: {resolution} =====")

        for i, prompt in enumerate(PROMPTS):

            print(f"\nPrompt: {prompt}")

            # ----------------------------------
            # Baseline
            # ----------------------------------

            print("Running baseline...")

            image, runtime, memory = generate_and_measure(
                baseline_pipe,
                prompt,
                resolution
            )

            filename = f"{OUTPUT_BASELINE_DIR}/baseline_{resolution}_{i}.png"
            image.save(filename)

            print(f"Saved: {filename}")
            print(f"Time: {runtime:.2f}s | Memory: {memory:.2f}GB")

            save_result(
                "baseline",
                resolution,
                prompt,
                runtime,
                memory
            )


            # ----------------------------------
            # Window Attention
            # ----------------------------------

            print("Running window attention...")

            image, runtime, memory = generate_and_measure(
                window_pipe,
                prompt,
                resolution
            )

            filename = f"{OUTPUT_WINDOW_DIR}/window_{resolution}_{i}.png"
            image.save(filename)

            print(f"Saved: {filename}")
            print(f"Time: {runtime:.2f}s | Memory: {memory:.2f}GB")

            save_result(
                "window",
                resolution,
                prompt,
                runtime,
                memory
            )


    print("\nExperiment complete.")
    print(f"Results saved to: {RESULTS_CSV_PATH}")


# ----------------------------------------------------------

if __name__ == "__main__":
    main()