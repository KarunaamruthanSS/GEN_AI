# ==========================================================
# Advanced Scaling Experiment Script
# Compares: Baseline, Window (multiple sizes), Hybrid, Slicing
# Includes: Runtime, Memory, CLIP Score, LPIPS
# ==========================================================

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import time
import csv
import torch
from collections import defaultdict

from config import (
    DEVICE,
    RESOLUTIONS,
    PROMPTS,
    NUM_INFERENCE_STEPS,
    GUIDANCE_SCALE,
    SEED,
    OUTPUT_BASELINE_DIR,
    OUTPUT_WINDOW_DIR,
    RESULTS_CSV_PATH,
    WINDOW_SIZES,
    ENABLE_ABLATION_STUDY,
    ENABLE_QUALITY_METRICS,
    METHODS_TO_COMPARE
)

from src.models.modified_pipeline import (
    load_baseline_pipeline,
    load_pipeline_with_mode
)

from src.utils.metrics import compute_all_metrics


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
    """
    Generate image and measure performance metrics.
    
    Returns:
        image, runtime (seconds), memory (GB)
    """

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
# CSV Management
# ----------------------------------------------------------

def init_csv():
    """Initialize results CSV with headers."""

    headers = [
        "method",
        "window_size",
        "resolution",
        "prompt",
        "runtime_sec",
        "memory_gb"
    ]
    
    if ENABLE_QUALITY_METRICS:
        headers.extend(["clip_score", "lpips_score"])

    with open(RESULTS_CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)


def save_result(method, window_size, resolution, prompt, runtime, memory, 
                clip_score=None, lpips_score=None):
    """Append result to CSV."""

    row = [
        method,
        window_size if window_size else "N/A",
        resolution,
        prompt,
        round(runtime, 3),
        round(memory, 3)
    ]
    
    if ENABLE_QUALITY_METRICS:
        row.append(round(clip_score, 4) if clip_score else "N/A")
        row.append(round(lpips_score, 4) if lpips_score else "N/A")

    with open(RESULTS_CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


# ----------------------------------------------------------
# Main Experiment
# ----------------------------------------------------------

def main():

    if DEVICE == "cpu":
        print("⚠ CPU detected.")
        print("Run this script later on rented GPU.")
        return

    print("=" * 60)
    print("ADVANCED SCALING EXPERIMENT")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Resolutions: {RESOLUTIONS}")
    print(f"Methods: {METHODS_TO_COMPARE}")
    print(f"Ablation Study: {ENABLE_ABLATION_STUDY}")
    print(f"Quality Metrics: {ENABLE_QUALITY_METRICS}")
    print("=" * 60)

    init_csv()

    # Store baseline images for LPIPS comparison
    baseline_images = defaultdict(dict)  # {resolution: {prompt_idx: image}}

    # ----------------------------------------------------------
    # 1. BASELINE
    # ----------------------------------------------------------
    
    if "baseline" in METHODS_TO_COMPARE:
        
        print("\n" + "=" * 60)
        print("RUNNING BASELINE")
        print("=" * 60)
        
        baseline_pipe = load_baseline_pipeline()

        for resolution in RESOLUTIONS:
            
            print(f"\n--- Resolution: {resolution} ---")

            for i, prompt in enumerate(PROMPTS):

                print(f"\nPrompt {i+1}/{len(PROMPTS)}: {prompt}")

                image, runtime, memory = generate_and_measure(
                    baseline_pipe,
                    prompt,
                    resolution
                )

                # Save image
                filename = f"{OUTPUT_BASELINE_DIR}/baseline_{resolution}_{i}.png"
                image.save(filename)
                
                # Store for LPIPS
                baseline_images[resolution][i] = image

                # Compute quality metrics
                clip_score = None
                lpips_score = None
                
                if ENABLE_QUALITY_METRICS:
                    print("Computing quality metrics...")
                    metrics = compute_all_metrics(image, prompt, baseline_image=None)
                    clip_score = metrics.get('clip_score')
                    lpips_score = None  # Baseline has no LPIPS

                print(f"✓ Saved: {filename}")
                print(f"  Runtime: {runtime:.2f}s | Memory: {memory:.2f}GB", end="")
                if clip_score:
                    print(f" | CLIP: {clip_score:.4f}", end="")
                print()

                save_result(
                    "baseline",
                    None,
                    resolution,
                    prompt,
                    runtime,
                    memory,
                    clip_score,
                    lpips_score
                )

        # Clean up
        del baseline_pipe
        torch.cuda.empty_cache()

    # ----------------------------------------------------------
    # 2. WINDOW ATTENTION (with ablation study)
    # ----------------------------------------------------------
    
    if "window" in METHODS_TO_COMPARE:
        
        window_sizes_to_test = WINDOW_SIZES if ENABLE_ABLATION_STUDY else [WINDOW_SIZES[1]]
        
        for window_size in window_sizes_to_test:
            
            print("\n" + "=" * 60)
            print(f"RUNNING WINDOW ATTENTION (size={window_size})")
            print("=" * 60)
            
            window_pipe = load_pipeline_with_mode("window", window_size=window_size)

            for resolution in RESOLUTIONS:
                
                print(f"\n--- Resolution: {resolution} ---")

                for i, prompt in enumerate(PROMPTS):

                    print(f"\nPrompt {i+1}/{len(PROMPTS)}: {prompt}")

                    image, runtime, memory = generate_and_measure(
                        window_pipe,
                        prompt,
                        resolution
                    )

                    # Save image
                    filename = f"{OUTPUT_WINDOW_DIR}/window_{window_size}_{resolution}_{i}.png"
                    image.save(filename)

                    # Compute quality metrics
                    clip_score = None
                    lpips_score = None
                    
                    if ENABLE_QUALITY_METRICS:
                        print("Computing quality metrics...")
                        baseline_img = baseline_images.get(resolution, {}).get(i)
                        metrics = compute_all_metrics(image, prompt, baseline_image=baseline_img)
                        clip_score = metrics.get('clip_score')
                        lpips_score = metrics.get('lpips_score')

                    print(f"✓ Saved: {filename}")
                    print(f"  Runtime: {runtime:.2f}s | Memory: {memory:.2f}GB", end="")
                    if clip_score:
                        print(f" | CLIP: {clip_score:.4f}", end="")
                    if lpips_score:
                        print(f" | LPIPS: {lpips_score:.4f}", end="")
                    print()

                    save_result(
                        "window",
                        window_size,
                        resolution,
                        prompt,
                        runtime,
                        memory,
                        clip_score,
                        lpips_score
                    )

            # Clean up
            del window_pipe
            torch.cuda.empty_cache()

    # ----------------------------------------------------------
    # 3. ATTENTION SLICING
    # ----------------------------------------------------------
    
    if "slicing" in METHODS_TO_COMPARE:
        
        print("\n" + "=" * 60)
        print("RUNNING ATTENTION SLICING")
        print("=" * 60)
        
        slicing_pipe = load_pipeline_with_mode("slicing")

        for resolution in RESOLUTIONS:
            
            print(f"\n--- Resolution: {resolution} ---")

            for i, prompt in enumerate(PROMPTS):

                print(f"\nPrompt {i+1}/{len(PROMPTS)}: {prompt}")

                image, runtime, memory = generate_and_measure(
                    slicing_pipe,
                    prompt,
                    resolution
                )

                # Save image
                filename = f"{OUTPUT_WINDOW_DIR}/slicing_{resolution}_{i}.png"
                image.save(filename)

                # Compute quality metrics
                clip_score = None
                lpips_score = None
                
                if ENABLE_QUALITY_METRICS:
                    print("Computing quality metrics...")
                    baseline_img = baseline_images.get(resolution, {}).get(i)
                    metrics = compute_all_metrics(image, prompt, baseline_image=baseline_img)
                    clip_score = metrics.get('clip_score')
                    lpips_score = metrics.get('lpips_score')

                print(f"✓ Saved: {filename}")
                print(f"  Runtime: {runtime:.2f}s | Memory: {memory:.2f}GB", end="")
                if clip_score:
                    print(f" | CLIP: {clip_score:.4f}", end="")
                if lpips_score:
                    print(f" | LPIPS: {lpips_score:.4f}", end="")
                print()

                save_result(
                    "slicing",
                    None,
                    resolution,
                    prompt,
                    runtime,
                    memory,
                    clip_score,
                    lpips_score
                )

        # Clean up
        del slicing_pipe
        torch.cuda.empty_cache()

    # ----------------------------------------------------------
    # 4. HYBRID ATTENTION
    # ----------------------------------------------------------
    
    if "hybrid" in METHODS_TO_COMPARE:
        
        print("\n" + "=" * 60)
        print("RUNNING HYBRID ATTENTION")
        print("=" * 60)
        
        hybrid_pipe = load_pipeline_with_mode("hybrid")

        for resolution in RESOLUTIONS:
            
            print(f"\n--- Resolution: {resolution} ---")

            for i, prompt in enumerate(PROMPTS):

                print(f"\nPrompt {i+1}/{len(PROMPTS)}: {prompt}")

                image, runtime, memory = generate_and_measure(
                    hybrid_pipe,
                    prompt,
                    resolution
                )

                # Save image
                filename = f"{OUTPUT_WINDOW_DIR}/hybrid_{resolution}_{i}.png"
                image.save(filename)

                # Compute quality metrics
                clip_score = None
                lpips_score = None
                
                if ENABLE_QUALITY_METRICS:
                    print("Computing quality metrics...")
                    baseline_img = baseline_images.get(resolution, {}).get(i)
                    metrics = compute_all_metrics(image, prompt, baseline_image=baseline_img)
                    clip_score = metrics.get('clip_score')
                    lpips_score = metrics.get('lpips_score')

                print(f"✓ Saved: {filename}")
                print(f"  Runtime: {runtime:.2f}s | Memory: {memory:.2f}GB", end="")
                if clip_score:
                    print(f" | CLIP: {clip_score:.4f}", end="")
                if lpips_score:
                    print(f" | LPIPS: {lpips_score:.4f}", end="")
                print()

                save_result(
                    "hybrid",
                    None,
                    resolution,
                    prompt,
                    runtime,
                    memory,
                    clip_score,
                    lpips_score
                )

        # Clean up
        del hybrid_pipe
        torch.cuda.empty_cache()

    # ----------------------------------------------------------
    # DONE
    # ----------------------------------------------------------

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {RESULTS_CSV_PATH}")
    print("\nRun plotting script to visualize results:")
    print("  python src/utils/plot_utils.py")


# ----------------------------------------------------------

if __name__ == "__main__":
    main()
