# ==========================================================
# Advanced Plot Utilities
# Generates comprehensive visualizations for research paper
# ==========================================================
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from config import RESULTS_CSV_PATH, OUTPUT_PLOTS_DIR


# ----------------------------------------------------------
# Create plots folder
# ----------------------------------------------------------

os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)


# ----------------------------------------------------------
# Load data
# ----------------------------------------------------------

def load_data():
    """Load experiment results from CSV."""

    if not os.path.exists(RESULTS_CSV_PATH):
        print("Results file not found:", RESULTS_CSV_PATH)
        return None

    df = pd.read_csv(RESULTS_CSV_PATH)
    
    # Clean up window_size column
    df['window_size'] = df['window_size'].replace('N/A', np.nan)
    
    return df


# ----------------------------------------------------------
# Plot 1: Runtime vs Resolution (by method)
# ----------------------------------------------------------

def plot_runtime_by_method(df):
    """Compare runtime across methods at different resolutions."""

    grouped = df.groupby(["method", "resolution"])["runtime_sec"].mean().reset_index()

    plt.figure(figsize=(10, 6))

    for method in grouped["method"].unique():
        subset = grouped[grouped["method"] == method]
        plt.plot(
            subset["resolution"],
            subset["runtime_sec"],
            marker="o",
            linewidth=2,
            markersize=8,
            label=method
        )

    plt.title("Runtime vs Resolution (by Method)", fontsize=14, fontweight='bold')
    plt.xlabel("Resolution", fontsize=12)
    plt.ylabel("Runtime (seconds)", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUT_PLOTS_DIR, "runtime_vs_resolution_by_method.png")
    plt.savefig(path, dpi=300)
    plt.close()

    print("✓ Saved:", path)


# ----------------------------------------------------------
# Plot 2: Memory vs Resolution (by method)
# ----------------------------------------------------------

def plot_memory_by_method(df):
    """Compare memory usage across methods at different resolutions."""

    grouped = df.groupby(["method", "resolution"])["memory_gb"].mean().reset_index()

    plt.figure(figsize=(10, 6))

    for method in grouped["method"].unique():
        subset = grouped[grouped["method"] == method]
        plt.plot(
            subset["resolution"],
            subset["memory_gb"],
            marker="s",
            linewidth=2,
            markersize=8,
            label=method
        )

    plt.title("Memory Usage vs Resolution (by Method)", fontsize=14, fontweight='bold')
    plt.xlabel("Resolution", fontsize=12)
    plt.ylabel("Memory (GB)", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUT_PLOTS_DIR, "memory_vs_resolution_by_method.png")
    plt.savefig(path, dpi=300)
    plt.close()

    print("✓ Saved:", path)


# ----------------------------------------------------------
# Plot 3: Window Size Ablation (Runtime)
# ----------------------------------------------------------

def plot_window_size_ablation_runtime(df):
    """Ablation study: runtime vs window size at different resolutions."""

    # Filter only window method
    window_df = df[df["method"] == "window"].copy()
    
    if window_df.empty or window_df['window_size'].isna().all():
        print("⚠ No window size data available for ablation study")
        return

    grouped = window_df.groupby(["window_size", "resolution"])["runtime_sec"].mean().reset_index()

    plt.figure(figsize=(10, 6))

    for resolution in sorted(grouped["resolution"].unique()):
        subset = grouped[grouped["resolution"] == resolution]
        plt.plot(
            subset["window_size"],
            subset["runtime_sec"],
            marker="o",
            linewidth=2,
            markersize=8,
            label=f"{resolution}px"
        )

    plt.title("Window Size Ablation: Runtime", fontsize=14, fontweight='bold')
    plt.xlabel("Window Size", fontsize=12)
    plt.ylabel("Runtime (seconds)", fontsize=12)
    plt.legend(title="Resolution", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUT_PLOTS_DIR, "ablation_window_size_runtime.png")
    plt.savefig(path, dpi=300)
    plt.close()

    print("✓ Saved:", path)


# ----------------------------------------------------------
# Plot 4: Window Size Ablation (Memory)
# ----------------------------------------------------------

def plot_window_size_ablation_memory(df):
    """Ablation study: memory vs window size at different resolutions."""

    # Filter only window method
    window_df = df[df["method"] == "window"].copy()
    
    if window_df.empty or window_df['window_size'].isna().all():
        print("⚠ No window size data available for ablation study")
        return

    grouped = window_df.groupby(["window_size", "resolution"])["memory_gb"].mean().reset_index()

    plt.figure(figsize=(10, 6))

    for resolution in sorted(grouped["resolution"].unique()):
        subset = grouped[grouped["resolution"] == resolution]
        plt.plot(
            subset["window_size"],
            subset["memory_gb"],
            marker="s",
            linewidth=2,
            markersize=8,
            label=f"{resolution}px"
        )

    plt.title("Window Size Ablation: Memory Usage", fontsize=14, fontweight='bold')
    plt.xlabel("Window Size", fontsize=12)
    plt.ylabel("Memory (GB)", fontsize=12)
    plt.legend(title="Resolution", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUT_PLOTS_DIR, "ablation_window_size_memory.png")
    plt.savefig(path, dpi=300)
    plt.close()

    print("✓ Saved:", path)


# ----------------------------------------------------------
# Plot 5: CLIP Score Comparison
# ----------------------------------------------------------

def plot_clip_scores(df):
    """Compare CLIP scores across methods."""

    if 'clip_score' not in df.columns:
        print("⚠ No CLIP score data available")
        return

    # Remove N/A values
    df_clean = df[df['clip_score'] != 'N/A'].copy()
    df_clean['clip_score'] = pd.to_numeric(df_clean['clip_score'])

    grouped = df_clean.groupby(["method", "resolution"])["clip_score"].mean().reset_index()

    plt.figure(figsize=(10, 6))

    for method in grouped["method"].unique():
        subset = grouped[grouped["method"] == method]
        plt.plot(
            subset["resolution"],
            subset["clip_score"],
            marker="D",
            linewidth=2,
            markersize=8,
            label=method
        )

    plt.title("CLIP Score vs Resolution (Text-Image Alignment)", fontsize=14, fontweight='bold')
    plt.xlabel("Resolution", fontsize=12)
    plt.ylabel("CLIP Score", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUT_PLOTS_DIR, "clip_score_comparison.png")
    plt.savefig(path, dpi=300)
    plt.close()

    print("✓ Saved:", path)


# ----------------------------------------------------------
# Plot 6: LPIPS Score Comparison
# ----------------------------------------------------------

def plot_lpips_scores(df):
    """Compare LPIPS scores (perceptual similarity to baseline)."""

    if 'lpips_score' not in df.columns:
        print("⚠ No LPIPS score data available")
        return

    # Remove N/A values and baseline (which has no LPIPS)
    df_clean = df[(df['lpips_score'] != 'N/A') & (df['method'] != 'baseline')].copy()
    
    if df_clean.empty:
        print("⚠ No valid LPIPS data")
        return
    
    df_clean['lpips_score'] = pd.to_numeric(df_clean['lpips_score'])

    grouped = df_clean.groupby(["method", "resolution"])["lpips_score"].mean().reset_index()

    plt.figure(figsize=(10, 6))

    for method in grouped["method"].unique():
        subset = grouped[grouped["method"] == method]
        plt.plot(
            subset["resolution"],
            subset["lpips_score"],
            marker="^",
            linewidth=2,
            markersize=8,
            label=method
        )

    plt.title("LPIPS Score vs Resolution (Lower = More Similar to Baseline)", 
              fontsize=14, fontweight='bold')
    plt.xlabel("Resolution", fontsize=12)
    plt.ylabel("LPIPS Distance", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUT_PLOTS_DIR, "lpips_score_comparison.png")
    plt.savefig(path, dpi=300)
    plt.close()

    print("✓ Saved:", path)


# ----------------------------------------------------------
# Plot 7: Speedup Factor
# ----------------------------------------------------------

def plot_speedup_factor(df):
    """Calculate and plot speedup relative to baseline."""

    # Get baseline times
    baseline_times = df[df['method'] == 'baseline'].groupby('resolution')['runtime_sec'].mean()
    
    if baseline_times.empty:
        print("⚠ No baseline data for speedup calculation")
        return

    # Calculate speedup for each method
    methods = df['method'].unique()
    methods = [m for m in methods if m != 'baseline']

    plt.figure(figsize=(10, 6))

    for method in methods:
        method_df = df[df['method'] == method]
        method_times = method_df.groupby('resolution')['runtime_sec'].mean()
        
        speedup = baseline_times / method_times
        
        plt.plot(
            speedup.index,
            speedup.values,
            marker="o",
            linewidth=2,
            markersize=8,
            label=method
        )

    plt.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Baseline (1x)')
    plt.title("Speedup Factor vs Resolution", fontsize=14, fontweight='bold')
    plt.xlabel("Resolution", fontsize=12)
    plt.ylabel("Speedup (×)", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUT_PLOTS_DIR, "speedup_factor.png")
    plt.savefig(path, dpi=300)
    plt.close()

    print("✓ Saved:", path)


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

def main():
    """Generate all plots."""

    print("=" * 60)
    print("GENERATING RESEARCH PLOTS")
    print("=" * 60)

    df = load_data()

    if df is None:
        return

    print(f"\nLoaded {len(df)} results")
    print(f"Methods: {df['method'].unique()}")
    print(f"Resolutions: {sorted(df['resolution'].unique())}")
    print()

    # Generate all plots
    plot_runtime_by_method(df)
    plot_memory_by_method(df)
    plot_window_size_ablation_runtime(df)
    plot_window_size_ablation_memory(df)
    plot_clip_scores(df)
    plot_lpips_scores(df)
    plot_speedup_factor(df)

    print("\n" + "=" * 60)
    print("ALL PLOTS GENERATED")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_PLOTS_DIR}")


# ----------------------------------------------------------

if __name__ == "__main__":
    main()
