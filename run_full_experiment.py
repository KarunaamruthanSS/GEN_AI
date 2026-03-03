#!/usr/bin/env python
# ==========================================================
# Complete Experiment Pipeline
# Runs benchmarking and generates all visualizations
# ==========================================================

import sys
import os
import subprocess

def run_command(cmd, description):
    """Run a command and handle errors."""
    print("\n" + "=" * 60)
    print(f"STEP: {description}")
    print("=" * 60)
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n❌ Error in: {description}")
        return False
    
    print(f"\n✅ Completed: {description}")
    return True


def main():
    """Run complete experimental pipeline."""
    
    print("=" * 60)
    print("COMPLETE EXPERIMENTAL PIPELINE")
    print("=" * 60)
    print("\nThis script will:")
    print("1. Run scaling experiments (baseline, window, slicing)")
    print("2. Generate all visualizations")
    print("3. Display summary statistics")
    print("\n⚠️  WARNING: This requires GPU and may take 30-60 minutes")
    print("=" * 60)
    
    response = input("\nContinue? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Step 1: Run experiments
    success = run_command(
        "python src/benchmarking/scaling_experiment.py",
        "Running Scaling Experiments"
    )
    
    if not success:
        print("\n❌ Experiment failed. Check GPU availability and dependencies.")
        return
    
    # Step 2: Generate plots
    success = run_command(
        "python src/utils/plot_utils.py",
        "Generating Visualizations"
    )
    
    if not success:
        print("\n⚠️  Plotting failed, but experiment data is saved.")
    
    # Step 3: Display summary
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    
    print("\n📁 Output Locations:")
    print(f"  • Results CSV: experiments/results.csv")
    print(f"  • Baseline Images: outputs/baseline_images/")
    print(f"  • Window Images: outputs/window_images/")
    print(f"  • Plots: outputs/plots/")
    
    print("\n📊 Next Steps:")
    print("  1. Review results.csv for detailed metrics")
    print("  2. Check plots/ directory for visualizations")
    print("  3. Launch UI: python app.py")
    print("  4. Compare generated images visually")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
