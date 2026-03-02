# ==========================================================
# Plot Experiment Results
# Generates Runtime and Memory Scaling Graphs
# ==========================================================

import os
import pandas as pd
import matplotlib.pyplot as plt

from config import RESULTS_CSV_PATH, OUTPUT_PLOTS_DIR


# ----------------------------------------------------------
# Create plots folder
# ----------------------------------------------------------

os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)


# ----------------------------------------------------------
# Load data
# ----------------------------------------------------------

def load_data():

    if not os.path.exists(RESULTS_CSV_PATH):
        print("Results file not found:", RESULTS_CSV_PATH)
        return None

    df = pd.read_csv(RESULTS_CSV_PATH)

    return df


# ----------------------------------------------------------
# Plot Runtime Scaling
# ----------------------------------------------------------

def plot_runtime(df):

    grouped = df.groupby(["method", "resolution"])["runtime_sec"].mean().reset_index()

    plt.figure()

    for method in grouped["method"].unique():

        subset = grouped[grouped["method"] == method]

        plt.plot(
            subset["resolution"],
            subset["runtime_sec"],
            marker="o",
            label=method
        )

    plt.title("Runtime vs Resolution")
    plt.xlabel("Resolution")
    plt.ylabel("Runtime (seconds)")
    plt.legend()
    plt.grid()

    path = os.path.join(OUTPUT_PLOTS_DIR, "runtime_vs_resolution.png")
    plt.savefig(path, dpi=300)
    plt.close()

    print("Saved:", path)


# ----------------------------------------------------------
# Plot Memory Scaling
# ----------------------------------------------------------

def plot_memory(df):

    grouped = df.groupby(["method", "resolution"])["memory_gb"].mean().reset_index()

    plt.figure()

    for method in grouped["method"].unique():

        subset = grouped[grouped["method"] == method]

        plt.plot(
            subset["resolution"],
            subset["memory_gb"],
            marker="o",
            label=method
        )

    plt.title("Memory Usage vs Resolution")
    plt.xlabel("Resolution")
    plt.ylabel("Memory (GB)")
    plt.legend()
    plt.grid()

    path = os.path.join(OUTPUT_PLOTS_DIR, "memory_vs_resolution.png")
    plt.savefig(path, dpi=300)
    plt.close()

    print("Saved:", path)


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

def main():

    df = load_data()

    if df is None:
        return

    plot_runtime(df)
    plot_memory(df)

    print("All plots generated.")


# ----------------------------------------------------------

if __name__ == "__main__":
    main()