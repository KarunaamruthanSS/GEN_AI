import sys
import os

# add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pandas as pd
from src.utils.plot_utils import plot_runtime, plot_memory

RESULTS_FILE = "experiments/results.csv"

def main():

    if not os.path.exists(RESULTS_FILE):
        print("Results file not found")
        return

    df = pd.read_csv(RESULTS_FILE)

    plot_runtime(df)
    plot_memory(df)

    print("Plots generated successfully.")


if __name__ == "__main__":
    main()
