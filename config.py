# ==========================================================
# Project Configuration File
# Efficient Windowed Attention for Diffusion Models
# ==========================================================

import torch

# ----------------------------------------------------------
# DEVICE CONFIGURATION
# ----------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16  # Use float16 for memory efficiency


# ----------------------------------------------------------
# MODEL CONFIGURATION
# ----------------------------------------------------------

MODEL_NAME = "runwayml/stable-diffusion-v1-5"

# Safety checker off for faster inference (optional)
USE_SAFETY_CHECKER = False


# ----------------------------------------------------------
# WINDOW ATTENTION SETTINGS
# ----------------------------------------------------------

# Window sizes for ablation study
WINDOW_SIZES = [4, 8, 16]

# Default window size (used for single runs)
WINDOW_SIZE = 8

# Attention mode selection
# Options: "baseline", "window", "hybrid", "slicing"
ATTENTION_MODE = "window"

# Hybrid attention configuration
# Applies window attention to early blocks, full attention to deeper blocks
HYBRID_SPLIT_DEPTH = 2  # blocks 0-1 use window, rest use full attention


# ----------------------------------------------------------
# IMAGE RESOLUTION SETTINGS
# ----------------------------------------------------------

# Resolutions to test for scaling experiments
RESOLUTIONS = [512, 768, 1024]

# Default generation resolution
DEFAULT_RESOLUTION = 512


# ----------------------------------------------------------
# GENERATION SETTINGS
# ----------------------------------------------------------

NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 7.5
SEED = 42


# ----------------------------------------------------------
# BENCHMARK SETTINGS
# ----------------------------------------------------------

# Number of runs per resolution (for averaging time)
NUM_RUNS = 1  # Keep 1 for GPU cost control

# Enable ablation study (tests multiple window sizes)
ENABLE_ABLATION_STUDY = True

# Enable quality metrics (CLIP, LPIPS)
ENABLE_QUALITY_METRICS = True

# Methods to compare
METHODS_TO_COMPARE = ["baseline", "window", "slicing"]  # Add "hybrid" if needed


# ----------------------------------------------------------
# OUTPUT PATHS
# ----------------------------------------------------------

OUTPUT_BASELINE_DIR = "outputs/baseline_images/"
OUTPUT_WINDOW_DIR = "outputs/window_images/"
OUTPUT_PLOTS_DIR = "outputs/plots/"

RESULTS_CSV_PATH = "experiments/results.csv"


# ----------------------------------------------------------
# PROMPTS
# ----------------------------------------------------------

PROMPTS = [
    "A futuristic city at sunset",
    "A dragon flying over snowy mountains",
    "A hyper-realistic portrait of a humanoid robot",
]