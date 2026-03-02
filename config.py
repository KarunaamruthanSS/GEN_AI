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

# Window size in latent space (start small like 8 or 16)
WINDOW_SIZE = 8

# You can experiment later with:
# WINDOW_SIZE = 16


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