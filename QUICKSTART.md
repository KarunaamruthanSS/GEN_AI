# Quick Start Guide

Get up and running in 5 minutes.

## Installation

```bash
# Clone or navigate to project
cd efficient-window-attention

# Create virtual environment
python -m venv .venv

# Activate (Linux/Mac)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify setup
python test_setup.py
```

## Quick Test (CPU - No GPU Required)

```bash
# Test imports and modules
python test_setup.py

# Should show all ✅ PASS
```

## Run Experiments (GPU Required)

### Option 1: Minimal Test (Fastest)

```bash
# Edit config.py:
# RESOLUTIONS = [512]
# METHODS_TO_COMPARE = ["baseline", "window"]
# ENABLE_ABLATION_STUDY = False
# ENABLE_QUALITY_METRICS = False

python src/benchmarking/scaling_experiment.py
```

**Time**: ~5 minutes  
**Output**: Basic runtime/memory comparison

### Option 2: With Quality Metrics

```bash
# Edit config.py:
# ENABLE_QUALITY_METRICS = True

python src/benchmarking/scaling_experiment.py
python src/utils/plot_utils.py
```

**Time**: ~10 minutes  
**Output**: Runtime, memory, CLIP, LPIPS + plots

### Option 3: Full Experiment

```bash
# Edit config.py:
# RESOLUTIONS = [512, 768, 1024]
# METHODS_TO_COMPARE = ["baseline", "window", "slicing"]
# ENABLE_ABLATION_STUDY = True
# ENABLE_QUALITY_METRICS = True

python run_full_experiment.py
```

**Time**: ~30-60 minutes  
**Output**: Complete research dataset

## Launch Interactive UI

```bash
python app.py

# Open browser to: http://localhost:7860
```

## View Results

```bash
# Results CSV
cat experiments/results.csv

# Or open in Excel/Pandas
python -c "import pandas as pd; print(pd.read_csv('experiments/results.csv'))"

# View plots
# Open outputs/plots/ directory
```

## Common Commands

```bash
# Verify setup
python test_setup.py

# Run experiments
python src/benchmarking/scaling_experiment.py

# Generate plots
python src/utils/plot_utils.py

# Launch UI
python app.py

# Complete pipeline
python run_full_experiment.py
```

## Configuration Quick Reference

Edit `config.py`:

```python
# Test single resolution
RESOLUTIONS = [512]

# Test all resolutions
RESOLUTIONS = [512, 768, 1024]

# Compare methods
METHODS_TO_COMPARE = ["baseline", "window"]
METHODS_TO_COMPARE = ["baseline", "window", "hybrid", "slicing"]

# Enable ablation (tests window sizes 4, 8, 16)
ENABLE_ABLATION_STUDY = True

# Enable quality metrics (CLIP, LPIPS)
ENABLE_QUALITY_METRICS = True

# Change window size
WINDOW_SIZE = 8  # Try 4, 8, or 16

# Change attention mode (for single runs)
ATTENTION_MODE = "window"  # baseline, window, hybrid, slicing
```

## Troubleshooting

### CUDA Out of Memory
```python
# Reduce resolution
RESOLUTIONS = [512]

# Or use slicing
METHODS_TO_COMPARE = ["slicing"]
```

### Slow on CPU
```
⚠️ This project requires GPU for reasonable performance.
CPU mode works but is very slow (10-20x slower).
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Verify
python test_setup.py
```

### Missing CLIP/LPIPS Models
```
First run will download models (~500MB).
Requires internet connection.
```

## File Locations

```
Results:     experiments/results.csv
Images:      outputs/baseline_images/
             outputs/window_images/
Plots:       outputs/plots/
Config:      config.py
```

## Next Steps

1. ✅ Run `test_setup.py`
2. ✅ Edit `config.py` for your experiment
3. ✅ Run experiment
4. ✅ Generate plots
5. ✅ Analyze results
6. 📖 Read `EXPERIMENTS.md` for detailed protocols
7. 📖 Read `ARCHITECTURE.md` for technical details

## Support

- Documentation: See `README.md`
- Experiments: See `EXPERIMENTS.md`
- Architecture: See `ARCHITECTURE.md`
- Upgrades: See `UPGRADE_SUMMARY.md`

---

**Ready to start!** 🚀

Run: `python test_setup.py`
