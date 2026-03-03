# Project Structure

Complete overview of the upgraded Window Attention research project.

## Directory Tree

```
efficient-window-attention/
│
├── 📄 config.py                          # Central configuration
├── 📄 requirements.txt                   # Python dependencies
├── 📄 app.py                            # Gradio UI (NEW)
│
├── 📚 Documentation/
│   ├── README.md                        # Main documentation (NEW)
│   ├── QUICKSTART.md                    # Quick start guide (NEW)
│   ├── ARCHITECTURE.md                  # Technical architecture (NEW)
│   ├── EXPERIMENTS.md                   # Experiment protocols (NEW)
│   ├── UPGRADE_SUMMARY.md               # Upgrade details (NEW)
│   ├── CHANGELOG.md                     # Version history (NEW)
│   └── PROJECT_STRUCTURE.md             # This file (NEW)
│
├── 🧪 Testing & Utilities/
│   ├── test_setup.py                    # Setup verification (NEW)
│   └── run_full_experiment.py           # Complete pipeline (NEW)
│
├── 📁 src/
│   │
│   ├── 📁 baseline/
│   │   ├── generate_baseline.py         # Baseline generation
│   │   └── __init__.py
│   │
│   ├── 📁 window_attention/
│   │   ├── window_attention.py          # Core window attention
│   │   ├── attention_processor.py       # Multi-mode processor (UPGRADED)
│   │   └── __init__.py
│   │
│   ├── 📁 models/
│   │   ├── modified_pipeline.py         # Pipeline loader (UPGRADED)
│   │   └── __init__.py
│   │
│   ├── 📁 benchmarking/
│   │   ├── scaling_experiment.py        # Main experiment suite (UPGRADED)
│   │   ├── benchmark_memory.py          # (placeholder)
│   │   ├── benchmark_runtime.py         # (placeholder)
│   │   └── __init__.py
│   │
│   ├── 📁 utils/
│   │   ├── metrics.py                   # Quality metrics (NEW)
│   │   ├── plot_utils.py                # Advanced plotting (UPGRADED)
│   │   ├── run_plots.py                 # Plot runner (UPDATED)
│   │   ├── memory_utils.py              # (placeholder)
│   │   ├── time_utils.py                # (placeholder)
│   │   └── __init__.py
│   │
│   └── __init__.py
│
├── 📁 outputs/
│   ├── 📁 baseline_images/              # Baseline outputs
│   ├── 📁 window_images/                # Modified method outputs
│   └── 📁 plots/                        # Generated visualizations
│       ├── runtime_vs_resolution_by_method.png
│       ├── memory_vs_resolution_by_method.png
│       ├── ablation_window_size_runtime.png
│       ├── ablation_window_size_memory.png
│       ├── clip_score_comparison.png
│       ├── lpips_score_comparison.png
│       └── speedup_factor.png
│
├── 📁 experiments/
│   ├── results.csv                      # Benchmark results (UPGRADED SCHEMA)
│   ├── prompts.txt                      # Test prompts
│   └── logs/                            # Experiment logs
│
└── 📁 .venv/                            # Virtual environment
```

## File Descriptions

### Core Configuration

#### `config.py`
Central configuration file for all experiments.

**Key Sections**:
- Device configuration (CUDA/CPU)
- Model settings (SD 1.5)
- Window attention settings (sizes, modes)
- Generation parameters (steps, guidance)
- Benchmark settings (resolutions, methods)
- Output paths

**New in v2.0**:
- `WINDOW_SIZES` for ablation
- `ENABLE_ABLATION_STUDY` flag
- `ENABLE_QUALITY_METRICS` flag
- `METHODS_TO_COMPARE` list
- `ATTENTION_MODE` selection
- `HYBRID_SPLIT_DEPTH` parameter

#### `requirements.txt`
Python package dependencies.

**New in v2.0**:
- `lpips==0.1.4`
- `gradio==4.16.0`

---

### Source Code (`src/`)

#### `src/window_attention/window_attention.py`
Core window attention implementation.

**Key Class**: `WindowAttention`
- Partitions feature maps into windows
- Computes self-attention within windows
- Merges windows back to spatial layout

**Complexity**: O(n) vs O(n²) for standard attention

#### `src/window_attention/attention_processor.py` ⭐ UPGRADED
Multi-mode attention processor.

**Key Classes**:
- `WindowAttentionProcessor`: Window attention for self-attention
- `HybridAttentionProcessor`: Mixed window/full strategy (NEW)

**Key Functions**:
- `apply_window_attention()`: Apply to all layers
- `apply_hybrid_attention()`: Apply hybrid strategy (NEW)
- `apply_attention_slicing()`: Enable slicing (NEW)

#### `src/models/modified_pipeline.py` ⭐ UPGRADED
Unified pipeline loader.

**Key Functions**:
- `load_baseline_pipeline()`: Standard SD
- `load_pipeline_with_mode()`: Mode-based loading (NEW)
- `load_window_pipeline()`: Legacy compatibility

**Modes**: baseline, window, hybrid, slicing

#### `src/benchmarking/scaling_experiment.py` ⭐ UPGRADED
Comprehensive benchmarking suite.

**Features**:
- Multi-method comparison
- Ablation study support
- Quality metrics integration
- Baseline image caching
- Progress tracking
- CSV logging

**Measures**:
- Runtime (seconds)
- Peak VRAM (GB)
- CLIP score (text-image alignment)
- LPIPS score (perceptual similarity)

#### `src/utils/metrics.py` ⭐ NEW
Quality metric computation.

**Functions**:
- `compute_clip_score()`: Text-image alignment
- `compute_lpips_score()`: Perceptual similarity
- `compute_all_metrics()`: Batch computation

**Models Used**:
- CLIP: `openai/clip-vit-base-patch32`
- LPIPS: AlexNet-based

#### `src/utils/plot_utils.py` ⭐ UPGRADED
Advanced visualization suite.

**Plots Generated** (7 total):
1. Runtime vs resolution (by method)
2. Memory vs resolution (by method)
3. Window size ablation (runtime)
4. Window size ablation (memory)
5. CLIP score comparison
6. LPIPS score comparison
7. Speedup factor

**Features**:
- Publication-ready (300 DPI)
- Automatic aggregation
- Graceful error handling
- Consistent styling

---

### User Interface

#### `app.py` ⭐ NEW
Professional Gradio UI for interactive demos.

**Features**:
- Prompt input with examples
- Resolution selector (512/768/1024)
- Window size slider (4-16)
- Attention mode selector
- Advanced settings (steps, guidance, seed)
- Real-time metrics display
- Pipeline caching
- Professional styling

**Use Cases**:
- Interactive demos
- User studies
- Rapid prototyping
- Presentations

---

### Documentation

#### `README.md` ⭐ NEW
Main project documentation.

**Sections**:
- Project overview
- Installation instructions
- Quick start guide
- Feature descriptions
- Configuration reference
- Expected results
- Research extensions

#### `QUICKSTART.md` ⭐ NEW
5-minute quick start guide.

**Contents**:
- Installation steps
- Quick test commands
- Common configurations
- Troubleshooting

#### `ARCHITECTURE.md` ⭐ NEW
Technical architecture documentation.

**Contents**:
- System overview
- Component descriptions
- Data flow diagrams
- Design principles
- Performance characteristics
- Extension points

#### `EXPERIMENTS.md` ⭐ NEW
Detailed experiment protocols.

**Contents**:
- 10 experiment templates
- Analysis frameworks
- Interpretation guides
- Statistical methods
- Troubleshooting

#### `UPGRADE_SUMMARY.md` ⭐ NEW
Summary of v2.0 upgrades.

**Contents**:
- Feature-by-feature comparison
- New files created
- Configuration changes
- Workflow improvements
- Migration guide

#### `CHANGELOG.md` ⭐ NEW
Version history and changes.

**Contents**:
- Version 2.0.0 changes
- Version 1.0.0 baseline
- Future roadmap
- Migration guides

---

### Testing & Utilities

#### `test_setup.py` ⭐ NEW
Automated setup verification.

**Tests**:
- Import checks
- Configuration loading
- Module functionality
- Directory structure
- Window attention computation

**Usage**: `python test_setup.py`

#### `run_full_experiment.py` ⭐ NEW
Complete experimental pipeline.

**Steps**:
1. Run scaling experiments
2. Generate visualizations
3. Display summary

**Usage**: `python run_full_experiment.py`

---

### Outputs

#### `outputs/baseline_images/`
Generated images from baseline method.

**Naming**: `baseline_{resolution}_{prompt_idx}.png`

#### `outputs/window_images/`
Generated images from modified methods.

**Naming**: 
- `window_{window_size}_{resolution}_{prompt_idx}.png`
- `hybrid_{resolution}_{prompt_idx}.png`
- `slicing_{resolution}_{prompt_idx}.png`

#### `outputs/plots/`
Publication-ready visualizations (300 DPI).

**Files** (7 plots):
- `runtime_vs_resolution_by_method.png`
- `memory_vs_resolution_by_method.png`
- `ablation_window_size_runtime.png`
- `ablation_window_size_memory.png`
- `clip_score_comparison.png`
- `lpips_score_comparison.png`
- `speedup_factor.png`

---

### Experiments

#### `experiments/results.csv` ⭐ UPGRADED
Comprehensive benchmark results.

**Schema**:
```
method, window_size, resolution, prompt, runtime_sec, memory_gb, clip_score, lpips_score
```

**New Columns**:
- `window_size`: For ablation study
- `clip_score`: Text-image alignment
- `lpips_score`: Perceptual similarity

---

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    Configuration                        │
│                     (config.py)                         │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Experiment Runner                          │
│         (scaling_experiment.py)                         │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │  For each method:                               │   │
│  │    For each resolution:                         │   │
│  │      For each prompt:                           │   │
│  │        1. Load pipeline                         │   │
│  │        2. Generate image                        │   │
│  │        3. Measure performance                   │   │
│  │        4. Compute metrics                       │   │
│  │        5. Save results                          │   │
│  └─────────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  Results CSV                            │
│            (experiments/results.csv)                    │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Visualization                              │
│              (plot_utils.py)                            │
│                                                         │
│  Generates 7 publication-ready plots                    │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                Research Outputs                         │
│  • Plots (outputs/plots/)                               │
│  • Images (outputs/*/images/)                           │
│  • Data (experiments/results.csv)                       │
└─────────────────────────────────────────────────────────┘
```

---

## Module Dependencies

```
config.py
    ↓
    ├─→ src/window_attention/window_attention.py
    │       ↓
    │   src/window_attention/attention_processor.py
    │       ↓
    │   src/models/modified_pipeline.py
    │       ↓
    ├─→ src/benchmarking/scaling_experiment.py
    │       ↓
    │   src/utils/metrics.py
    │       ↓
    │   experiments/results.csv
    │       ↓
    │   src/utils/plot_utils.py
    │       ↓
    │   outputs/plots/
    │
    └─→ app.py (Gradio UI)
```

---

## Key Metrics

### Code Statistics
- **Total Files**: 30+ (including docs)
- **Python Modules**: 15
- **Documentation Files**: 7
- **Lines of Code**: ~3000+
- **Test Coverage**: Core modules tested

### Research Capabilities
- **Attention Methods**: 4 (baseline, window, hybrid, slicing)
- **Quality Metrics**: 4 (runtime, memory, CLIP, LPIPS)
- **Visualizations**: 7 publication-ready plots
- **Experiment Types**: 10+ protocols documented

---

## Version Information

- **Current Version**: 2.0.0
- **Previous Version**: 1.0.0
- **Status**: Stable, Production-Ready
- **Python**: 3.8+
- **PyTorch**: 2.1.2
- **Diffusers**: 0.25.0

---

## Quick Reference

### Most Important Files

1. `config.py` - Configure experiments
2. `src/benchmarking/scaling_experiment.py` - Run experiments
3. `src/utils/plot_utils.py` - Generate plots
4. `app.py` - Interactive UI
5. `README.md` - Main documentation

### Most Common Commands

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

---

**Project Status**: ✅ Ready for Research Publication
