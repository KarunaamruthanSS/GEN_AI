# Architecture Documentation

## System Overview

This document explains the technical architecture of the Window Attention research project, detailing how components interact and the research motivations behind design decisions.

## Core Components

### 1. Window Attention Module (`src/window_attention/window_attention.py`)

**Purpose**: Implements the core window attention mechanism.

**Algorithm**:
```
Input: Feature map x of shape (B, C, H, W)

1. Partition x into non-overlapping windows of size w×w
   - Reshape: (B, C, H, W) -> (B, C, H/w, w, W/w, w)
   - Rearrange: -> (B*num_windows, w*w, C)

2. Compute self-attention within each window independently
   - Q = K = V = window_features
   - Attention = softmax(QK^T / sqrt(C))
   - Output = Attention @ V

3. Merge windows back to original spatial layout
   - Reshape: (B*num_windows, w*w, C) -> (B, C, H, W)

Output: Attended features of shape (B, C, H, W)
```

**Complexity Analysis**:
- Standard attention: O(n²) where n = H×W
- Window attention: O(w²×(H/w)×(W/w)) = O(w²×n/w²) = O(n)
- Memory: Linear instead of quadratic

**Research Motivation**: 
- Local spatial coherence in images means nearby pixels are more relevant
- Global attention may be unnecessary for low-level features
- Reduces computational bottleneck for high-resolution generation

### 2. Attention Processor (`src/window_attention/attention_processor.py`)

**Purpose**: Integrates window attention into Stable Diffusion's UNet architecture.

**Key Classes**:

#### WindowAttentionProcessor
- Replaces standard attention in self-attention layers
- Preserves cross-attention (text conditioning) unchanged
- Handles spatial reshaping between sequence and 2D formats

#### HybridAttentionProcessor
- Applies different strategies at different network depths
- Early blocks (low-level features): Window attention
- Deep blocks (high-level semantics): Full attention
- Configurable split depth

**Research Motivation for Hybrid**:
```
UNet Architecture:
┌─────────────────────────────────────┐
│ Down Block 0 (64×64)  → Window      │  Low-level features
│ Down Block 1 (32×32)  → Window      │  (edges, textures)
├─────────────────────────────────────┤
│ Down Block 2 (16×16)  → Full        │  Mid-level features
│ Mid Block    (8×8)    → Full        │  (objects, composition)
├─────────────────────────────────────┤
│ Up Block 0   (16×16)  → Full        │  High-level semantics
│ Up Block 1   (32×32)  → Full        │  (global structure)
│ Up Block 2   (64×64)  → Full        │
└─────────────────────────────────────┘
```

Early blocks benefit from local attention (efficiency), while deep blocks need global context (quality).

### 3. Pipeline Manager (`src/models/modified_pipeline.py`)

**Purpose**: Unified interface for loading pipelines with different attention modes.

**Function**: `load_pipeline_with_mode(mode, window_size)`

**Modes**:
1. **baseline**: Standard Stable Diffusion (no modifications)
2. **window**: Window attention applied to all self-attention layers
3. **hybrid**: Mixed strategy (window + full attention)
4. **slicing**: Built-in Diffusers memory optimization

**Design Pattern**: Factory pattern for pipeline creation with caching support.

### 4. Metrics Module (`src/utils/metrics.py`)

**Purpose**: Compute quality metrics for generated images.

#### CLIP Score
```python
score = cosine_similarity(
    CLIP_text_encoder(prompt),
    CLIP_image_encoder(image)
)
```

**Interpretation**:
- Range: Typically 20-35 for good generations
- Higher = better text-image alignment
- Measures semantic similarity

**Research Use**: Validates that window attention doesn't degrade semantic quality.

#### LPIPS Score
```python
distance = LPIPS_model(
    preprocess(baseline_image),
    preprocess(modified_image)
)
```

**Interpretation**:
- Range: 0.0 (identical) to 1.0 (completely different)
- Lower = more perceptually similar
- Uses deep features from AlexNet

**Research Use**: Quantifies perceptual difference from baseline.

### 5. Benchmarking Suite (`src/benchmarking/scaling_experiment.py`)

**Purpose**: Comprehensive experimental framework.

**Workflow**:
```
For each method in [baseline, window, hybrid, slicing]:
    For each resolution in [512, 768, 1024]:
        For each prompt in test_prompts:
            1. Generate image
            2. Measure runtime
            3. Measure peak VRAM
            4. Compute CLIP score
            5. Compute LPIPS vs baseline
            6. Save image
            7. Log to CSV
```

**Ablation Study Extension**:
- When enabled, tests multiple window sizes [4, 8, 16]
- Adds window_size column to results
- Enables analysis of window size impact

**Design Decisions**:
- Baseline images cached for LPIPS comparison
- Fixed seed for reproducibility
- GPU memory cleared between runs
- Progressive execution (can resume if interrupted)

### 6. Visualization Suite (`src/utils/plot_utils.py`)

**Purpose**: Generate publication-ready figures.

**Plots Generated**:

1. **Runtime vs Resolution**: Shows scaling behavior
2. **Memory vs Resolution**: Validates memory efficiency
3. **Window Size Ablation (Runtime)**: Optimal window size analysis
4. **Window Size Ablation (Memory)**: Memory impact of window size
5. **CLIP Score Comparison**: Quality validation
6. **LPIPS Score Comparison**: Perceptual similarity analysis
7. **Speedup Factor**: Relative performance gains

**Design**: 
- 300 DPI for publication quality
- Consistent styling across plots
- Automatic grouping and aggregation
- Handles missing data gracefully

### 7. Interactive UI (`app.py`)

**Purpose**: Demo interface for real-time experimentation.

**Architecture**:
```
┌─────────────────────────────────────────┐
│           Gradio Interface              │
├─────────────────────────────────────────┤
│  Input Panel          Output Panel      │
│  ├─ Prompt            ├─ Generated Image│
│  ├─ Resolution        ├─ Metrics Display│
│  ├─ Attention Mode    └─ CLIP Score     │
│  ├─ Window Size                         │
│  └─ Advanced Settings                   │
├─────────────────────────────────────────┤
│         Pipeline Cache                  │
│  {mode_windowsize: pipeline}            │
└─────────────────────────────────────────┘
```

**Optimization**: Pipeline caching prevents reloading models on each generation.

## Data Flow

### Training/Inference Pipeline

```
User Prompt
    ↓
Text Encoder (CLIP)
    ↓
Text Embeddings
    ↓
┌─────────────────────────────────────┐
│         UNet Denoising              │
│                                     │
│  ┌─────────────────────────────┐   │
│  │  Self-Attention Layers      │   │
│  │  (Window/Hybrid/Slicing)    │   │
│  └─────────────────────────────┘   │
│                                     │
│  ┌─────────────────────────────┐   │
│  │  Cross-Attention Layers     │   │
│  │  (Standard - unchanged)     │   │
│  └─────────────────────────────┘   │
│                                     │
│  ┌─────────────────────────────┐   │
│  │  ResNet Blocks              │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
    ↓
Latent Representation
    ↓
VAE Decoder
    ↓
Generated Image
    ↓
┌─────────────────────────────────────┐
│      Quality Metrics                │
│  ├─ CLIP Score (text alignment)     │
│  └─ LPIPS Score (perceptual sim)    │
└─────────────────────────────────────┘
```

### Experimental Pipeline

```
Configuration (config.py)
    ↓
Experiment Runner (scaling_experiment.py)
    ↓
┌─────────────────────────────────────┐
│  For each configuration:            │
│    ├─ Load pipeline                 │
│    ├─ Generate images               │
│    ├─ Measure performance           │
│    ├─ Compute metrics               │
│    └─ Log results                   │
└─────────────────────────────────────┘
    ↓
results.csv
    ↓
Visualization (plot_utils.py)
    ↓
Publication Figures
```

## Configuration System

**Central Config** (`config.py`):
- Single source of truth for all parameters
- Imported by all modules
- Easy to modify for different experiments

**Key Configuration Groups**:

1. **Model Settings**: Model name, device, dtype
2. **Attention Settings**: Window sizes, modes, hybrid config
3. **Generation Settings**: Steps, guidance, seed
4. **Benchmark Settings**: Resolutions, methods, ablation flags
5. **Output Paths**: Organized directory structure

## Research Design Principles

### 1. Modularity
- Each component has single responsibility
- Easy to swap attention mechanisms
- Extensible for new methods

### 2. Reproducibility
- Fixed seeds for deterministic generation
- All parameters logged to CSV
- Version-controlled configuration

### 3. Efficiency
- Pipeline caching in UI
- GPU memory management
- Lazy loading of metric models

### 4. Extensibility
- Easy to add new attention modes
- Pluggable metric functions
- Configurable experiment parameters

### 5. Research Rigor
- Multiple quality metrics
- Comprehensive ablation studies
- Statistical aggregation across prompts
- Publication-ready visualizations

## Performance Characteristics

### Computational Complexity

| Method | Self-Attention | Cross-Attention | Total |
|--------|---------------|-----------------|-------|
| Baseline | O(n²) | O(n×m) | O(n²) |
| Window | O(n) | O(n×m) | O(n) |
| Hybrid | O(n) to O(n²) | O(n×m) | Mixed |
| Slicing | O(n²) | O(n×m) | O(n²) |

Where:
- n = spatial resolution (H×W)
- m = text sequence length
- w = window size

### Memory Characteristics

| Method | Attention Memory | Activation Memory |
|--------|-----------------|-------------------|
| Baseline | O(n²) | O(n) |
| Window | O(w²×num_windows) = O(n) | O(n) |
| Hybrid | Mixed | O(n) |
| Slicing | O(n²/slices) | O(n) |

## Extension Points

### Adding New Attention Mechanisms

1. Implement attention module in `src/window_attention/`
2. Create processor in `attention_processor.py`
3. Add mode to `load_pipeline_with_mode()`
4. Update `METHODS_TO_COMPARE` in config
5. Run experiments

### Adding New Metrics

1. Implement metric function in `src/utils/metrics.py`
2. Add to `compute_all_metrics()`
3. Update CSV headers in `scaling_experiment.py`
4. Add visualization in `plot_utils.py`

### Adding New Visualizations

1. Create plot function in `plot_utils.py`
2. Call from `main()`
3. Save to `OUTPUT_PLOTS_DIR`

## Testing Strategy

**Unit Tests**: `test_setup.py`
- Import verification
- Module loading
- Window attention computation
- Directory structure

**Integration Tests**: `run_full_experiment.py`
- End-to-end pipeline
- Data generation
- Visualization creation

**Manual Testing**: `app.py`
- Interactive validation
- Visual quality assessment
- Real-time metric computation

## Deployment Considerations

### Local Development
- CPU mode for testing (slow)
- GPU recommended for experiments
- ~8GB VRAM minimum for 512px
- ~12GB VRAM for 1024px

### Cloud/HPC
- Batch experiment execution
- Parallel prompt processing
- Result aggregation
- Automated plotting

### Production Demo
- Gradio UI with caching
- Model preloading
- Error handling
- Resource monitoring

## Future Architecture Improvements

1. **Distributed Training**: Multi-GPU support for faster experiments
2. **Checkpoint System**: Resume interrupted experiments
3. **Database Backend**: Replace CSV with proper database
4. **API Server**: REST API for programmatic access
5. **Experiment Tracking**: Integration with Weights & Biases
6. **Automated Hyperparameter Search**: Grid/random search for optimal configs

---

This architecture balances research flexibility with engineering rigor, enabling rapid experimentation while maintaining reproducibility and code quality.
