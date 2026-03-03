# Experiment Guide

This guide provides step-by-step instructions for running specific experiments and interpreting results.

## Prerequisites

```bash
# Verify setup
python test_setup.py

# Should show all ✅ PASS
```

## Experiment 1: Basic Window Attention Comparison

**Goal**: Compare baseline vs window attention at 512px resolution.

**Configuration** (`config.py`):
```python
RESOLUTIONS = [512]
METHODS_TO_COMPARE = ["baseline", "window"]
ENABLE_ABLATION_STUDY = False
ENABLE_QUALITY_METRICS = True
WINDOW_SIZE = 8
```

**Run**:
```bash
python src/benchmarking/scaling_experiment.py
python src/utils/plot_utils.py
```

**Expected Results**:
- Runtime: ~20-25% speedup
- Memory: Similar usage
- CLIP: Comparable scores
- LPIPS: < 0.15 (high similarity)

**Interpretation**: Window attention provides modest speedup at low resolution with minimal quality loss.

---

## Experiment 2: Window Size Ablation Study

**Goal**: Find optimal window size for different resolutions.

**Configuration**:
```python
RESOLUTIONS = [512, 768, 1024]
METHODS_TO_COMPARE = ["window"]
ENABLE_ABLATION_STUDY = True  # Tests window sizes [4, 8, 16]
ENABLE_QUALITY_METRICS = True
```

**Run**:
```bash
python src/benchmarking/scaling_experiment.py
python src/utils/plot_utils.py
```

**Analysis**:
Look at `ablation_window_size_runtime.png`:
- Smaller windows (4): Faster but may lose global context
- Larger windows (16): Slower but better quality
- Optimal: Usually 8 for balance

**Research Question**: Does optimal window size vary with resolution?

---

## Experiment 3: Scaling Behavior Analysis

**Goal**: Understand how methods scale with resolution.

**Configuration**:
```python
RESOLUTIONS = [512, 768, 1024]
METHODS_TO_COMPARE = ["baseline", "window", "slicing"]
ENABLE_ABLATION_STUDY = False
ENABLE_QUALITY_METRICS = True
WINDOW_SIZE = 8
```

**Run**:
```bash
python src/benchmarking/scaling_experiment.py
python src/utils/plot_utils.py
```

**Analysis**:
Look at `speedup_factor.png`:
- Baseline: 1.0× (reference)
- Window: Should increase with resolution (1.2× → 1.5× → 2.0×)
- Slicing: Modest speedup, better memory

**Key Insight**: Window attention advantage grows with resolution due to O(n) vs O(n²) complexity.

---

## Experiment 4: Hybrid Attention Evaluation

**Goal**: Test hybrid strategy (window + full attention).

**Configuration**:
```python
RESOLUTIONS = [512, 768, 1024]
METHODS_TO_COMPARE = ["baseline", "window", "hybrid"]
ENABLE_ABLATION_STUDY = False
ENABLE_QUALITY_METRICS = True
WINDOW_SIZE = 8
HYBRID_SPLIT_DEPTH = 2  # Blocks 0-1 use window, rest use full
```

**Run**:
```bash
python src/benchmarking/scaling_experiment.py
python src/utils/plot_utils.py
```

**Analysis**:
Compare three methods:
- Baseline: Slowest, highest quality (reference)
- Window: Fastest, slight quality drop
- Hybrid: Middle ground (speed + quality)

**Research Question**: Does hybrid achieve better quality-efficiency trade-off?

---

## Experiment 5: Quality-Efficiency Trade-off

**Goal**: Comprehensive comparison across all methods.

**Configuration**:
```python
RESOLUTIONS = [512, 768, 1024]
METHODS_TO_COMPARE = ["baseline", "window", "hybrid", "slicing"]
ENABLE_ABLATION_STUDY = False
ENABLE_QUALITY_METRICS = True
WINDOW_SIZE = 8
```

**Run**:
```bash
python src/benchmarking/scaling_experiment.py
python src/utils/plot_utils.py
```

**Analysis Framework**:

1. **Runtime Efficiency**:
   - Check `runtime_vs_resolution_by_method.png`
   - Rank methods by speed

2. **Memory Efficiency**:
   - Check `memory_vs_resolution_by_method.png`
   - Identify memory-constrained scenarios

3. **Quality Preservation**:
   - Check `clip_score_comparison.png`
   - Verify scores within 5% of baseline

4. **Perceptual Similarity**:
   - Check `lpips_score_comparison.png`
   - Lower is better (< 0.2 acceptable)

**Decision Matrix**:
```
Scenario                    Recommended Method
─────────────────────────────────────────────
High resolution (1024+)     Window or Hybrid
Limited VRAM                Slicing
Quality critical            Baseline or Hybrid
Speed critical              Window
Balanced                    Hybrid
```

---

## Experiment 6: Prompt Sensitivity Analysis

**Goal**: Test if results generalize across different prompt types.

**Configuration**:
```python
# Add diverse prompts to config.py
PROMPTS = [
    "A futuristic city at sunset",           # Architecture
    "A dragon flying over mountains",        # Fantasy
    "A portrait of a robot",                 # Portrait
    "Abstract colorful patterns",            # Abstract
    "A photorealistic landscape",            # Realism
    "A cartoon character",                   # Stylized
]

RESOLUTIONS = [512]
METHODS_TO_COMPARE = ["baseline", "window"]
ENABLE_QUALITY_METRICS = True
```

**Run**:
```bash
python src/benchmarking/scaling_experiment.py
```

**Analysis**:
```python
import pandas as pd

df = pd.read_csv('experiments/results.csv')

# Group by prompt and method
grouped = df.groupby(['prompt', 'method']).agg({
    'runtime_sec': 'mean',
    'clip_score': 'mean',
    'lpips_score': 'mean'
})

print(grouped)
```

**Research Question**: Are certain prompt types more sensitive to window attention?

---

## Experiment 7: Interactive Quality Assessment

**Goal**: Visual comparison and user study preparation.

**Run**:
```bash
python app.py
```

**Protocol**:
1. Generate same prompt with different methods
2. Save all outputs
3. Compare visually side-by-side
4. Rate on scale 1-5 for:
   - Overall quality
   - Detail preservation
   - Prompt adherence
   - Artifacts

**Blind Test Setup**:
- Rename images to remove method labels
- Randomize order
- Have multiple raters score
- Compute inter-rater agreement

---

## Experiment 8: Extreme Resolution Testing

**Goal**: Test limits of window attention at very high resolutions.

**Configuration**:
```python
RESOLUTIONS = [512, 1024, 1536, 2048]  # If VRAM allows
METHODS_TO_COMPARE = ["window", "slicing"]  # Skip baseline (too slow)
ENABLE_QUALITY_METRICS = False  # Speed up
WINDOW_SIZE = 8
```

**Run**:
```bash
python src/benchmarking/scaling_experiment.py
```

**Analysis**:
- Plot runtime vs resolution (should be linear for window)
- Check memory scaling
- Identify VRAM limits

**Research Question**: At what resolution does window attention become essential?

---

## Experiment 9: Hybrid Configuration Sweep

**Goal**: Find optimal hybrid split depth.

**Manual Configuration**:
```python
# Test different split depths
for split_depth in [1, 2, 3, 4]:
    # Update config.py
    HYBRID_SPLIT_DEPTH = split_depth
    
    # Run experiment
    # Save results with split_depth label
```

**Analysis**:
- Plot runtime vs split_depth
- Plot CLIP score vs split_depth
- Find Pareto optimal point

---

## Experiment 10: Publication-Ready Results

**Goal**: Generate complete dataset for research paper.

**Configuration**:
```python
RESOLUTIONS = [512, 768, 1024]
METHODS_TO_COMPARE = ["baseline", "window", "hybrid", "slicing"]
ENABLE_ABLATION_STUDY = True
ENABLE_QUALITY_METRICS = True
PROMPTS = [
    # Diverse set of 10+ prompts
]
NUM_RUNS = 3  # For statistical significance
```

**Run**:
```bash
python run_full_experiment.py
```

**Post-Processing**:
```python
import pandas as pd
import numpy as np

df = pd.read_csv('experiments/results.csv')

# Compute statistics
stats = df.groupby(['method', 'resolution']).agg({
    'runtime_sec': ['mean', 'std'],
    'memory_gb': ['mean', 'std'],
    'clip_score': ['mean', 'std'],
    'lpips_score': ['mean', 'std']
})

# Export for LaTeX table
stats.to_latex('results_table.tex')
```

**Deliverables**:
- Complete results CSV
- All visualization plots
- Statistical summary table
- Generated image gallery

---

## Interpreting Results

### Runtime Analysis

**Good Results**:
- Window attention: 20-50% faster than baseline
- Speedup increases with resolution
- Consistent across prompts

**Red Flags**:
- Window slower than baseline (implementation issue)
- High variance across runs (GPU throttling?)
- No scaling benefit (check window size)

### Memory Analysis

**Expected**:
- Similar memory across methods (dominated by model weights)
- Slight reduction with slicing
- Linear scaling with resolution

**Issues**:
- OOM errors: Reduce resolution or use slicing
- Unexpected spikes: Check for memory leaks

### Quality Analysis

**CLIP Score**:
- Baseline: 25-30 (typical range)
- Window: Within 5% of baseline (acceptable)
- Large drop (>10%): Quality degradation

**LPIPS Score**:
- < 0.10: Visually identical
- 0.10-0.20: Minor differences
- 0.20-0.30: Noticeable differences
- > 0.30: Significant degradation

### Statistical Significance

For publication, compute:
```python
from scipy import stats

baseline_times = df[df['method']=='baseline']['runtime_sec']
window_times = df[df['method']=='window']['runtime_sec']

t_stat, p_value = stats.ttest_ind(baseline_times, window_times)

if p_value < 0.05:
    print("Speedup is statistically significant")
```

---

## Troubleshooting

### Experiment Fails

**CUDA OOM**:
- Reduce resolution
- Enable attention slicing
- Reduce batch size (if modified)

**Slow Generation**:
- Check GPU utilization: `nvidia-smi`
- Verify CUDA is being used
- Reduce num_inference_steps for testing

**Quality Metrics Fail**:
- CLIP/LPIPS models may need download
- Check internet connection
- Disable metrics for faster testing

### Unexpected Results

**No Speedup**:
- Verify window attention is applied: Check logs
- Test with larger resolution (512 may be too small)
- Profile with `torch.profiler`

**Quality Degradation**:
- Try larger window size
- Use hybrid mode
- Check if cross-attention is preserved

**High Variance**:
- Increase NUM_RUNS
- Fix random seed
- Check GPU temperature/throttling

---

## Next Steps

After running experiments:

1. **Analyze Results**: Review CSV and plots
2. **Visual Inspection**: Compare generated images
3. **Statistical Tests**: Verify significance
4. **Write Report**: Document findings
5. **Share Results**: Publish or present

## Citation

When publishing results from these experiments:

```bibtex
@misc{window-attention-experiments,
  title={Window Attention for Efficient Diffusion Models: Experimental Results},
  author={Your Name},
  year={2024},
  note={Experiments conducted using efficient-window-attention framework}
}
```
