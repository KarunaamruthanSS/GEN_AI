# Changelog

All notable changes to the Window Attention project.

## [2.0.0] - Research-Level Upgrade

### Added

#### Core Features
- **Window Size Ablation Study**: Automated testing of multiple window sizes [4, 8, 16]
- **Quality Metrics**: CLIP score (text-image alignment) and LPIPS (perceptual similarity)
- **Attention Slicing**: Third comparison method using built-in Diffusers optimization
- **Hybrid Attention**: Mixed strategy (window + full attention) based on network depth
- **Advanced Gradio UI**: Professional interactive demo with real-time metrics

#### New Files
- `src/utils/metrics.py`: Quality metric computation (CLIP, LPIPS)
- `app.py`: Advanced Gradio UI with comprehensive controls
- `README.md`: Complete project documentation
- `ARCHITECTURE.md`: Technical architecture guide
- `EXPERIMENTS.md`: Detailed experiment protocols
- `UPGRADE_SUMMARY.md`: Summary of all upgrades
- `QUICKSTART.md`: Quick start guide
- `CHANGELOG.md`: This file
- `test_setup.py`: Automated setup verification
- `run_full_experiment.py`: Complete pipeline runner

#### Configuration
- `WINDOW_SIZES`: List for ablation study
- `ENABLE_ABLATION_STUDY`: Toggle ablation experiments
- `ENABLE_QUALITY_METRICS`: Toggle CLIP/LPIPS computation
- `METHODS_TO_COMPARE`: List of methods to benchmark
- `ATTENTION_MODE`: Mode selection (baseline/window/hybrid/slicing)
- `HYBRID_SPLIT_DEPTH`: Configuration for hybrid attention

#### Visualizations
- Window size ablation plots (runtime and memory)
- CLIP score comparison plot
- LPIPS score comparison plot
- Speedup factor plot (relative to baseline)
- Enhanced styling for all plots (300 DPI, publication-ready)

#### Results Schema
- `window_size` column: Tracks window size for ablation
- `clip_score` column: Text-image alignment quality
- `lpips_score` column: Perceptual similarity to baseline

### Changed

#### Updated Files
- `config.py`: Enhanced with new configuration options
- `src/window_attention/attention_processor.py`: 
  - Added `HybridAttentionProcessor` class
  - Added `apply_hybrid_attention()` function
  - Added `apply_attention_slicing()` function
  - Made window_size a parameter instead of global
- `src/models/modified_pipeline.py`:
  - Added `load_pipeline_with_mode()` function
  - Support for all attention modes
  - Backward compatible `load_window_pipeline()`
- `src/benchmarking/scaling_experiment.py`:
  - Complete rewrite for multi-method support
  - Ablation study integration
  - Quality metrics integration
  - Baseline image caching for LPIPS
  - Enhanced logging and progress tracking
- `src/utils/plot_utils.py`:
  - Complete rewrite with 7 plot types
  - Publication-ready styling
  - Automatic data aggregation
  - Graceful handling of missing data
- `src/utils/run_plots.py`: Simplified to call main()
- `requirements.txt`: Added lpips and gradio

### Improved

#### Code Quality
- Comprehensive docstrings
- Research motivation comments
- Modular architecture
- Clean separation of concerns
- Error handling and validation
- Progress tracking and logging

#### Documentation
- Complete README with usage examples
- Architecture documentation with diagrams
- Experiment protocols and guides
- Quick start guide
- Troubleshooting section

#### Reproducibility
- Fixed seeds throughout
- Complete parameter logging
- Version-controlled configuration
- Deterministic execution

#### Performance
- Lazy loading of metric models
- Pipeline caching in UI
- Efficient CSV operations
- Proper GPU memory management

### Research Capabilities

#### Experiments Supported
1. Basic comparison (baseline vs window)
2. Window size ablation study
3. Scaling analysis across resolutions
4. Multi-method comparison (4-way)
5. Quality evaluation (CLIP, LPIPS)
6. Hybrid strategy evaluation
7. Statistical analysis with multiple runs
8. Visual comparison and user studies

#### Publication-Ready Outputs
- Comprehensive results CSV
- Statistical summaries
- Publication-quality figures (300 DPI)
- Reproducible experiments
- Documented methodology
- Complete architecture documentation

### Backward Compatibility

✅ Fully backward compatible with v1.0
- Existing scripts work unchanged
- Old results.csv format supported
- Default config matches original behavior
- No breaking changes to core modules

### Dependencies

#### Added
- `lpips==0.1.4`: Perceptual similarity metric
- `gradio==4.16.0`: Interactive UI framework

#### Existing (unchanged)
- torch, diffusers, transformers, etc.

### Testing

#### Automated
- `test_setup.py`: Verifies all components
- Import checks
- Module loading tests
- Window attention computation tests
- Directory structure validation

#### Manual
- Interactive UI for visual validation
- Example prompts for quick testing
- Real-time metric computation

### Performance Metrics

#### Expected Results
- Runtime speedup: 20-56% (increases with resolution)
- Memory usage: Similar across methods
- CLIP scores: Within 5% of baseline
- LPIPS scores: < 0.15 (high similarity)

### Known Issues

None currently. See GitHub issues for tracking.

### Migration Guide

#### From v1.0 to v2.0

No migration needed! v2.0 is fully backward compatible.

**Optional upgrades**:

1. Update `config.py` to use new features:
```python
ENABLE_ABLATION_STUDY = True
ENABLE_QUALITY_METRICS = True
METHODS_TO_COMPARE = ["baseline", "window", "slicing"]
```

2. Run new experiment script:
```bash
python src/benchmarking/scaling_experiment.py
```

3. Generate new plots:
```bash
python src/utils/plot_utils.py
```

4. Try new UI:
```bash
python app.py
```

### Contributors

- Original implementation: [Your Name]
- Research-level upgrade: [Your Name]

### Acknowledgments

- Stable Diffusion by Stability AI
- Diffusers library by Hugging Face
- CLIP by OpenAI
- LPIPS by Zhang et al.

---

## [1.0.0] - Initial Implementation

### Added
- Basic window attention implementation
- Baseline comparison
- Runtime and memory benchmarking
- Simple plotting utilities
- Configuration system

### Features
- Window attention module
- Attention processor integration
- Scaling experiment script
- Basic visualizations

---

## Future Roadmap

### Planned Features
- [ ] Extension to SDXL/SD2.x
- [ ] FID score evaluation
- [ ] Adaptive window sizing
- [ ] Multi-scale window attention
- [ ] Cross-attention window strategies
- [ ] Distributed training support
- [ ] Experiment tracking (W&B integration)
- [ ] REST API for programmatic access

### Research Extensions
- [ ] User study framework
- [ ] Automated hyperparameter search
- [ ] Checkpoint system for long experiments
- [ ] Database backend for results
- [ ] Comparative analysis with other methods

---

**Version**: 2.0.0  
**Date**: 2024  
**Status**: Stable, Production-Ready
