#!/usr/bin/env python
# ==========================================================
# Setup Verification Script
# Tests all components without running full experiments
# ==========================================================

import sys
import os

def test_imports():
    """Test all required imports."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
        print(f"    CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"    CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"  ✗ PyTorch: {e}")
        return False
    
    try:
        import diffusers
        print(f"  ✓ Diffusers {diffusers.__version__}")
    except ImportError as e:
        print(f"  ✗ Diffusers: {e}")
        return False
    
    try:
        import transformers
        print(f"  ✓ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"  ✗ Transformers: {e}")
        return False
    
    try:
        import lpips
        print(f"  ✓ LPIPS")
    except ImportError as e:
        print(f"  ✗ LPIPS: {e}")
        print(f"    Install with: pip install lpips")
        return False
    
    try:
        import gradio
        print(f"  ✓ Gradio {gradio.__version__}")
    except ImportError as e:
        print(f"  ✗ Gradio: {e}")
        print(f"    Install with: pip install gradio")
        return False
    
    try:
        import pandas
        print(f"  ✓ Pandas {pandas.__version__}")
    except ImportError as e:
        print(f"  ✗ Pandas: {e}")
        return False
    
    try:
        import matplotlib
        print(f"  ✓ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"  ✗ Matplotlib: {e}")
        return False
    
    return True


def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        from config import (
            DEVICE, MODEL_NAME, WINDOW_SIZES, RESOLUTIONS,
            ENABLE_ABLATION_STUDY, ENABLE_QUALITY_METRICS
        )
        print(f"  ✓ Config loaded")
        print(f"    Device: {DEVICE}")
        print(f"    Model: {MODEL_NAME}")
        print(f"    Window sizes: {WINDOW_SIZES}")
        print(f"    Resolutions: {RESOLUTIONS}")
        print(f"    Ablation study: {ENABLE_ABLATION_STUDY}")
        print(f"    Quality metrics: {ENABLE_QUALITY_METRICS}")
        return True
    except Exception as e:
        print(f"  ✗ Config error: {e}")
        return False


def test_modules():
    """Test custom modules."""
    print("\nTesting custom modules...")
    
    try:
        from src.window_attention.window_attention import WindowAttention
        print(f"  ✓ WindowAttention module")
    except Exception as e:
        print(f"  ✗ WindowAttention: {e}")
        return False
    
    try:
        from src.window_attention.attention_processor import (
            apply_window_attention,
            apply_hybrid_attention,
            apply_attention_slicing
        )
        print(f"  ✓ Attention processor module")
    except Exception as e:
        print(f"  ✗ Attention processor: {e}")
        return False
    
    try:
        from src.models.modified_pipeline import load_pipeline_with_mode
        print(f"  ✓ Pipeline module")
    except Exception as e:
        print(f"  ✗ Pipeline: {e}")
        return False
    
    try:
        from src.utils.metrics import compute_clip_score, compute_lpips_score
        print(f"  ✓ Metrics module")
    except Exception as e:
        print(f"  ✗ Metrics: {e}")
        return False
    
    return True


def test_directories():
    """Test directory structure."""
    print("\nTesting directories...")
    
    required_dirs = [
        "src/baseline",
        "src/window_attention",
        "src/models",
        "src/benchmarking",
        "src/utils",
        "outputs",
        "experiments"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path} (missing)")
            all_exist = False
    
    return all_exist


def test_window_attention():
    """Test window attention computation."""
    print("\nTesting window attention computation...")
    
    try:
        import torch
        from src.window_attention.window_attention import WindowAttention
        
        # Create test input
        x = torch.randn(1, 320, 64, 64)
        
        # Test different window sizes
        for ws in [4, 8, 16]:
            model = WindowAttention(320, ws)
            y = model(x)
            
            if y.shape == x.shape:
                print(f"  ✓ Window size {ws}: {x.shape} -> {y.shape}")
            else:
                print(f"  ✗ Window size {ws}: shape mismatch")
                return False
        
        return True
    
    except Exception as e:
        print(f"  ✗ Window attention test failed: {e}")
        return False


def main():
    """Run all tests."""
    
    print("=" * 60)
    print("SETUP VERIFICATION")
    print("=" * 60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Configuration", test_config()))
    results.append(("Modules", test_modules()))
    results.append(("Directories", test_directories()))
    results.append(("Window Attention", test_window_attention()))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✅ All tests passed! Setup is ready.")
        print("\nNext steps:")
        print("  1. Run experiments: python src/benchmarking/scaling_experiment.py")
        print("  2. Generate plots: python src/utils/plot_utils.py")
        print("  3. Launch UI: python app.py")
        print("  4. Or run all: python run_full_experiment.py")
    else:
        print("\n❌ Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  • Install missing packages: pip install -r requirements.txt")
        print("  • Check Python path and imports")
        print("  • Verify directory structure")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
