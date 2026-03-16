# ==========================================================
# Advanced Gradio UI for Window Attention Research
# Interactive demo with performance metrics and quality scores
# ==========================================================

import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import gradio as gr
import torch
import time
from PIL import Image

from config import (
    DEVICE,
    NUM_INFERENCE_STEPS,
    GUIDANCE_SCALE,
    SEED,
    WINDOW_SIZES
)

from src.models.modified_pipeline import load_pipeline_with_mode
from src.utils.metrics import compute_clip_score

# Global pipeline cache — only one pipeline kept in memory at a time
# to avoid CUDA OOM when switching modes
_pipeline_cache = {}
_current_cache_key = None


# ----------------------------------------------------------
# Pipeline Management
# ----------------------------------------------------------

def get_pipeline(mode, window_size):
    """
    Get or load pipeline. Evicts previous pipeline from GPU memory
    before loading a new one to prevent CUDA OOM.

    Uses enable_model_cpu_offload() instead of .to(DEVICE) so that
    submodules are moved to GPU one at a time during the forward pass,
    keeping peak VRAM around 4-5 GB instead of 10+ GB.
    """
    global _pipeline_cache, _current_cache_key

    cache_key = f"{mode}_{window_size}"

    if cache_key != _current_cache_key:
        # Evict old pipeline from GPU before loading new one
        if _current_cache_key and _current_cache_key in _pipeline_cache:
            print("Unloading previous pipeline to free VRAM...")
            old_pipe = _pipeline_cache.pop(_current_cache_key)
            del old_pipe
            torch.cuda.empty_cache()

        print(f"Loading pipeline: {mode} (window_size={window_size})")

        # move_to_device=False — cpu_offload manages device placement
        pipe = load_pipeline_with_mode(mode, window_size, move_to_device=False)

        # Offloads each submodule to GPU only during its forward pass.
        # This is the key fix: cuts peak VRAM from ~10 GB to ~4-5 GB.
        pipe.enable_model_cpu_offload()

        # Decode VAE in slices to avoid a large spike at the final step
        pipe.enable_vae_slicing()

        # xformers gives extra memory + speed savings if installed
        if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            try:
                pipe.enable_xformers_memory_efficient_attention()
                print("xformers memory efficient attention enabled.")
            except Exception:
                pass  # not installed — safe to skip

        _pipeline_cache[cache_key] = pipe
        _current_cache_key = cache_key

    return _pipeline_cache[cache_key]


# ----------------------------------------------------------
# Image Generation with Metrics
# ----------------------------------------------------------

def generate_image(
    prompt,
    resolution,
    window_size,
    attention_mode,
    num_steps,
    guidance_scale,
    seed
):
    """
    Generate image with selected parameters and return metrics.
    
    Returns:
        image, info_text
    """
    
    if DEVICE == "cpu":
        return None, "⚠️ GPU required for image generation. Please run on a GPU-enabled machine."
    
    if not prompt or prompt.strip() == "":
        return None, "⚠️ Please enter a prompt."
    
    try:
        # Get pipeline
        pipe = get_pipeline(attention_mode, window_size)
        
        # Setup generator — use CPU since model_cpu_offload manages device placement
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        
        # Clear memory and track
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Generate
        start_time = time.time()
        
        image = pipe(
            prompt,
            height=resolution,
            width=resolution,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]
        
        end_time = time.time()
        
        # Metrics
        runtime = end_time - start_time
        memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
        
        # CLIP Score
        try:
            clip_score = compute_clip_score(image, prompt)
        except Exception as e:
            print(f"CLIP score error: {e}")
            clip_score = None
        
        # Format info
        info_lines = [
            f"✅ Generation Complete",
            f"",
            f"⚙️ Configuration:",
            f"  • Mode: {attention_mode}",
            f"  • Resolution: {resolution}×{resolution}",
        ]
        
        if attention_mode in ["window", "hybrid"]:
            info_lines.append(f"  • Window Size: {window_size}")
        
        info_lines.extend([
            f"  • Steps: {num_steps}",
            f"  • Guidance: {guidance_scale}",
            f"  • Seed: {seed}",
            f"",
            f"📊 Performance:",
            f"  • Runtime: {runtime:.2f} seconds",
            f"  • Peak VRAM: {memory:.2f} GB",
        ])
        
        if clip_score is not None:
            info_lines.extend([
                f"",
                f"🎯 Quality:",
                f"  • CLIP Score: {clip_score:.4f}",
                f"    (Higher = better text-image alignment)"
            ])
        
        info_text = "\n".join(info_lines)
        
        return image, info_text
    
    except Exception as e:
        error_msg = f"❌ Error during generation:\n{str(e)}"
        print(error_msg)
        return None, error_msg


# ----------------------------------------------------------
# Gradio Interface
# ----------------------------------------------------------

def create_ui():
    """Create advanced Gradio interface."""
    
    with gr.Blocks(title="Window Attention Research Demo") as demo:        
        gr.Markdown("""
        # 🔬 Window Attention for Stable Diffusion
        ### Research Demo: Efficient Attention Mechanisms
        
        Compare different attention strategies for diffusion models:
        - **Baseline**: Standard full attention (O(n²) complexity)
        - **Window**: Localized window attention (O(w²) complexity per window)
        - **Hybrid**: Window attention in early blocks, full attention in deep blocks
        - **Slicing**: Memory-efficient attention slicing
        """)
        
        with gr.Row():
            
            # Left column: Inputs
            with gr.Column(scale=1):
                
                gr.Markdown("### 🎨 Generation Settings")
                
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="A futuristic city at sunset...",
                    lines=3,
                    value="A futuristic city at sunset"
                )
                
                resolution_input = gr.Dropdown(
                    label="Resolution",
                    choices=[512, 768, 1024],
                    value=512,
                    info="Higher resolution = more computation"
                )
                
                attention_mode_input = gr.Radio(
                    label="Attention Mode",
                    choices=["baseline", "window", "hybrid", "slicing"],
                    value="window",
                    info="Select attention mechanism to test"
                )
                
                window_size_input = gr.Slider(
                    label="Window Size",
                    minimum=4,
                    maximum=16,
                    step=4,
                    value=8,
                    info="Only applies to window/hybrid modes"
                )
                
                with gr.Accordion("Advanced Settings", open=False):
                    
                    num_steps_input = gr.Slider(
                        label="Inference Steps",
                        minimum=10,
                        maximum=50,
                        step=5,
                        value=NUM_INFERENCE_STEPS,
                        info="More steps = better quality but slower"
                    )
                    
                    guidance_input = gr.Slider(
                        label="Guidance Scale",
                        minimum=1.0,
                        maximum=15.0,
                        step=0.5,
                        value=GUIDANCE_SCALE,
                        info="Higher = stronger prompt adherence"
                    )
                    
                    seed_input = gr.Number(
                        label="Seed",
                        value=SEED,
                        precision=0,
                        info="For reproducible results"
                    )
                
                generate_btn = gr.Button("🚀 Generate Image", variant="primary", size="lg")
                
                gr.Markdown("""
                ---
                ### 📖 Research Notes
                
                **Window Attention** reduces computational complexity by processing 
                attention in localized spatial windows instead of globally.
                
                **Expected Results:**
                - Faster inference at higher resolutions
                - Similar or slightly lower memory usage
                - Comparable image quality (measured by CLIP score)
                
                **Hybrid Mode** balances efficiency and quality by using window 
                attention for low-level features and full attention for high-level semantics.
                """)
            
            # Right column: Outputs
            with gr.Column(scale=1):
                
                gr.Markdown("### 🖼️ Generated Image")
                
                image_output = gr.Image(
                    label="Output",
                    type="pil",
                    height=512
                )
                
                info_output = gr.Textbox(
                    label="📊 Metrics & Info",
                    lines=20,
                    max_lines=25,
                    interactive=False
                )
        
        # Examples
        gr.Markdown("### 💡 Example Prompts")
        
        gr.Examples(
            examples=[
                ["A futuristic city at sunset", 512, 8, "window"],
                ["A dragon flying over snowy mountains", 768, 8, "window"],
                ["A hyper-realistic portrait of a humanoid robot", 512, 16, "hybrid"],
                ["An astronaut riding a horse on Mars", 512, 8, "baseline"],
                ["A serene Japanese garden with cherry blossoms", 768, 4, "window"],
            ],
            inputs=[prompt_input, resolution_input, window_size_input, attention_mode_input],
            label="Click to load example"
        )
        
        # Connect button
        generate_btn.click(
            fn=generate_image,
            inputs=[
                prompt_input,
                resolution_input,
                window_size_input,
                attention_mode_input,
                num_steps_input,
                guidance_input,
                seed_input
            ],
            outputs=[image_output, info_output]
        )
        
        gr.Markdown("""
        ---
        ### 🔧 System Info
        - Device: {}
        - Model: Stable Diffusion v1.5
        - Precision: FP16
        """.format(DEVICE))
    
    return demo


# ----------------------------------------------------------
# Launch
# ----------------------------------------------------------

if __name__ == "__main__":
    
    print("=" * 60)
    print("WINDOW ATTENTION RESEARCH DEMO")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    
    if DEVICE == "cpu":
        print("\n⚠️  WARNING: Running on CPU")
        print("Image generation will be very slow.")
        print("For best experience, use a GPU-enabled machine.\n")
    
    demo = create_ui()
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
        theme=gr.themes.Soft()
    )
