# ==========================================================
# Streamlit UI for Window Attention Research
# Interactive demo with performance metrics and quality scores
#
# Usage (rented GPU with public IP):
#   streamlit run app.py --server.headless true --server.address 0.0.0.0 --server.port 8501
#
# Then visit http://<PUBLIC_IP>:8501 from your local browser.
# ==========================================================

import sys
import os
import gc
import time

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import streamlit as st
import torch
from PIL import Image

from config import (
    DEVICE,
    NUM_INFERENCE_STEPS,
    GUIDANCE_SCALE,
    SEED,
    WINDOW_SIZES,
)

from src.models.modified_pipeline import load_pipeline_with_mode
from src.utils.metrics import compute_clip_score


# ----------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------

st.set_page_config(
    page_title="Window Attention Research Demo",
    page_icon="🔬",
    layout="wide",
)


# ----------------------------------------------------------
# Custom CSS for dark, premium look
# ----------------------------------------------------------

st.markdown("""
<style>
    /* Dark gradient header */
    .main-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2.2rem;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #b0b0d0;
        font-size: 1.05rem;
    }

    /* Metrics cards */
    .metric-card {
        background: linear-gradient(145deg, #1e1e2f, #2a2a40);
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid #3a3a5a;
        margin-bottom: 0.8rem;
    }
    .metric-card h4 {
        color: #a78bfa;
        margin: 0 0 0.4rem 0;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-card .value {
        color: #e0e0ff;
        font-size: 1.6rem;
        font-weight: 700;
    }

    /* Info panel */
    .info-panel {
        background: #1a1a2e;
        border: 1px solid #3a3a5a;
        border-radius: 10px;
        padding: 1.2rem;
        color: #c0c0e0;
        font-family: monospace;
        font-size: 0.92rem;
        line-height: 1.6;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 100%);
    }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------
# Pipeline Management (cached in session state)
# ----------------------------------------------------------

def get_pipeline(mode, window_size):
    """
    Get or load pipeline. Evicts previous pipeline from GPU memory
    before loading a new one to prevent CUDA OOM.
    """

    cache_key = f"{mode}_{window_size}"

    if "pipeline_cache_key" not in st.session_state:
        st.session_state.pipeline_cache_key = None
        st.session_state.pipeline = None

    if cache_key != st.session_state.pipeline_cache_key:
        # Evict old pipeline
        if st.session_state.pipeline is not None:
            st.toast("♻️ Unloading previous pipeline to free VRAM...")
            del st.session_state.pipeline
            st.session_state.pipeline = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        st.toast(f"📦 Loading pipeline: **{mode}** (window_size={window_size})")

        # move_to_device=False — cpu_offload manages device placement
        pipe = load_pipeline_with_mode(mode, window_size, move_to_device=False)

        # Offloads each submodule to GPU only during its forward pass
        pipe.enable_model_cpu_offload()

        # Decode VAE in slices to reduce spike at final step
        pipe.enable_vae_slicing()

        # xformers for extra memory + speed savings if installed
        if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass  # not installed — safe to skip

        st.session_state.pipeline = pipe
        st.session_state.pipeline_cache_key = cache_key

    return st.session_state.pipeline


# ----------------------------------------------------------
# Image Generation
# ----------------------------------------------------------

def generate_image(prompt, resolution, window_size, attention_mode,
                   num_steps, guidance_scale, seed):
    """Generate image with selected parameters and return metrics."""

    if DEVICE == "cpu":
        return None, {"error": "⚠️ GPU required for image generation."}

    if not prompt or prompt.strip() == "":
        return None, {"error": "⚠️ Please enter a prompt."}

    pipe = get_pipeline(attention_mode, window_size)

    # Generator on CPU — enable_model_cpu_offload manages device placement
    generator = torch.Generator(device="cpu").manual_seed(int(seed))

    # Clear memory and track
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()

    image = pipe(
        prompt,
        height=resolution,
        width=resolution,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

    end_time = time.time()

    runtime = end_time - start_time
    memory = (torch.cuda.max_memory_allocated() / (1024 ** 3)
              if torch.cuda.is_available() else 0.0)

    # CLIP Score
    clip_score = None
    try:
        clip_score = compute_clip_score(image, prompt)
    except Exception as e:
        print(f"CLIP score error: {e}")

    metrics = {
        "mode": attention_mode,
        "resolution": resolution,
        "window_size": window_size,
        "steps": num_steps,
        "guidance": guidance_scale,
        "seed": seed,
        "runtime": runtime,
        "memory": memory,
        "clip_score": clip_score,
    }

    return image, metrics


# ----------------------------------------------------------
# Sidebar: Generation Settings
# ----------------------------------------------------------

with st.sidebar:

    st.markdown("## 🎨 Generation Settings")

    prompt = st.text_area(
        "Prompt",
        value="A futuristic city at sunset",
        height=100,
        placeholder="Describe the image you want to generate...",
    )

    st.markdown("---")

    resolution = st.selectbox(
        "Resolution",
        options=[512, 768, 1024],
        index=0,
        help="Higher resolution = more computation",
    )

    attention_mode = st.radio(
        "Attention Mode",
        options=["baseline", "window", "hybrid", "slicing"],
        index=1,
        help="Select attention mechanism to test",
    )

    window_size = st.select_slider(
        "Window Size",
        options=[4, 8, 16],
        value=8,
        help="Only applies to window/hybrid modes",
    )

    st.markdown("---")

    with st.expander("⚙️ Advanced Settings", expanded=False):
        num_steps = st.slider(
            "Inference Steps",
            min_value=10, max_value=50, step=5,
            value=NUM_INFERENCE_STEPS,
            help="More steps = better quality but slower",
        )

        guidance = st.slider(
            "Guidance Scale",
            min_value=1.0, max_value=15.0, step=0.5,
            value=GUIDANCE_SCALE,
            help="Higher = stronger prompt adherence",
        )

        seed = st.number_input(
            "Seed",
            value=SEED,
            step=1,
            help="For reproducible results",
        )

    generate_clicked = st.button("🚀 Generate Image", type="primary", use_container_width=True)

    st.markdown("---")

    st.markdown("""
    ### 💡 Example Prompts
    Click to copy into the prompt box above.
    """)

    example_prompts = [
        "A futuristic city at sunset",
        "A dragon flying over snowy mountains",
        "A hyper-realistic portrait of a humanoid robot",
        "An astronaut riding a horse on Mars",
        "A serene Japanese garden with cherry blossoms",
    ]

    for ex in example_prompts:
        if st.button(f"📝 {ex}", key=f"ex_{ex}", use_container_width=True):
            st.session_state["prompt_override"] = ex
            st.rerun()

    st.markdown("---")
    st.markdown(f"""
    ### 🔧 System Info
    - **Device:** `{DEVICE}`
    - **Model:** Stable Diffusion v1.5
    - **Precision:** FP16
    """)


# ----------------------------------------------------------
# Handle example prompt override
# ----------------------------------------------------------

if "prompt_override" in st.session_state:
    prompt = st.session_state.pop("prompt_override")


# ----------------------------------------------------------
# Main Area: Header
# ----------------------------------------------------------

st.markdown("""
<div class="main-header">
    <h1>🔬 Window Attention for Stable Diffusion</h1>
    <p>Research Demo · Efficient Attention Mechanisms · Compare Baseline vs Window vs Hybrid vs Slicing</p>
</div>
""", unsafe_allow_html=True)


# ----------------------------------------------------------
# Main Area: Research Notes
# ----------------------------------------------------------

with st.expander("📖 Research Notes", expanded=False):
    st.markdown("""
    **Window Attention** reduces computational complexity by processing
    attention in localized spatial windows instead of globally.

    | Mode | Complexity | Description |
    |------|-----------|-------------|
    | **Baseline** | O(n²) | Standard full attention |
    | **Window** | O(w²) per window | Localized window attention |
    | **Hybrid** | Mixed | Window in early blocks, full in deep blocks |
    | **Slicing** | O(n²) sliced | Memory-efficient attention slicing |

    **Expected Results:**
    - Faster inference at higher resolutions
    - Similar or slightly lower memory usage
    - Comparable image quality (measured by CLIP score)
    """)


# ----------------------------------------------------------
# Main Area: Generation Result
# ----------------------------------------------------------

if generate_clicked:

    if DEVICE == "cpu":
        st.error("⚠️ GPU required for image generation. Please run on a GPU-enabled machine.")
    elif not prompt or prompt.strip() == "":
        st.warning("⚠️ Please enter a prompt.")
    else:
        with st.spinner("🎨 Generating image... This may take a moment."):
            try:
                image, metrics = generate_image(
                    prompt, resolution, window_size,
                    attention_mode, num_steps, guidance, int(seed)
                )
            except Exception as e:
                st.error(f"❌ Error during generation:\n{str(e)}")
                image, metrics = None, None

        if image is not None and metrics is not None:

            col_img, col_metrics = st.columns([3, 2])

            with col_img:
                st.markdown("### 🖼️ Generated Image")
                st.image(image, use_container_width=True)

            with col_metrics:
                st.markdown("### 📊 Performance Metrics")

                # Metric cards
                m1, m2 = st.columns(2)
                with m1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>⏱️ Runtime</h4>
                        <div class="value">{metrics['runtime']:.2f}s</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>💾 Peak VRAM</h4>
                        <div class="value">{metrics['memory']:.2f} GB</div>
                    </div>
                    """, unsafe_allow_html=True)

                if metrics.get("clip_score") is not None:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>🎯 CLIP Score</h4>
                        <div class="value">{metrics['clip_score']:.4f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.caption("Higher CLIP score = better text-image alignment")

                st.markdown("---")
                st.markdown("#### ⚙️ Configuration")

                config_lines = [
                    f"- **Mode:** {metrics['mode']}",
                    f"- **Resolution:** {metrics['resolution']}×{metrics['resolution']}",
                ]
                if metrics["mode"] in ["window", "hybrid"]:
                    config_lines.append(f"- **Window Size:** {metrics['window_size']}")
                config_lines += [
                    f"- **Steps:** {metrics['steps']}",
                    f"- **Guidance:** {metrics['guidance']}",
                    f"- **Seed:** {metrics['seed']}",
                ]

                st.markdown("\n".join(config_lines))

else:
    # Placeholder when no image generated yet
    st.info(
        "👈 Configure your settings in the sidebar and click **🚀 Generate Image** to begin."
    )


# ----------------------------------------------------------
# Footer
# ----------------------------------------------------------

st.markdown("---")
st.caption(
    "Window Attention Research Demo · Stable Diffusion v1.5 · "
    f"Running on **{DEVICE}**"
)
