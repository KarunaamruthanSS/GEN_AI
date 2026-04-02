# ==========================================================
# Attention Processor with Multiple Modes
# Supports: Window, Hybrid, and Slicing
#
# *** Fixed version ***
# - Uses attn.to_q / to_k / to_v / to_out projections
# - Multi-head compatible
# - Accepts **kwargs for diffusers pipeline compatibility
# ==========================================================

import torch
from diffusers.models.attention_processor import AttnProcessor

from src.window_attention.window_attention import WindowAttention


# ----------------------------------------------------------
# Window Attention Processor
# ----------------------------------------------------------

class WindowAttentionProcessor:
    """
    Custom attention processor that applies *windowed* self-attention
    while preserving the UNet's learned Q / K / V / Out projections.

    Cross-attention (encoder_hidden_states != None) falls back to the
    default full-attention processor.
    """

    def __init__(self, window_size=8):
        self.window_attention = WindowAttention(window_size=window_size)
        self.default_processor = AttnProcessor()

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        **kwargs,
    ):
        # --- CROSS ATTENTION → default full attention ---
        if encoder_hidden_states is not None:
            return self.default_processor(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                **kwargs,
            )

        # --- SELF ATTENTION → windowed attention ---

        batch, sequence, _channels = hidden_states.shape
        size = int(sequence ** 0.5)

        # Non-square sequence length — fallback to default
        if size * size != sequence:
            return self.default_processor(
                attn, hidden_states, None, attention_mask, **kwargs
            )

        # 1. Project through learned Q / K / V linear layers
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = query.shape[-1]

        # 2. Reshape to spatial: (B, seq, inner_dim) → (B, inner_dim, H, W)
        query = query.transpose(1, 2).reshape(batch, inner_dim, size, size)
        key = key.transpose(1, 2).reshape(batch, inner_dim, size, size)
        value = value.transpose(1, 2).reshape(batch, inner_dim, size, size)

        # 3. Windowed multi-head attention
        out = self.window_attention(query, key, value, attn.heads)

        # 4. Reshape back: (B, inner_dim, H, W) → (B, seq, inner_dim)
        out = out.reshape(batch, inner_dim, sequence).transpose(1, 2)

        # 5. Output projection + dropout
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)

        return out


# ----------------------------------------------------------
# Hybrid Attention Processor
# ----------------------------------------------------------

class HybridAttentionProcessor:
    """
    Hybrid processor: uses windowed self-attention for early UNet blocks,
    full attention for deeper blocks.

    Research motivation: Early blocks process low-level features that
    benefit from local context, while deeper blocks need global context
    for semantic understanding.
    """

    def __init__(self, window_size=8, use_window=True):
        self.use_window = use_window

        if use_window:
            self.window_attention = WindowAttention(window_size=window_size)

        self.default_processor = AttnProcessor()

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        **kwargs,
    ):
        # --- CROSS ATTENTION → always default ---
        if encoder_hidden_states is not None:
            return self.default_processor(
                attn, hidden_states, encoder_hidden_states,
                attention_mask, **kwargs
            )

        # --- SELF ATTENTION: full or windowed based on block depth ---
        if not self.use_window:
            return self.default_processor(
                attn, hidden_states, None, attention_mask, **kwargs
            )

        # Windowed path
        batch, sequence, _channels = hidden_states.shape
        size = int(sequence ** 0.5)

        if size * size != sequence:
            return self.default_processor(
                attn, hidden_states, None, attention_mask, **kwargs
            )

        # 1. Project
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = query.shape[-1]

        # 2. Spatial reshape
        query = query.transpose(1, 2).reshape(batch, inner_dim, size, size)
        key = key.transpose(1, 2).reshape(batch, inner_dim, size, size)
        value = value.transpose(1, 2).reshape(batch, inner_dim, size, size)

        # 3. Windowed attention
        out = self.window_attention(query, key, value, attn.heads)

        # 4. Flatten back
        out = out.reshape(batch, inner_dim, sequence).transpose(1, 2)

        # 5. Output projection + dropout
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)

        return out


# ----------------------------------------------------------
# Application Functions
# ----------------------------------------------------------

def apply_window_attention(unet, window_size=8):
    """
    Apply window attention to all self-attention layers in UNet.

    Args:
        unet: Diffusion UNet model
        window_size: Size of attention window
    """

    new_processors = {}

    for name, _processor in unet.attn_processors.items():
        new_processors[name] = WindowAttentionProcessor(window_size)

    unet.set_attn_processor(new_processors)

    print(f"Window attention (size={window_size}) successfully applied.")


def apply_hybrid_attention(unet, window_size=8, split_depth=2):
    """
    Apply hybrid attention: window attention for early blocks,
    full attention for deeper blocks.

    Args:
        unet: Diffusion UNet model
        window_size: Size of attention window
        split_depth: Block depth threshold (0-indexed)
    """

    new_processors = {}

    for name, _processor in unet.attn_processors.items():

        # Determine block depth
        if name.startswith("mid_block"):
            block_id = 999  # Always use full attention for mid block

        elif name.startswith("up_blocks"):
            block_id = int(name.split(".")[1])

        elif name.startswith("down_blocks"):
            block_id = int(name.split(".")[1])

        else:
            block_id = 0

        # Decide: window or full attention
        use_window = block_id < split_depth

        new_processors[name] = HybridAttentionProcessor(
            window_size,
            use_window,
        )

    unet.set_attn_processor(new_processors)

    print(f"Hybrid attention applied (window size={window_size}, split_depth={split_depth}).")


def apply_attention_slicing(unet):
    """
    Apply attention slicing (built-in Diffusers optimization).

    Research motivation: Reduces memory by computing attention in slices,
    trading off some speed for memory efficiency.
    """
    unet.enable_attention_slicing()
    print("Attention slicing enabled.")
