# ==========================================================
# Attention Processor with Multiple Modes
# Supports: Window, Hybrid, and Slicing
# ==========================================================

import torch
from diffusers.models.attention_processor import AttnProcessor

from src.window_attention.window_attention import WindowAttention


class WindowAttentionProcessor:
    """
    Custom attention processor that applies window attention to self-attention.
    Cross-attention uses default processor.
    """

    def __init__(self, hidden_size, window_size=8):

        self.window_attention = WindowAttention(
            channels=hidden_size,
            window_size=window_size
        )

        # default processor for fallback
        self.default_processor = AttnProcessor()


    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        **kwargs
    ):

        # CROSS ATTENTION → use default processor
        if encoder_hidden_states is not None:

            return self.default_processor(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                **kwargs
            )


        # SELF ATTENTION → apply window attention

        batch, sequence, channels = hidden_states.shape

        size = int(sequence ** 0.5)

        if size * size != sequence:
            return hidden_states


        x = hidden_states.transpose(1, 2)
        x = x.reshape(batch, channels, size, size)

        x = self.window_attention(x)

        x = x.reshape(batch, channels, sequence)
        x = x.transpose(1, 2)

        return x


class HybridAttentionProcessor:
    """
    Hybrid processor: uses window attention for early blocks,
    full attention for deeper blocks.
    
    Research motivation: Early blocks process low-level features
    that benefit from local context, while deeper blocks need
    global context for semantic understanding.
    """

    def __init__(self, hidden_size, window_size=8, use_window=True):
        
        self.use_window = use_window
        
        if use_window:
            self.window_attention = WindowAttention(
                channels=hidden_size,
                window_size=window_size
            )
        
        self.default_processor = AttnProcessor()


    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        **kwargs
    ):

        # CROSS ATTENTION → always use default
        if encoder_hidden_states is not None:
            return self.default_processor(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                **kwargs
            )

        # SELF ATTENTION → window or full based on block depth
        if not self.use_window:
            return self.default_processor(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                **kwargs
            )

        # Apply window attention
        batch, sequence, channels = hidden_states.shape
        size = int(sequence ** 0.5)

        if size * size != sequence:
            return hidden_states

        x = hidden_states.transpose(1, 2)
        x = x.reshape(batch, channels, size, size)
        x = self.window_attention(x)
        x = x.reshape(batch, channels, sequence)
        x = x.transpose(1, 2)

        return x


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

    for name, processor in unet.attn_processors.items():

        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]

        elif name.startswith("up_blocks"):
            block_id = int(name.split(".")[1])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]

        elif name.startswith("down_blocks"):
            block_id = int(name.split(".")[1])
            hidden_size = unet.config.block_out_channels[block_id]

        else:
            hidden_size = unet.config.block_out_channels[0]

        new_processors[name] = WindowAttentionProcessor(hidden_size, window_size)

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

    for name, processor in unet.attn_processors.items():

        # Determine hidden size
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
            block_id = 999  # Always use full attention for mid block

        elif name.startswith("up_blocks"):
            block_id = int(name.split(".")[1])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]

        elif name.startswith("down_blocks"):
            block_id = int(name.split(".")[1])
            hidden_size = unet.config.block_out_channels[block_id]

        else:
            hidden_size = unet.config.block_out_channels[0]
            block_id = 0

        # Decide: window or full attention
        use_window = block_id < split_depth

        new_processors[name] = HybridAttentionProcessor(
            hidden_size,
            window_size,
            use_window
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

