# ==========================================================
# Custom Attention Processor for Stable Diffusion
# Integrates Window Attention into Diffusers UNet
# ==========================================================

import torch
import torch.nn as nn
from diffusers.models.attention_processor import AttnProcessor

from src.window_attention.window_attention import WindowAttention
from config import WINDOW_SIZE


class WindowAttentionProcessor(AttnProcessor):
    """
    Custom attention processor that replaces self-attention
    with window-based attention.

    Only applied when encoder_hidden_states is None
    (self-attention, not cross-attention).
    """

    def __init__(self, hidden_size):
        super().__init__()

        self.window_attention = WindowAttention(
            channels=hidden_size,
            window_size=WINDOW_SIZE
        )

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        **kwargs
    ):
        """
        hidden_states shape:
        (batch, sequence, channels)
        """

        # --------------------------------------------------
        # If cross-attention, use default implementation
        # --------------------------------------------------

        if encoder_hidden_states is not None:
            return attn.processor(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                **kwargs
            )

        # --------------------------------------------------
        # Convert sequence to feature map
        # --------------------------------------------------

        batch, sequence, channels = hidden_states.shape

        height = width = int(sequence ** 0.5)

        if height * width != sequence:
            # fallback to default if shape not square
            return attn.processor(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                **kwargs
            )

        x = hidden_states.transpose(1, 2)
        x = x.reshape(batch, channels, height, width)

        # --------------------------------------------------
        # Apply window attention
        # --------------------------------------------------

        x = self.window_attention(x)

        # --------------------------------------------------
        # Convert back to sequence format
        # --------------------------------------------------

        x = x.reshape(batch, channels, sequence)
        x = x.transpose(1, 2)

        return x


# ==========================================================
# Utility function to apply processor to UNet
# ==========================================================

def apply_window_attention(unet):
    """
    Replace self-attention processors in UNet
    """

    processors = {}

    for name, module in unet.named_modules():

        if hasattr(module, "set_processor"):
            hidden_size = module.to_q.in_features

            processors[name] = WindowAttentionProcessor(
                hidden_size
            )

    unet.set_attn_processor(processors)

    print("Window attention successfully applied.")