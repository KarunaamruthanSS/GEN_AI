import torch
from diffusers.models.attention_processor import AttnProcessor

from src.window_attention.window_attention import WindowAttention
from config import WINDOW_SIZE


class WindowAttentionProcessor:

    def __init__(self, hidden_size):

        self.window_attention = WindowAttention(
            channels=hidden_size,
            window_size=WINDOW_SIZE
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


def apply_window_attention(unet):

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


        new_processors[name] = WindowAttentionProcessor(hidden_size)


    unet.set_attn_processor(new_processors)

    print("Window attention successfully applied.")
