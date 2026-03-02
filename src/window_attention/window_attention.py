# ==========================================================
# Window Attention Module
# Efficient Windowed Attention for Diffusion Models
# ==========================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class WindowAttention(nn.Module):
    """
    Local window self-attention for 2D feature maps.

    Input shape:
        (B, C, H, W)

    Output shape:
        (B, C, H, W)
    """

    def __init__(self, channels, window_size=8):
        super().__init__()

        self.channels = channels
        self.window_size = window_size

        # Query, Key, Value projections
        self.to_q = nn.Conv2d(channels, channels, 1)
        self.to_k = nn.Conv2d(channels, channels, 1)
        self.to_v = nn.Conv2d(channels, channels, 1)

        # Output projection
        self.to_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        """
        x: (B, C, H, W)
        """

        B, C, H, W = x.shape
        ws = self.window_size

        assert H % ws == 0 and W % ws == 0, \
            "Height and Width must be divisible by window size"

        # Project Q K V
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # Split into windows
        q = self._split_windows(q)
        k = self._split_windows(k)
        v = self._split_windows(v)

        # Compute attention per window
        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = attn / (C ** 0.5)
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)

        # Merge windows
        out = self._merge_windows(out, B, C, H, W)

        out = self.to_out(out)

        return out

    def _split_windows(self, x):
        """
        Convert (B,C,H,W) -> (num_windows*B, ws*ws, C)
        """

        B, C, H, W = x.shape
        ws = self.window_size

        x = x.view(
            B,
            C,
            H // ws,
            ws,
            W // ws,
            ws
        )

        x = x.permute(0, 2, 4, 3, 5, 1)

        x = x.reshape(-1, ws * ws, C)

        return x

    def _merge_windows(self, x, B, C, H, W):
        """
        Convert back to (B,C,H,W)
        """

        ws = self.window_size

        x = x.view(
            B,
            H // ws,
            W // ws,
            ws,
            ws,
            C
        )

        x = x.permute(0, 5, 1, 3, 2, 4)

        x = x.reshape(B, C, H, W)

        return x


# ==========================================================
# Simple test (CPU safe)
# ==========================================================

if __name__ == "__main__":

    print("Testing Window Attention...")

    x = torch.randn(1, 320, 64, 64)

    model = WindowAttention(
        channels=320,
        window_size=8
    )

    out = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", out.shape)

    print("Success!")