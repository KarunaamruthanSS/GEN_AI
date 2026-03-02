# ==========================================================
# Window Attention (projection-free version)
# Works with Stable Diffusion attention processor
# ==========================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class WindowAttention(nn.Module):

    def __init__(self, channels, window_size=8):
        super().__init__()

        self.channels = channels
        self.window_size = window_size


    def forward(self, x):
        """
        x shape: (B, C, H, W)
        """

        B, C, H, W = x.shape
        ws = self.window_size

        # ensure divisible
        if H % ws != 0 or W % ws != 0:
            return x

        # split windows
        x_windows = x.view(
            B,
            C,
            H // ws,
            ws,
            W // ws,
            ws
        )

        x_windows = x_windows.permute(0, 2, 4, 3, 5, 1)
        x_windows = x_windows.reshape(-1, ws * ws, C)

        # self attention inside window
        q = x_windows
        k = x_windows
        v = x_windows

        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = attn / (C ** 0.5)
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)

        # merge windows back
        out = out.view(
            B,
            H // ws,
            W // ws,
            ws,
            ws,
            C
        )

        out = out.permute(0, 5, 1, 3, 2, 4)
        out = out.reshape(B, C, H, W)

        return out


# test
if __name__ == "__main__":

    x = torch.randn(1, 320, 64, 64)

    model = WindowAttention(320, 8)

    y = model(x)

    print(x.shape, y.shape)
