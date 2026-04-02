# ==========================================================
# Window Attention (multi-head, projection-aware version)
# Accepts pre-projected Q, K, V from the UNet's attention layers
# ==========================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class WindowAttention(nn.Module):
    """
    Windowed self-attention module.

    Accepts Q, K, V tensors that have *already* been projected by
    the UNet's ``attn.to_q / to_k / to_v`` linear layers.  The module
    partitions the spatial dimensions into non-overlapping windows of
    size ``window_size x window_size`` and computes multi-head scaled
    dot-product attention within each window independently.

    Input shapes:  (B, C, H, W)   where C = num_heads * head_dim
    Output shape:  (B, C, H, W)
    """

    def __init__(self, window_size=8):
        super().__init__()
        self.window_size = window_size

    # ----------------------------------------------------------

    def forward(self, q, k, v, num_heads):
        """
        Args:
            q, k, v : (B, C, H, W)  — pre-projected query / key / value
            num_heads: int           — number of attention heads

        Returns:
            out : (B, C, H, W)
        """

        B, C, H, W = q.shape
        ws = self.window_size
        head_dim = C // num_heads

        # Fallback to identity if spatial dims are not divisible by ws
        if H % ws != 0 or W % ws != 0:
            return v

        nH = H // ws  # number of windows along height
        nW = W // ws  # number of windows along width

        # ----- partition into windows -----
        # (B, C, H, W)
        #   -> (B, num_heads, head_dim, nH, ws, nW, ws)
        #   -> (B, num_heads, nH, nW, ws, ws, head_dim)
        #   -> (B * num_heads * nH * nW, ws*ws, head_dim)

        def window_partition(x):
            x = x.reshape(B, num_heads, head_dim, nH, ws, nW, ws)
            x = x.permute(0, 1, 3, 5, 4, 6, 2)          # (B, heads, nH, nW, ws, ws, hd)
            x = x.reshape(-1, ws * ws, head_dim)          # (B*heads*nH*nW, ws², hd)
            return x

        q_w = window_partition(q)
        k_w = window_partition(k)
        v_w = window_partition(v)

        # ----- scaled dot-product attention inside each window -----
        scale = head_dim ** -0.5
        attn_weights = torch.matmul(q_w, k_w.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        out = torch.matmul(attn_weights, v_w)

        # ----- merge windows back -----
        # (B*heads*nH*nW, ws², hd) -> (B, heads, nH, nW, ws, ws, hd)
        out = out.reshape(B, num_heads, nH, nW, ws, ws, head_dim)
        # -> (B, heads, hd, nH, ws, nW, ws)
        out = out.permute(0, 1, 6, 2, 4, 3, 5)
        # -> (B, C, H, W)
        out = out.reshape(B, C, H, W)

        return out


# ----------------------------------------------------------
# Quick sanity test
# ----------------------------------------------------------

if __name__ == "__main__":

    B, heads, head_dim = 1, 8, 40
    C = heads * head_dim  # 320
    H = W = 64

    q = torch.randn(B, C, H, W)
    k = torch.randn(B, C, H, W)
    v = torch.randn(B, C, H, W)

    model = WindowAttention(window_size=8)

    y = model(q, k, v, num_heads=heads)

    print(f"Input:  {q.shape}")
    print(f"Output: {y.shape}")
    assert y.shape == q.shape, "Shape mismatch!"
    print("✓ WindowAttention test passed.")
