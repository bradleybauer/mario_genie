"""OneHotConv3d: causal 3D convolution optimized for palette-index input.

Provides two implementations behind a single ``nn.Module``:

* **PyTorch fallback** — loops over kernel offsets with ``F.embedding`` lookups.
  Works on any device.
* **Triton fused kernel** — a single GPU kernel that gathers weight rows for all
  27 kernel offsets and accumulates them per output element.  Autotuned for the
  typical training shapes (B=24, T=19, H=112/224, W=112/224, C_out=24, K=23).

The Triton path is selected automatically when the input is on a CUDA device and
Triton is importable.  Pass ``force_triton=False`` at construction time to
disable it.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# Triton kernel (optional)
# ---------------------------------------------------------------------------
_triton_available = False
try:
    import triton
    import triton.language as tl

    _triton_available = True
except ModuleNotFoundError:
    pass

if _triton_available:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 512, "BLOCK_C": 32}, num_warps=4),
            triton.Config({"BLOCK_N": 1024, "BLOCK_C": 32}, num_warps=4),
            triton.Config({"BLOCK_N": 2048, "BLOCK_C": 32}, num_warps=8),
            triton.Config({"BLOCK_N": 512, "BLOCK_C": 16}, num_warps=4),
            triton.Config({"BLOCK_N": 1024, "BLOCK_C": 16}, num_warps=4),
        ],
        key=["N", "C_out", "K_vol"],
    )
    @triton.jit
    def _onehot_conv3d_kernel(
        # Pointers
        indices_ptr,  # padded indices: (B, T_pad, H_pad, W_pad) int64
        weight_ptr,   # (C_out, K, kT, kH, kW) float — stored as (C_out, K, K_vol)
        bias_ptr,     # (C_out,) float or nullptr
        out_ptr,      # (B, C_out, T, H, W) float — output in BCTHW layout
        # Dims
        N: tl.constexpr,        # B * T * H * W
        C_out: tl.constexpr,
        K: tl.constexpr,        # num_classes
        K_vol: tl.constexpr,    # kT * kH * kW
        T: tl.constexpr,
        H: tl.constexpr,
        W: tl.constexpr,
        H_pad: tl.constexpr,
        W_pad: tl.constexpr,
        # Kernel shape
        kT: tl.constexpr,
        kH: tl.constexpr,
        kW: tl.constexpr,
        # Strides for padded indices (row-major: B, T_pad, H_pad, W_pad)
        stride_ib: tl.constexpr,
        stride_it: tl.constexpr,
        stride_ih: tl.constexpr,
        # stride_iw == 1 (contiguous innermost)
        # Strides for output (BCTHW)
        stride_ob: tl.constexpr,
        stride_oc: tl.constexpr,
        stride_ot: tl.constexpr,
        stride_oh: tl.constexpr,
        # stride_ow == 1 (contiguous innermost)
        HAS_BIAS: tl.constexpr,
        # Block sizes
        BLOCK_N: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        """Each program instance computes a tile of (spatial positions, output channels)."""
        pid_n = tl.program_id(0)
        pid_c = tl.program_id(1)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # spatial indices
        offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)  # channel indices

        mask_n = offs_n < N
        mask_c = offs_c < C_out
        mask = mask_n[:, None] & mask_c[None, :]

        # Decompose flat spatial index -> (b, t, h, w)
        b = offs_n // (T * H * W)
        rem = offs_n % (T * H * W)
        t = rem // (H * W)
        rem2 = rem % (H * W)
        h = rem2 // W
        w = rem2 % W

        # Accumulator
        acc = tl.zeros((BLOCK_N, BLOCK_C), dtype=tl.float32)

        # Loop over kernel volume
        for k_idx in tl.static_range(0, K_vol):
            # Decompose k_idx -> (dt, dh, dw)
            dt = k_idx // (kH * kW)
            dh = (k_idx % (kH * kW)) // kW
            dw = k_idx % kW

            # Read padded index at (b, t+dt, h+dh, w+dw)
            idx_ptr = indices_ptr + b * stride_ib + (t + dt) * stride_it + (h + dh) * stride_ih + (w + dw)
            cls_idx = tl.load(idx_ptr, mask=mask_n, other=0).to(tl.int32)  # (BLOCK_N,)

            # Weight lookup: weight[c, cls_idx, k_idx] for each (n, c) pair
            # weight is stored as (C_out, K, K_vol) row-major
            # w[c, cls, k_idx] = weight_ptr + c * K * K_vol + cls * K_vol + k_idx
            w_ptr = weight_ptr + offs_c[None, :] * (K * K_vol) + cls_idx[:, None] * K_vol + k_idx
            w_val = tl.load(w_ptr, mask=mask, other=0.0)  # (BLOCK_N, BLOCK_C)
            acc += w_val

        if HAS_BIAS:
            bias_val = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0)
            acc += bias_val[None, :]

        # Store output in BCTHW layout:  out[b, c, t, h, w]
        out_off = b[:, None] * stride_ob + offs_c[None, :] * stride_oc + t[:, None] * stride_ot + h[:, None] * stride_oh + w[:, None]
        tl.store(out_ptr + out_off, acc, mask=mask)


def _onehot_conv3d_triton(
    padded_indices: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    T: int,
    H: int,
    W: int,
    kT: int,
    kH: int,
    kW: int,
) -> Tensor:
    """Launch the Triton kernel for OneHotConv3d."""
    B = padded_indices.shape[0]
    C_out, K, K_vol = weight.shape[0], weight.shape[1], kT * kH * kW

    # Reshape weight to (C_out, K, K_vol) contiguous
    w = weight.reshape(C_out, K, K_vol).contiguous()

    N = B * T * H * W
    out = torch.empty((B, C_out, T, H, W), dtype=weight.dtype, device=weight.device)

    grid = lambda meta: (
        triton.cdiv(N, meta["BLOCK_N"]),
        triton.cdiv(C_out, meta["BLOCK_C"]),
    )

    _onehot_conv3d_kernel[grid](
        padded_indices,
        w,
        bias if bias is not None else w,  # dummy pointer, won't be read
        out,
        N=N,
        C_out=C_out,
        K=K,
        K_vol=K_vol,
        T=T,
        H=H,
        W=W,
        H_pad=H + (kH // 2) * 2,
        W_pad=W + (kW // 2) * 2,
        kT=kT,
        kH=kH,
        kW=kW,
        stride_ib=padded_indices.stride(0),
        stride_it=padded_indices.stride(1),
        stride_ih=padded_indices.stride(2),
        stride_ob=out.stride(0),
        stride_oc=out.stride(1),
        stride_ot=out.stride(2),
        stride_oh=out.stride(3),
        HAS_BIAS=bias is not None,
    )
    return out


# ---------------------------------------------------------------------------
# PyTorch fallback
# ---------------------------------------------------------------------------

def _onehot_conv3d_pytorch(
    padded_indices: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    T: int,
    H: int,
    W: int,
    kT: int,
    kH: int,
    kW: int,
) -> Tensor:
    """Pure-PyTorch implementation using F.embedding per kernel offset."""
    C_out = weight.shape[0]
    num_classes = weight.shape[1]
    # (kT*kH*kW, num_classes, out_channels)
    w = weight.permute(2, 3, 4, 1, 0).reshape(-1, num_classes, C_out)

    # F.embedding requires int64 indices
    if padded_indices.dtype != torch.int64:
        padded_indices = padded_indices.long()

    out: Tensor | None = None
    k = 0
    for dt in range(kT):
        for dh in range(kH):
            for dw in range(kW):
                shifted = padded_indices[:, dt : dt + T, dh : dh + H, dw : dw + W]
                contribution = F.embedding(shifted, w[k])
                if out is None:
                    out = contribution
                else:
                    out = out + contribution
                k += 1

    assert out is not None
    # (B, T, H, W, C) -> (B, C, T, H, W)
    out = out.permute(0, 4, 1, 2, 3)
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1, 1)
    return out


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

def _pad_indices(indices: Tensor, kT: int, kH: int, kW: int) -> Tensor:
    """Causal temporal + spatial replicate padding on int64 indices."""
    pad_t = kT - 1
    pad_h = kH // 2
    pad_w = kW // 2

    x = indices
    if pad_t > 0:
        first = x[:, :1].expand(-1, pad_t, -1, -1)
        x = torch.cat((first, x), dim=1)
    if pad_h > 0:
        top = x[:, :, :1, :].expand(-1, -1, pad_h, -1)
        bot = x[:, :, -1:, :].expand(-1, -1, pad_h, -1)
        x = torch.cat((top, x, bot), dim=2)
    if pad_w > 0:
        left = x[:, :, :, :1].expand(-1, -1, -1, pad_w)
        right = x[:, :, :, -1:].expand(-1, -1, -1, pad_w)
        x = torch.cat((left, x, right), dim=3)
    return x


class OneHotConv3d(nn.Module):
    """Causal 3D convolution optimized for one-hot (palette-index) input.

    Mathematically equivalent to ``CausalConv3d(num_classes, out_channels, kernel_size)``
    applied to a one-hot tensor, but accepts integer palette indices ``(B, T, H, W)``
    directly and replaces the dense multiply-accumulate with embedding lookups
    (one per kernel offset), eliminating ~96% of wasted multiply-by-zero work.

    The weight tensor has shape ``(out_channels, num_classes, kT, kH, kW)`` —
    identical to the inner ``nn.Conv3d`` of ``CausalConv3d`` — so existing
    checkpoints can be loaded with a simple key remap.

    When ``force_triton=True`` (default), a fused Triton kernel is used on CUDA
    if Triton is available.  Set ``force_triton=False`` to always use the PyTorch
    fallback (useful for CPU, MPS, or debugging).
    """

    def __init__(
        self,
        num_classes: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int] = 3,
        bias: bool = True,
        force_triton: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        self.num_classes = num_classes
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.force_triton = force_triton
        # Same layout as nn.Conv3d(num_classes, out_channels, kernel_size).
        self.weight = nn.Parameter(
            torch.empty(out_channels, num_classes, *kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    @classmethod
    def from_causal_conv3d(cls, conv: object) -> "OneHotConv3d":
        """Create from a ``CausalConv3d`` instance (avoids circular import)."""
        inner = conv.conv  # type: ignore[attr-defined]
        layer = cls(
            num_classes=inner.in_channels,
            out_channels=inner.out_channels,
            kernel_size=conv.kernel_size,  # type: ignore[attr-defined]
            bias=inner.bias is not None,
        )
        layer.weight.data.copy_(inner.weight.data)
        if inner.bias is not None and layer.bias is not None:
            layer.bias.data.copy_(inner.bias.data)
        return layer

    @property
    def _use_triton(self) -> bool:
        return self.force_triton and _triton_available

    def forward(self, indices: Tensor) -> Tensor:
        """
        Parameters
        ----------
        indices : Tensor
            ``(B, T, H, W)`` integer palette indices in ``[0, num_classes)``.
            Any integer dtype is accepted (uint8, int32, int64, etc.).

        Returns
        -------
        Tensor
            ``(B, out_channels, T, H, W)``
        """
        kT, kH, kW = self.kernel_size
        T, H, W = indices.shape[1], indices.shape[2], indices.shape[3]
        padded = _pad_indices(indices, kT, kH, kW)

        if self._use_triton and indices.is_cuda:
            padded = padded.contiguous()
            return _onehot_conv3d_triton(padded, self.weight, self.bias, T, H, W, kT, kH, kW)

        return _onehot_conv3d_pytorch(padded, self.weight, self.bias, T, H, W, kT, kH, kW)
