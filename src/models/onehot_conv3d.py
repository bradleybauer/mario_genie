"""OneHotConv3d: Triton-accelerated causal 3D convolution for palette-index input.

A single GPU kernel gathers weight rows for all 27 kernel offsets and accumulates
them per output element.  Autotuned for the typical training shapes
(B=24, T=19, H=112/224, W=112/224, C_out=24, K=23).

The backward pass materializes the one-hot input and delegates to cuDNN's
``conv3d_weight`` for the weight gradient.

Requires Triton and a CUDA device.  For CPU or non-Triton environments, use
``CausalConv3d`` directly.


Variant	Forward	Backward	Train Step
OneHot (Triton fwd + cuDNN bwd) —  new	1.43 ms	5.75 ms	7.18 ms
OneHot (Triton fwd + Triton bwd) — old	1.2 ms	35.8 ms	38.1 ms
CausalConv3d (dense, cuDNN)	            4.82 ms	5.40 ms	10.21 ms


"""
from __future__ import annotations

import torch
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------
import triton
import triton.language as tl

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


class _OneHotConv3dFunction(torch.autograd.Function):
    """Custom autograd for palette-index conv with a sparse weight-gradient update."""

    @staticmethod
    def forward(
        ctx,
        padded_indices: Tensor,
        weight: Tensor,
        bias: Tensor | None,
    ) -> Tensor:
        kT, kH, kW = weight.shape[2:]
        padded_indices_long = padded_indices if padded_indices.dtype == torch.int64 else padded_indices.long()

        ctx.save_for_backward(padded_indices_long)
        ctx.weight_shape = weight.shape
        ctx.weight_dtype = weight.dtype
        ctx.has_bias = bias is not None

        T = padded_indices_long.shape[1] - kT + 1
        H = padded_indices_long.shape[2] - 2 * (kH // 2)
        W = padded_indices_long.shape[3] - 2 * (kW // 2)
        return _onehot_conv3d_triton(
            padded_indices_long.contiguous(),
            weight,
            bias,
            T,
            H,
            W,
            kT,
            kH,
            kW,
        )

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[None, Tensor | None, Tensor | None, None]:
        (padded_indices,) = ctx.saved_tensors
        _, num_classes, kT, kH, kW = ctx.weight_shape
        grad_output_contig = grad_output.contiguous()

        grad_weight = None
        if ctx.needs_input_grad[1]:
            # Materialize one-hot from saved indices, then use cuDNN weight-grad.
            padded_onehot = torch.zeros(
                (padded_indices.shape[0], num_classes, *padded_indices.shape[1:]),
                device=padded_indices.device,
                dtype=grad_output_contig.dtype,
            )
            padded_onehot.scatter_(1, padded_indices.unsqueeze(1), 1.0)
            grad_weight = torch.nn.grad.conv3d_weight(
                padded_onehot,
                ctx.weight_shape,
                grad_output_contig,
                stride=(1, 1, 1),
                padding=(0, 0, 0),
                dilation=(1, 1, 1),
                groups=1,
            )
            if grad_weight.dtype != ctx.weight_dtype:
                grad_weight = grad_weight.to(ctx.weight_dtype)

        grad_bias = None
        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = grad_output_contig.sum(dim=(0, 2, 3, 4), dtype=ctx.weight_dtype)

        return None, grad_weight, grad_bias, None


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
    """Triton-accelerated causal 3D convolution for palette-index input.

    Mathematically equivalent to ``CausalConv3d(num_classes, out_channels, kernel_size)``
    applied to a one-hot tensor, but accepts integer palette indices ``(B, T, H, W)``
    directly.  A fused Triton kernel gathers weight rows for all kernel offsets,
    and the backward pass uses cuDNN's ``conv3d_weight`` for the weight gradient.

    Requires Triton and a CUDA device.  For CPU or non-Triton environments, use
    ``CausalConv3d`` directly.
    """

    def __init__(
        self,
        num_classes: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int] = 3,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        self.num_classes = num_classes
        self.out_channels = out_channels
        self.kernel_size = kernel_size
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
        padded = _pad_indices(indices, kT, kH, kW)

        if torch.is_grad_enabled():
            # Training: Triton forward + cuDNN weight-gradient backward.
            return _OneHotConv3dFunction.apply(padded, self.weight, self.bias)

        # Inference: Triton fused kernel (no autograd graph needed).
        T, H, W = indices.shape[1], indices.shape[2], indices.shape[3]
        padded = padded.contiguous()
        return _onehot_conv3d_triton(padded, self.weight, self.bias, T, H, W, kT, kH, kW)
