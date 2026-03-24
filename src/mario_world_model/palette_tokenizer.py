"""PaletteVideoTokenizer — cross-entropy over a fixed colour palette.

Subclasses the upstream ``VideoTokenizer`` so that:
* ``channels = K`` (one per palette colour) instead of 3 (RGB).
* ``forward(return_loss=True)`` computes **cross-entropy** between the
  K-channel decoder logits and integer palette-index targets, replacing
  the library's built-in MSE loss.
* All non-loss forward paths (``return_codes``, ``return_recon``, etc.)
  delegate to the parent unchanged.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import Tensor

from magvit2_pytorch import VideoTokenizer
from magvit2_pytorch import magvit2_pytorch as magvit_impl
from magvit2_pytorch.magvit2_pytorch import LossBreakdown


_ORIGINAL_RESIDUAL_UNIT = magvit_impl.ResidualUnit
_ORIGINAL_RESIDUAL_UNIT_MOD = magvit_impl.ResidualUnitMod


@contextmanager
def _patched_residual_pad_mode(pad_mode: str):
    """Temporarily propagate a chosen pad mode into nested residual blocks."""

    def residual_unit(dim, kernel_size, pad_mode_override: str | None = None):
        return _ORIGINAL_RESIDUAL_UNIT(
            dim,
            kernel_size,
            pad_mode=pad_mode if pad_mode_override is None else pad_mode_override,
        )

    def residual_unit_mod(dim, kernel_size, *args, pad_mode_override: str | None = None, **kwargs):
        return _ORIGINAL_RESIDUAL_UNIT_MOD(
            dim,
            kernel_size,
            *args,
            pad_mode=pad_mode if pad_mode_override is None else pad_mode_override,
            **kwargs,
        )

    prev_residual_unit = magvit_impl.ResidualUnit
    prev_residual_unit_mod = magvit_impl.ResidualUnitMod
    magvit_impl.ResidualUnit = residual_unit
    magvit_impl.ResidualUnitMod = residual_unit_mod
    try:
        yield
    finally:
        magvit_impl.ResidualUnit = prev_residual_unit
        magvit_impl.ResidualUnitMod = prev_residual_unit_mod


class PaletteVideoTokenizer(VideoTokenizer):
    """``VideoTokenizer`` variant that operates in palette-index space.

    Parameters
    ----------
    num_palette_colors : int
        Number of discrete colours in the palette (``K``).  Sets the
        ``channels`` dimension of the underlying tokenizer.
    **kwargs
        Forwarded to ``VideoTokenizer.__init__``.  ``channels``,
        ``use_gan`` and ``perceptual_loss_weight`` are overridden.
    """

    def __init__(self, *, num_palette_colors: int, pad_mode: str = "replicate", **kwargs):
        kwargs["channels"] = num_palette_colors
        kwargs["use_gan"] = False
        kwargs["perceptual_loss_weight"] = 0.0
        kwargs["pad_mode"] = pad_mode
        with _patched_residual_pad_mode(pad_mode):
            super().__init__(**kwargs)
        self.num_palette_colors = num_palette_colors

    # ── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def indices_to_onehot(indices: Tensor, num_colors: int) -> Tensor:
        """``(B, T, H, W)`` long  →  ``(B, K, T, H, W)`` float one-hot."""
        return (
            F.one_hot(indices.long(), num_colors)
            .float()
            .permute(0, 4, 1, 2, 3)
        )

    # ── forward ─────────────────────────────────────────────────────

    def forward(
        self,
        video_or_images: Tensor,
        cond: Tensor | None = None,
        return_loss: bool = False,
        return_codes: bool = False,
        return_recon: bool = False,
        return_discr_loss: bool = False,
        return_recon_loss_only: bool = False,
        apply_gradient_penalty: bool = True,
        video_contains_first_frame: bool = True,
        adversarial_loss_weight: float | None = None,
        multiscale_adversarial_loss_weight: float | None = None,
        *,
        targets: Tensor | None = None,
        context_frames: int = 0,
    ):
        """Forward pass with optional cross-entropy loss.

        When ``return_loss=True`` the caller must supply ``targets`` —
        a ``(B, T, H, W)`` long tensor of palette colour indices.
        The model input ``video_or_images`` should be the corresponding
        ``(B, K, T, H, W)`` one-hot tensor (use :meth:`indices_to_onehot`).

        All other code-paths (``return_codes``, plain reconstruction, …)
        are forwarded to the parent class unchanged.
        """
        if not return_loss:
            return super().forward(
                video_or_images,
                cond=cond,
                return_loss=False,
                return_codes=return_codes,
                return_recon=return_recon,
                return_discr_loss=return_discr_loss,
                return_recon_loss_only=return_recon_loss_only,
                apply_gradient_penalty=apply_gradient_penalty,
                video_contains_first_frame=video_contains_first_frame,
                adversarial_loss_weight=adversarial_loss_weight,
                multiscale_adversarial_loss_weight=multiscale_adversarial_loss_weight,
            )

        assert targets is not None, (
            "targets (B, T, H, W) of long palette indices are required "
            "when return_loss=True"
        )

        # Encode
        x = self.encode(
            video_or_images,
            video_contains_first_frame=video_contains_first_frame,
        )

        if context_frames > 0:
          B, D, latent_T, H, W = x.shape
          downsampling_factor = video_or_images.shape[2] / latent_T
          latent_context_frames = int(context_frames / downsampling_factor)
          mask = torch.ones((B, latent_T, H, W), dtype=torch.bool, device=x.device)
          mask[:, :latent_context_frames, :, :] = False
          mask = mask.reshape(B, -1)
        else:
          mask = None

        # Quantize
        (quantized, codes, aux_losses), quantizer_loss_breakdown = (
            self.quantizers(x, inv_temperature = 5.0, return_loss_breakdown=True, mask=mask)
            # self.quantizers(x, return_loss_breakdown=True, mask=mask)
        )

        # Decode → (B, K, T, H, W) logits
        logits = self.decode(
            quantized,
            video_contains_first_frame=video_contains_first_frame,
        )

        # Cross-entropy loss over palette classes (skip context frames)
        if context_frames > 0:
            logits = logits[:, :, context_frames:]
            targets = targets[:, context_frames:]
        ce_loss = F.cross_entropy(logits, targets)
        total_loss = ce_loss + self.quantizer_aux_loss_weight * aux_losses

        loss_breakdown = LossBreakdown(
            recon_loss=ce_loss,
            lfq_aux_loss=aux_losses,
            quantizer_loss_breakdown=quantizer_loss_breakdown,
            perceptual_loss=self.zero,
            adversarial_gen_loss=self.zero,
            adaptive_adversarial_weight=0.0,
            multiscale_gen_losses=[],
            multiscale_gen_adaptive_weights=[],
        )

        return total_loss, loss_breakdown
