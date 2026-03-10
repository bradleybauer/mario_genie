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

import torch
import torch.nn.functional as F
from torch import Tensor

from magvit2_pytorch import VideoTokenizer
from magvit2_pytorch.magvit2_pytorch import LossBreakdown


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

    def __init__(self, *, num_palette_colors: int, **kwargs):
        kwargs["channels"] = num_palette_colors
        kwargs["use_gan"] = False
        kwargs["perceptual_loss_weight"] = 0.0
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
        video_contains_first_frame: bool = True,
        *,
        targets: Tensor | None = None,
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
                video_contains_first_frame=video_contains_first_frame,
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

        # Quantize
        (quantized, codes, aux_losses), quantizer_loss_breakdown = (
            self.quantizers(x, return_loss_breakdown=True)
        )

        # Decode → (B, K, T, H, W) logits
        logits = self.decode(
            quantized,
            video_contains_first_frame=video_contains_first_frame,
        )

        # Cross-entropy loss over palette classes
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
