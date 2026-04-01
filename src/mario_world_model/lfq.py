from __future__ import annotations

from contextlib import nullcontext
from functools import partial

import torch
import torch.distributed as dist
from torch.distributed import nn as dist_nn
import torch.nn.functional as F
from einops import pack, rearrange, reduce, unpack
from torch import Tensor
from torch.amp import autocast
from vector_quantize_pytorch.lookup_free_quantization import LFQ, LossBreakdown, Return


def _exists(value) -> bool:
    return value is not None


def _default(value, default_value):
    if _exists(value):
        return value
    return default_value() if callable(default_value) else default_value


def _pack_one(tensor: Tensor, pattern: str) -> tuple[Tensor, list[Tensor]]:
    packed, shapes = pack([tensor], pattern)
    return packed, shapes


def _unpack_one(tensor: Tensor, shapes: list[Tensor], pattern: str) -> Tensor:
    return unpack(tensor, shapes, pattern)[0]


def _is_distributed() -> bool:
    return dist.is_initialized() and dist.get_world_size() > 1


def _maybe_distributed_mean(tensor: Tensor) -> Tensor:
    if not _is_distributed():
        return tensor

    dist_nn.all_reduce(tensor)
    tensor = tensor / dist.get_world_size()
    return tensor


def _binary_entropy(prob: Tensor, eps: float = 1e-6) -> Tensor:
    prob = prob.clamp(min=eps, max=1.0 - eps)
    return -(prob * prob.log() + (1.0 - prob) * (1.0 - prob).log())


# not sure if this is mathematically correct but lets see how it works
# 
# The per-sample entropy is **exact**. The batch entropy is an **upper bound**, not exact.
# 
# Here's why:
# 
# **Per-sample entropy (exact)**
# 
# In LFQ the upstream logit for codeword $c$ is $2\tau \langle x, c \rangle$ (from `distance = -2 * einsum(...)` then `(-distance * inv_temperature).softmax(...)`). Since each $c_j \in \{+s, -s\}$, the partition function factorizes:
# 
# $$Z = \prod_j \bigl(\exp(2\tau s\, x_j) + \exp(-2\tau s\, x_j)\bigr)$$
# 
# so the joint distribution over codewords is a product of independent Bernoulli terms per bit, with $p_j^+ = \sigma(4\tau s\, x_j)$. Therefore:
# 
# $$H(P_x) = \sum_j H_{\text{bin}}(p_j^+)$$
# 
# This matches the code exactly — the `4.0` scaling factor is correct.
# 
# **Batch entropy (upper bound, not exact)**
# 
# Each individual sample's distribution $P_{x_i}$ is a product distribution. But the batch-averaged distribution $\bar{P} = \frac{1}{N}\sum_i P_{x_i}$ is a **mixture** of product distributions, which is generally **not** itself a product distribution. The code computes:
# 
# $$\hat{H}_{\text{batch}} = \sum_j H_{\text{bin}}\!\bigl(\bar{p}_j^+\bigr)$$
# 
# which is the entropy of the product distribution with marginals $\bar{p}_j^+$. By maximum-entropy / subadditivity:
# 
# $$H(\bar{P}) \;\leq\; \sum_j H_{\text{bin}}(\bar{p}_j^+) = \hat{H}_{\text{batch}}$$
# 
# The gap equals the **multi-information** (total correlation) among bits under the mixture — i.e., the correlations between bits that arise when different samples "activate" correlated bit patterns.
# 
# **Practical impact on the loss**
# 
# The loss term is $H_{\text{per-sample}} - \gamma \cdot H_{\text{batch}}$. Overestimating $H_{\text{batch}}$ makes the loss appear lower than it truly is, meaning the diversity pressure is slightly **weaker** than intended — the optimizer thinks codebook utilization is already higher than it really is. In practice for large batches with relatively uncorrelated bit assignments this gap is small, but it can miss pathological cases where individual bit marginals look uniform yet bits are strongly correlated (e.g., two bits always agree → only 2 of 4 joint states used, but per-bit marginals are both 0.5).
# 
# **Summary**: the `# not sure if this is mathematically correct` comment is warranted for the batch entropy term. The per-sample entropy is exact; the batch entropy is a correct upper bound but not an equality.
class EfficientLFQ(LFQ):
    """LFQ variant with a low-memory entropy estimator.

    ``entropy_mode='factorized'`` avoids constructing the full
    ``(tokens, codebooks, codebook_size)`` probability tensor used by the
    upstream LFQ entropy loss.

    For LFQ's binary codebook, per-code logits factorize across bit dimensions,
    so entropy can be estimated with per-bit Bernoulli terms in
    O(tokens * codebooks * log2(codebook_size)) memory instead of
    O(tokens * codebooks * codebook_size).
    """

    def __init__(self, *args, entropy_mode: str = "factorized", **kwargs):
        super().__init__(*args, **kwargs)
        if entropy_mode not in {"factorized", "legacy"}:
            raise ValueError("entropy_mode must be 'factorized' or 'legacy'")
        self.entropy_mode = entropy_mode

    def _factorized_entropy_terms(
        self,
        inputs: Tensor,
        *,
        inv_temperature: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if inputs.numel() == 0:
            return self.zero, self.zero, self.zero

        if self.frac_per_sample_entropy < 1.0:
            num_tokens = inputs.size(0)
            num_sampled_tokens = max(1, int(num_tokens * self.frac_per_sample_entropy))
            permutation = torch.randperm(num_tokens, device=inputs.device)
            inputs = inputs[permutation[:num_sampled_tokens]]

        # Codebook entries are binary per bit (plus/minus scale), so the
        # softmax over all codewords factorizes into independent Bernoulli terms.
        bit_scale = self.codebook.abs().amax().to(device=inputs.device, dtype=inputs.dtype)
        bit_logits = inputs * (4.0 * inv_temperature * bit_scale)
        p_plus = torch.sigmoid(bit_logits)

        per_sample_entropy = _binary_entropy(p_plus).sum(dim=-1).mean()

        avg_p = p_plus.mean(dim=0)
        avg_p = _maybe_distributed_mean(avg_p)
        batch_entropy = _binary_entropy(avg_p).sum(dim=-1).mean()

        entropy_aux_loss = per_sample_entropy - self.diversity_gamma * batch_entropy
        return entropy_aux_loss, per_sample_entropy, batch_entropy

    def forward(
        self,
        x: Tensor,
        inv_temperature: float = 100.0,
        return_loss_breakdown: bool = False,
        mask: Tensor | None = None,
    ):
        if self.entropy_mode == "legacy":
            return super().forward(
                x,
                inv_temperature=inv_temperature,
                return_loss_breakdown=return_loss_breakdown,
                mask=mask,
            )

        is_img_or_video = x.ndim >= 4
        should_transpose = _default(self.channel_first, is_img_or_video)

        if should_transpose:
            x = rearrange(x, "b d ... -> b ... d")
            x, packed_shapes = _pack_one(x, "b * d")

        if x.shape[-1] != self.dim:
            raise ValueError(f"expected dimension {self.dim}, got {x.shape[-1]}")

        x = self.project_in(x)

        if _exists(self.soft_clamp_input_value):
            clamp_value = self.soft_clamp_input_value
            x = (x / clamp_value).tanh() * clamp_value

        x = rearrange(x, "b n (c d) -> b n c d", c=self.num_codebooks)
        x = self.maybe_l2norm(x)

        force_f32 = self.force_quantization_f32
        quantization_context = partial(autocast, "cuda", enabled=False) if force_f32 else nullcontext

        with quantization_context():
            if force_f32:
                original_dtype = x.dtype
                x = x.float()

            original_input = x

            codebook_value = torch.ones_like(x) * self.codebook_scale
            quantized = torch.where(x > 0, codebook_value, -codebook_value)
            indices = reduce((quantized > 0).int() * self.mask.int(), "b n c d -> b n c", "sum")

            quantized = self.maybe_l2norm(quantized)

            if self.training:
                x = self.activation(x)
                x = x + (quantized - x).detach()
            else:
                x = quantized

            if self.training:
                input_for_entropy = original_input
                if _exists(mask):
                    input_for_entropy = original_input[mask]
                input_for_entropy = rearrange(input_for_entropy, "b n ... -> (b n) ...")

                entropy_aux_loss, per_sample_entropy, batch_entropy = self._factorized_entropy_terms(
                    input_for_entropy,
                    inv_temperature=inv_temperature,
                )
            else:
                entropy_aux_loss = per_sample_entropy = batch_entropy = self.zero

            if self.training and self.experimental_softplus_entropy_loss:
                entropy_aux_loss = F.softplus(entropy_aux_loss + self.entropy_loss_offset)

            if self.training and self.commitment_loss_weight > 0.0:
                commit_loss = F.mse_loss(original_input, quantized.detach(), reduction="none")
                if _exists(mask):
                    commit_loss = commit_loss[mask]
                commit_loss = commit_loss.mean()
            else:
                commit_loss = self.zero

            if force_f32:
                x = x.to(dtype=original_dtype)

        x = rearrange(x, "b n c d -> b n (c d)")
        x = self.project_out(x)

        if should_transpose:
            x = _unpack_one(x, packed_shapes, "b * d")
            x = rearrange(x, "b ... d -> b d ...")
            indices = _unpack_one(indices, packed_shapes, "b * c")

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, "... 1 -> ...")

        aux_loss = entropy_aux_loss * self.entropy_loss_weight + commit_loss * self.commitment_loss_weight

        ret = Return(x, indices, aux_loss)
        if not return_loss_breakdown:
            return ret

        return ret, LossBreakdown(per_sample_entropy, batch_entropy, commit_loss)


def swap_tokenizer_lfq(tokenizer, *, entropy_mode: str = "factorized") -> None:
    """Replace ``tokenizer.quantizers`` with ``EfficientLFQ`` in-place.

    This keeps existing quantizer weights/state while swapping the entropy
    implementation.
    """

    old_lfq = tokenizer.quantizers
    if not isinstance(old_lfq, LFQ):
        raise TypeError("tokenizer.quantizers is not an LFQ instance")

    if isinstance(old_lfq, EfficientLFQ):
        old_lfq.entropy_mode = entropy_mode
        return

    has_projections = old_lfq.has_projections
    projection_has_bias = bool(
        has_projections
        and hasattr(old_lfq.project_in, "bias")
        and old_lfq.project_in.bias is not None
    )

    cosine_sim_project_in = old_lfq.project_in.__class__.__name__ == "CosineSimLinear"
    cosine_sim_project_in_scale = (
        old_lfq.project_in.scale
        if cosine_sim_project_in and hasattr(old_lfq.project_in, "scale")
        else None
    )

    new_lfq = EfficientLFQ(
        dim=old_lfq.dim,
        codebook_size=old_lfq.codebook_size,
        entropy_loss_weight=old_lfq.entropy_loss_weight,
        commitment_loss_weight=old_lfq.commitment_loss_weight,
        diversity_gamma=old_lfq.diversity_gamma,
        straight_through_activation=old_lfq.activation,
        num_codebooks=old_lfq.num_codebooks,
        keep_num_codebooks_dim=old_lfq.keep_num_codebooks_dim,
        codebook_scale=old_lfq.codebook_scale,
        frac_per_sample_entropy=old_lfq.frac_per_sample_entropy,
        has_projections=has_projections,
        projection_has_bias=projection_has_bias,
        soft_clamp_input_value=old_lfq.soft_clamp_input_value,
        cosine_sim_project_in=cosine_sim_project_in,
        cosine_sim_project_in_scale=cosine_sim_project_in_scale,
        channel_first=old_lfq.channel_first,
        experimental_softplus_entropy_loss=old_lfq.experimental_softplus_entropy_loss,
        entropy_loss_offset=old_lfq.entropy_loss_offset,
        spherical=old_lfq.spherical,
        force_quantization_f32=old_lfq.force_quantization_f32,
        entropy_mode=entropy_mode,
    ).to(old_lfq.zero.device)

    new_lfq.load_state_dict(old_lfq.state_dict())
    tokenizer.quantizers = new_lfq
