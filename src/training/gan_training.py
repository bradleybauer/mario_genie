from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def hinge_discriminator_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    return (torch.relu(1.0 - real_logits) + torch.relu(1.0 + fake_logits)).mean()


def hinge_generator_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return -fake_logits.mean()


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


@dataclass
class LeCAMEMA:
    decay: float
    logits_real_ema: torch.Tensor | None = None
    logits_fake_ema: torch.Tensor | None = None

    def update(self, real_logit_mean: torch.Tensor, fake_logit_mean: torch.Tensor) -> None:
        real_value = real_logit_mean.detach()
        fake_value = fake_logit_mean.detach()
        if self.logits_real_ema is None or self.logits_fake_ema is None:
            self.logits_real_ema = real_value
            self.logits_fake_ema = fake_value
            return

        self.logits_real_ema = self.decay * self.logits_real_ema + (1.0 - self.decay) * real_value
        self.logits_fake_ema = self.decay * self.logits_fake_ema + (1.0 - self.decay) * fake_value

    def regularizer(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        if self.logits_real_ema is None or self.logits_fake_ema is None:
            return real_logits.new_zeros(())

        reg_real = torch.relu(real_logits - self.logits_fake_ema).pow(2).mean()
        reg_fake = torch.relu(self.logits_real_ema - fake_logits).pow(2).mean()
        return reg_real + reg_fake

    def state_dict(self) -> dict[str, torch.Tensor | float | None]:
        return {
            "decay": self.decay,
            "logits_real_ema": self.logits_real_ema,
            "logits_fake_ema": self.logits_fake_ema,
        }

    def load_state_dict(self, state: dict[str, torch.Tensor | float | None]) -> None:
        self.decay = float(state.get("decay", self.decay))
        self.logits_real_ema = state.get("logits_real_ema")
        self.logits_fake_ema = state.get("logits_fake_ema")