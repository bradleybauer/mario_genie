from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class MemoryReport:
    context_frames: int
    vocab_size: int
    parameter_mb: float
    embedding_mb: float
    output_head_mb: float
    logits_mb: float
    kv_cache_mb: float
    measured_peak_mb: float | None


def _format_mb(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:8.1f}"


class OpenGenieStyleDynamics(nn.Module):
    """Minimal next-frame dynamics model inspired by Open-Genie.

    Structure mirrors the relevant high-level pieces:
    token embedding + action embedding + transformer stack + vocab head.
    It predicts one latent frame (frame_tokens logits) in parallel.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        action_vocab_size: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: int,
        context_frames: int,
        frame_tokens: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.context_frames = context_frames
        self.frame_tokens = frame_tokens

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.action_embedding = nn.Embedding(action_vocab_size, embed_dim)
        total_tokens = (context_frames + 1) * frame_tokens
        self.position_embedding = nn.Parameter(torch.randn(1, total_tokens, embed_dim) * 0.02)
        self.future_queries = nn.Parameter(torch.randn(1, frame_tokens, embed_dim) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.output_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, tokens: torch.Tensor, action_ids: torch.Tensor) -> torch.Tensor:
        batch_size, context_frames, frame_tokens = tokens.shape
        if context_frames != self.context_frames or frame_tokens != self.frame_tokens:
            raise ValueError(
                f"expected tokens of shape (B, {self.context_frames}, {self.frame_tokens}), "
                f"got {tuple(tokens.shape)}"
            )

        token_features = self.token_embedding(tokens)
        action_features = self.action_embedding(action_ids).unsqueeze(2)
        action_features = action_features.expand(-1, -1, frame_tokens, -1)
        hidden = token_features + action_features
        hidden = hidden.reshape(batch_size, context_frames * frame_tokens, self.embed_dim)
        future_queries = self.future_queries.expand(batch_size, -1, -1)
        hidden = torch.cat([hidden, future_queries], dim=1)
        hidden = hidden + self.position_embedding[:, : hidden.shape[1]]
        hidden = self.decoder(hidden)
        hidden = self.norm(hidden[:, -frame_tokens:])
        return self.output_head(hidden)


def _build_report(
    *,
    context_frames: int,
    vocab_size: int,
    batch_size: int,
    frame_tokens: int,
    embed_dim: int,
    num_layers: int,
    device: torch.device,
) -> MemoryReport:
    bytes_per_elem = 2 if device.type == "cuda" else 4
    sequence_tokens = (context_frames + 1) * frame_tokens
    parameter_mb = 0.0
    embedding_mb = vocab_size * embed_dim * bytes_per_elem / 2**20
    output_head_mb = (embed_dim * vocab_size + vocab_size) * bytes_per_elem / 2**20
    logits_mb = batch_size * frame_tokens * vocab_size * bytes_per_elem / 2**20
    kv_cache_mb = 2 * batch_size * sequence_tokens * embed_dim * num_layers * bytes_per_elem / 2**20
    return MemoryReport(
        context_frames=context_frames,
        vocab_size=vocab_size,
        parameter_mb=parameter_mb,
        embedding_mb=embedding_mb,
        output_head_mb=output_head_mb,
        logits_mb=logits_mb,
        kv_cache_mb=kv_cache_mb,
        measured_peak_mb=None,
    )


def _measure_peak_memory(
    *,
    context_frames: int,
    vocab_size: int,
    batch_size: int,
    frame_tokens: int,
    embed_dim: int,
    num_layers: int,
    num_heads: int,
    mlp_ratio: int,
    action_vocab_size: int,
    device: torch.device,
) -> MemoryReport:
    report = _build_report(
        context_frames=context_frames,
        vocab_size=vocab_size,
        batch_size=batch_size,
        frame_tokens=frame_tokens,
        embed_dim=embed_dim,
        num_layers=num_layers,
        device=device,
    )

    model = OpenGenieStyleDynamics(
        vocab_size=vocab_size,
        action_vocab_size=action_vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        context_frames=context_frames,
        frame_tokens=frame_tokens,
    )
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = model.to(device=device, dtype=dtype)

    total_params = sum(param.numel() for param in model.parameters())
    parameter_mb = total_params * torch.tensor([], dtype=dtype).element_size() / 2**20

    tokens = torch.randint(
        0,
        vocab_size,
        (batch_size, context_frames, frame_tokens),
        device=device,
        dtype=torch.long,
    )
    action_ids = torch.randint(
        0,
        action_vocab_size,
        (batch_size, context_frames),
        device=device,
        dtype=torch.long,
    )
    targets = torch.randint(
        0,
        vocab_size,
        (batch_size, frame_tokens),
        device=device,
        dtype=torch.long,
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    logits = model(tokens, action_ids)
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
    loss.backward()

    peak_mb = None
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / 2**20

    del logits, loss, targets, action_ids, tokens, model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return MemoryReport(
        context_frames=context_frames,
        vocab_size=vocab_size,
        parameter_mb=parameter_mb,
        embedding_mb=report.embedding_mb,
        output_head_mb=report.output_head_mb,
        logits_mb=report.logits_mb,
        kv_cache_mb=report.kv_cache_mb,
        measured_peak_mb=peak_mb,
    )


def test_open_genie_style_memory_report(capsys: pytest.CaptureFixture[str]) -> None:
    # One predicted 14x14 latent frame conditioned on context frames.
    batch_size = 32 if torch.cuda.is_available() else 4
    context_frames = 8
    frame_tokens = 14 * 14
    embed_dim = 512
    num_layers = 12
    num_heads = 8
    mlp_ratio = 4
    action_vocab_size = 8
    vocab_sizes = [4096, 10_000, 65_536]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reports = [
        _measure_peak_memory(
            context_frames=context_frames,
            vocab_size=vocab_size,
            batch_size=batch_size,
            frame_tokens=frame_tokens,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            action_vocab_size=action_vocab_size,
            device=device,
        )
        for vocab_size in vocab_sizes
    ]

    header = (
        f"\nOpen-Genie-style dynamics memory on {device.type} "
        f"(batch={batch_size}, context_frames={context_frames}, frame_tokens={frame_tokens}, "
        f"embed_dim={embed_dim}, layers={num_layers})\n"
        "vocab | params MB | embed MB | head MB | logits MB | kv-cache MB | peak train MB"
    )
    rows = [
        f"{report.vocab_size:5d} | "
        f"{_format_mb(report.parameter_mb)} | "
        f"{_format_mb(report.embedding_mb)} | "
        f"{_format_mb(report.output_head_mb)} | "
        f"{_format_mb(report.logits_mb)} | "
        f"{_format_mb(report.kv_cache_mb)} | "
        f"{_format_mb(report.measured_peak_mb)}"
        for report in reports
    ]
    print(header)
    print("\n".join(rows))
    captured = capsys.readouterr()

    # Memory should rise monotonically with larger vocabularies.
    assert [report.embedding_mb for report in reports] == sorted(report.embedding_mb for report in reports)
    assert [report.output_head_mb for report in reports] == sorted(report.output_head_mb for report in reports)
    assert [report.logits_mb for report in reports] == sorted(report.logits_mb for report in reports)
    assert "Open-Genie-style dynamics memory" in captured.out


def test_open_genie_style_context_sweep(capsys: pytest.CaptureFixture[str]) -> None:
    batch_size = 32 if torch.cuda.is_available() else 4
    context_frames_list = [4, 8, 16, 32]
    frame_tokens = 14 * 14
    embed_dim = 512
    num_layers = 12
    num_heads = 8
    mlp_ratio = 4
    action_vocab_size = 8
    vocab_sizes = [4096, 65_536]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reports = [
        _measure_peak_memory(
            context_frames=context_frames,
            vocab_size=vocab_size,
            batch_size=batch_size,
            frame_tokens=frame_tokens,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            action_vocab_size=action_vocab_size,
            device=device,
        )
        for context_frames in context_frames_list
        for vocab_size in vocab_sizes
    ]

    print(
        f"\nOpen-Genie-style context sweep on {device.type} "
        f"(batch={batch_size}, frame_tokens={frame_tokens}, embed_dim={embed_dim}, layers={num_layers})"
    )
    print("context | vocab | logits MB | kv-cache MB | peak train MB")
    for report in reports:
        print(
            f"{report.context_frames:7d} | "
            f"{report.vocab_size:5d} | "
            f"{_format_mb(report.logits_mb)} | "
            f"{_format_mb(report.kv_cache_mb)} | "
            f"{_format_mb(report.measured_peak_mb)}"
        )

    captured = capsys.readouterr()
    assert "Open-Genie-style context sweep" in captured.out
    for vocab_size in vocab_sizes:
        peaks = [
            report.measured_peak_mb
            for report in reports
            if report.vocab_size == vocab_size and report.measured_peak_mb is not None
        ]
        assert peaks == sorted(peaks)