"""Discrepancy slots in embedding space. No prefill add.

v1 added a zero-init MLP onto the last prefill state. ||h|| drowned Δ,
ranking never left the margin, and shuffle did not change the patch. This
version maps (expected, got) to a unit-RMS vector, scales it to token RMS,
and adds K slot offsets. Shuffle keeps the offsets and swaps the I/O pair.
"""

from __future__ import annotations

import torch
from torch import nn

from .mbpp_generalization import RepairExample, discrepancy_fields


class DiscrepancyResidual(nn.Module):
    """Z_k = α * RMSNorm(MLP([e, g, e-g])) + offset_k."""

    def __init__(self, hidden_width: int, *, num_slots: int = 8, mlp_width: int = 256):
        super().__init__()
        if hidden_width <= 0 or num_slots <= 0:
            raise ValueError("hidden_width and num_slots must be positive")
        self.hidden_width = hidden_width
        self.num_slots = num_slots
        self.delta = nn.Sequential(
            nn.LayerNorm(hidden_width * 3),
            nn.Linear(hidden_width * 3, mlp_width),
            nn.GELU(),
            nn.Linear(mlp_width, hidden_width),
        )
        self.slot_offsets = nn.Parameter(torch.zeros(num_slots, hidden_width))
        self.scale = nn.Parameter(torch.ones(()))
        self.probe = nn.Linear(hidden_width, hidden_width * 2)

    def set_scale_from_embeddings(self, embedding: nn.Embedding) -> None:
        with torch.no_grad():
            rms = embedding.weight.detach().float().pow(2).mean().sqrt().clamp(min=1e-3)
            self.scale.copy_(rms)

    def delta_vector(self, expected: torch.Tensor, got: torch.Tensor) -> torch.Tensor:
        if expected.shape != got.shape or expected.ndim != 2:
            raise ValueError("expected and got must share shape [batch, hidden]")
        features = torch.cat((expected, got, expected - got), dim=-1)
        return self.delta(features)

    def rms_norm(self, vector: torch.Tensor) -> torch.Tensor:
        return vector * torch.rsqrt(vector.pow(2).mean(dim=-1, keepdim=True) + 1e-6)

    def compose(self, delta: torch.Tensor) -> torch.Tensor:
        if delta.ndim != 2:
            raise ValueError("delta must have shape [batch, hidden]")
        normalized = self.rms_norm(delta)
        return self.scale * normalized.unsqueeze(1) + self.slot_offsets.unsqueeze(0)

    def probe_loss(self, slots: torch.Tensor, expected: torch.Tensor, got: torch.Tensor) -> torch.Tensor:
        predicted = self.probe(slots.float().mean(1))
        target = torch.cat((expected, got), dim=-1).detach()
        return nn.functional.mse_loss(predicted, target)

    def pool_text(self, tokenizer, embedding: nn.Embedding, text: str, *, max_length: int = 64) -> torch.Tensor:
        device = embedding.weight.device
        width = embedding.weight.shape[-1]
        if not text.strip():
            return torch.zeros(1, width, device=device, dtype=torch.float32)
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        ids = encoded["input_ids"].to(device)
        if ids.numel() == 0:
            return torch.zeros(1, width, device=device, dtype=torch.float32)
        return embedding(ids).float().mean(1)

    def encode(
        self,
        model,
        tokenizer,
        example: RepairExample,
        events,
        *,
        include_specification: bool = False,
        include_public_test: bool = True,
        corrupt: bool = False,
        max_length: int = 768,
    ) -> torch.Tensor:
        del example, include_specification, include_public_test, max_length
        expected_text, got_text = discrepancy_fields(tuple(events))
        if corrupt:
            expected_text, got_text = got_text, expected_text
        embedding = model.get_input_embeddings()
        expected = self.pool_text(tokenizer, embedding, expected_text)
        got = self.pool_text(tokenizer, embedding, got_text)
        return self.compose(self.delta_vector(expected, got))
