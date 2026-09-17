"""Small model-native event resampler used by the latent sufficiency pilot."""

from __future__ import annotations

import math

import torch
from torch import nn


class NativePairCodebook(nn.Module):
    """Oracle bottleneck: one learned native-anchored vector for each value pair.

    This deliberately removes trace encoding from the experiment. If the frozen
    receiver cannot recover both values from these codes, a more complex trace
    encoder cannot rescue the one-slot interface.
    """

    def __init__(self, token_embedding: nn.Embedding, digit_token_ids: list[int]) -> None:
        super().__init__()
        if len(digit_token_ids) != 10:
            raise ValueError("expected token ids for digits 0 through 9")
        self.token_embedding = token_embedding
        self.token_embedding.requires_grad_(False)
        self.register_buffer("digit_token_ids", torch.tensor(digit_token_ids, dtype=torch.long))
        self.residual = nn.Embedding(100, token_embedding.embedding_dim)
        nn.init.zeros_(self.residual.weight)

    def forward(self, reference_values: torch.Tensor, buggy_values: torch.Tensor) -> torch.Tensor:
        pair_ids = reference_values * 10 + buggy_values
        reference_token_ids = self.digit_token_ids[reference_values]
        anchor = self.token_embedding(reference_token_ids).float()
        return (anchor + self.residual(pair_ids)).unsqueeze(1)


class NativePairEncoder(nn.Module):
    """Parametric one-slot encoder that must compose previously unseen value pairs."""

    def __init__(
        self,
        token_embedding: nn.Embedding,
        digit_token_ids: list[int],
        *,
        hidden_width: int = 512,
    ) -> None:
        super().__init__()
        if len(digit_token_ids) != 10:
            raise ValueError("expected token ids for digits 0 through 9")
        self.token_embedding = token_embedding
        self.token_embedding.requires_grad_(False)
        self.register_buffer("digit_token_ids", torch.tensor(digit_token_ids, dtype=torch.long))
        model_width = token_embedding.embedding_dim
        self.residual = nn.Sequential(
            nn.LayerNorm(model_width * 3),
            nn.Linear(model_width * 3, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, model_width),
        )
        # Exact native reference anchor at initialization while retaining a direct
        # gradient into the output layer from the first optimization step.
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, reference_values: torch.Tensor, buggy_values: torch.Tensor) -> torch.Tensor:
        reference_ids = self.digit_token_ids[reference_values]
        buggy_ids = self.digit_token_ids[buggy_values]
        reference = self.token_embedding(reference_ids).float()
        buggy = self.token_embedding(buggy_ids).float()
        features = torch.cat((reference, buggy, buggy - reference), dim=-1)
        return (reference + self.residual(features)).unsqueeze(1)


class NativeEventResampler(nn.Module):
    """Compress structured runtime events to K vectors in the receiver embedding space.

    Scalar values enter through the frozen receiver's native token embedding table. Event type and
    temporal position remain explicit structural fields.
    """

    def __init__(
        self,
        token_embedding: nn.Embedding,
        *,
        model_width: int,
        hidden_width: int = 256,
        num_latents: int = 4,
        num_event_types: int = 3,
        max_events: int = 32,
        num_layers: int = 2,
        num_heads: int = 4,
        native_sink_anchor: bool = False,
        zero_output_init: bool = False,
        fixed_positions: bool = False,
    ) -> None:
        super().__init__()
        self.token_embedding = token_embedding
        self.token_embedding.requires_grad_(False)
        self.type_embedding = nn.Embedding(num_event_types, hidden_width)
        self.fixed_positions = fixed_positions
        self.position_embedding = nn.Embedding(max_events, hidden_width)
        if fixed_positions:
            positions = torch.arange(max_events, dtype=torch.float32).unsqueeze(1)
            frequencies = torch.exp(
                torch.arange(0, hidden_width, 2, dtype=torch.float32)
                * (-math.log(10_000.0) / hidden_width)
            )
            encoding = torch.zeros(max_events, hidden_width)
            encoding[:, 0::2] = torch.sin(positions * frequencies)
            encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
            self.register_buffer("fixed_position_encoding", encoding)
        self.native_projection = nn.Linear(model_width * 2, hidden_width)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_width,
            nhead=num_heads,
            dim_feedforward=hidden_width * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.event_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.latent_queries = nn.Parameter(torch.randn(num_latents, hidden_width) * 0.02)
        self.resampler = nn.MultiheadAttention(
            hidden_width, num_heads=num_heads, dropout=0.0, batch_first=True
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_width),
            nn.Linear(hidden_width, hidden_width * 4),
            nn.GELU(),
            nn.Linear(hidden_width * 4, model_width),
        )
        if zero_output_init:
            nn.init.zeros_(self.output[-1].weight)
            nn.init.zeros_(self.output[-1].bias)
        self.native_sink_anchor = native_sink_anchor
        # Exact native anchor at initialization; the residual can open gradually during training.
        self.residual_gate = nn.Parameter(torch.tensor(0.0))

    @property
    def num_latents(self) -> int:
        return self.latent_queries.shape[0]

    def forward(
        self,
        event_types: torch.Tensor,
        argument_token_ids: torch.Tensor,
        state_token_ids: torch.Tensor,
        event_mask: torch.Tensor,
        anchor_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, event_count = event_types.shape
        if event_count > self.position_embedding.num_embeddings:
            raise ValueError("event sequence exceeds configured max_events")
        # The receiver embedding table is BF16 on GPU; the small adapter trains in FP32.
        argument = self.token_embedding(argument_token_ids).float()
        state = self.token_embedding(state_token_ids).float()
        native = self.native_projection(torch.cat((argument, state), dim=-1))
        positions = torch.arange(event_count, device=event_types.device).unsqueeze(0)
        position = (
            self.fixed_position_encoding[:event_count].unsqueeze(0)
            if self.fixed_positions
            else self.position_embedding(positions)
        )
        events = native + self.type_embedding(event_types) + position
        padding_mask = ~event_mask.bool()
        events = self.event_encoder(events, src_key_padding_mask=padding_mask)
        queries = self.latent_queries.unsqueeze(0).expand(batch_size, -1, -1)
        latents, _ = self.resampler(
            queries, events, events, key_padding_mask=padding_mask, need_weights=False
        )
        output = self.output(latents)
        if self.native_sink_anchor:
            if anchor_token_ids is None:
                last_indices = event_mask.long().sum(dim=1).sub(1).clamp_min(0)
                batch_indices = torch.arange(batch_size, device=event_types.device)
                sink = state[batch_indices, last_indices].unsqueeze(1)
            else:
                sink = self.token_embedding(anchor_token_ids).float().unsqueeze(1)
            sink = sink.expand(-1, self.num_latents, -1)
            # Stay close to a representation already understood by the frozen receiver.
            output = sink + self.residual_gate * torch.tanh(output)
        return output


class RoleAwareEventEncoder(nn.Module):
    """Compress native event contents while preserving role and test identity."""

    def __init__(
        self,
        *,
        model_width: int,
        output_anchor: torch.Tensor,
        hidden_width: int = 256,
        num_roles: int = 10,
        max_events: int = 192,
        max_tests: int = 8,
        num_layers: int = 2,
        num_heads: int = 4,
        slot_roles: tuple[tuple[int, ...], ...] | None = None,
        typed_feature_dim: int = 0,
    ) -> None:
        super().__init__()
        if output_anchor.ndim != 2 or output_anchor.shape[1] != model_width:
            raise ValueError("output_anchor must have shape [slots, model_width]")
        self.register_buffer("output_anchor", output_anchor.float().clone())
        self.content_projection = nn.Linear(model_width, hidden_width)
        self.typed_feature_dim = typed_feature_dim
        self.typed_projection = (
            nn.Sequential(nn.LayerNorm(typed_feature_dim), nn.Linear(typed_feature_dim, hidden_width, bias=False))
            if typed_feature_dim else None
        )
        self.role_embedding = nn.Embedding(num_roles, hidden_width)
        self.register_buffer(
            "event_positions", self._sinusoidal(max_events, hidden_width), persistent=False
        )
        self.register_buffer(
            "test_positions", self._sinusoidal(max_tests, hidden_width), persistent=False
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_width,
            nhead=num_heads,
            dim_feedforward=hidden_width * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.event_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.latent_queries = nn.Parameter(
            torch.randn(output_anchor.shape[0], hidden_width) * 0.02
        )
        self.resampler = nn.MultiheadAttention(
            hidden_width, num_heads=num_heads, dropout=0.0, batch_first=True
        )
        self.num_heads = num_heads
        if slot_roles is None:
            slot_roles = tuple(tuple(range(num_roles)) for _ in range(output_anchor.shape[0]))
        if len(slot_roles) != output_anchor.shape[0]:
            raise ValueError("slot_roles must specify one allowed-role set per latent slot")
        allowed = torch.zeros(output_anchor.shape[0], num_roles, dtype=torch.bool)
        for slot, roles in enumerate(slot_roles):
            if not roles or any(role < 0 or role >= num_roles for role in roles):
                raise ValueError("slot_roles contains an invalid or empty role set")
            allowed[slot, list(roles)] = True
        self.register_buffer("slot_role_mask", allowed)
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_width),
            nn.Linear(hidden_width, hidden_width * 4),
            nn.GELU(),
            nn.Linear(hidden_width * 4, model_width),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    @staticmethod
    def _sinusoidal(length: int, width: int) -> torch.Tensor:
        positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, width, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / width)
        )
        encoding = torch.zeros(length, width)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(
            positions * frequencies[: encoding[:, 1::2].shape[1]]
        )
        return encoding

    @property
    def num_latents(self) -> int:
        return self.output_anchor.shape[0]

    def forward(
        self,
        content_vectors: torch.Tensor,
        role_ids: torch.Tensor,
        test_ids: torch.Tensor,
        event_mask: torch.Tensor,
        typed_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, event_count, _ = content_vectors.shape
        if event_count > self.event_positions.shape[0]:
            raise ValueError("event sequence exceeds configured max_events")
        if test_ids.max().item() >= self.test_positions.shape[0]:
            raise ValueError("test index exceeds configured max_tests")
        positions = self.event_positions[:event_count].unsqueeze(0)
        tests = self.test_positions[test_ids]
        events = (
            self.content_projection(content_vectors.float())
            + self.role_embedding(role_ids)
            + positions
            + tests
        )
        if self.typed_projection is not None:
            if typed_features is None or typed_features.shape[:2] != event_mask.shape or typed_features.shape[-1] != self.typed_feature_dim:
                raise ValueError("typed features must match the configured event batch")
            events = events + self.typed_projection(typed_features.float())
        padding_mask = ~event_mask.bool()
        events = self.event_encoder(events, src_key_padding_mask=padding_mask)
        queries = self.latent_queries.unsqueeze(0).expand(batch_size, -1, -1)
        # Each latent slot has a declared evidence budget.  This makes a branch
        # slot unable to silently attend to arbitrary state events, while the
        # test role can be included as a stable fallback for sparse traces.
        allowed = self.slot_role_mask[:, role_ids].permute(1, 0, 2)
        blocked = padding_mask.unsqueeze(1) | ~allowed
        attention_mask = blocked.unsqueeze(1).expand(
            -1, self.num_heads, -1, -1
        ).reshape(batch_size * self.num_heads, self.num_latents, event_count)
        latents, _ = self.resampler(
            queries, events, events, attn_mask=attention_mask, need_weights=False
        )
        return self.output_anchor.unsqueeze(0) + self.output(latents)


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
