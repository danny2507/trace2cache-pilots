"""Frozen receiver-native payload features for paired-runtime v2.

``mean0`` exposes the receiver's input-embedding geometry. ``context2`` executes exactly
the first two frozen Qwen2 decoder layers, making each payload contextual without allowing
the trainable evidence adapter to modify the language model.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import torch
from torch import nn


SCHEMA_VERSION = 2


@dataclass(frozen=True)
class FeatureCacheKey:
    model_id: str
    tokenizer_fingerprint: str
    payload: str
    method: str
    pooling: str
    layer_count: int
    max_content_tokens: int
    schema_version: int = SCHEMA_VERSION

    def digest(self) -> str:
        raw = json.dumps(self.__dict__, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()


class FrozenNativeFeatureExtractor:
    """Feature-only facade over a frozen causal LM. It never owns/train()s the model."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: object,
        *,
        model_id: str,
        cache_dir: str | Path = ".local/cache/paired_runtime_v2/features",
        max_content_tokens: int = 128,
        layer_count: int = 2,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("native feature extraction requires CUDA")
        if layer_count != 2:
            raise ValueError("v2 preregisters exactly two contextual Qwen layers")
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.cache_dir = Path(cache_dir)
        self.max_content_tokens = max_content_tokens
        self.layer_count = layer_count
        self.device = next(model.parameters()).device
        self.width = model.config.hidden_size
        self._freeze()
        probe = tokenizer(" trace2cache", add_special_tokens=False)["input_ids"]
        self.tokenizer_fingerprint = hashlib.sha256(json.dumps(probe).encode()).hexdigest()

    def _freeze(self) -> None:
        self.model.eval()
        for parameter in self.model.parameters(): parameter.requires_grad_(False)

    def cache_key(self, payload: str, method: str) -> FeatureCacheKey:
        if method not in {"mean0", "context2"}: raise ValueError(method)
        return FeatureCacheKey(self.model_id, self.tokenizer_fingerprint, payload, method, "last_valid", self.layer_count if method == "context2" else 0, self.max_content_tokens)

    def _path(self, key: FeatureCacheKey) -> Path:
        return self.cache_dir / key.method / f"{key.digest()}.pt"

    def _load_cached(self, key: FeatureCacheKey) -> torch.Tensor | None:
        path = self._path(key)
        if not path.exists(): return None
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("key") != key.__dict__: raise RuntimeError(f"feature cache key mismatch: {path}")
        value = payload["feature"]
        if tuple(value.shape) != (self.width,) or not torch.isfinite(value.float()).all():
            raise RuntimeError(f"invalid feature cache entry: {path}")
        return value

    def _save_cached(self, key: FeatureCacheKey, feature: torch.Tensor) -> None:
        path = self._path(key); path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        torch.save({"key": key.__dict__, "feature": feature.detach().cpu().to(torch.bfloat16)}, temporary)
        temporary.replace(path)

    def _tokenize(self, contents: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        batch = self.tokenizer(
            contents, return_tensors="pt", add_special_tokens=False, padding=True,
            truncation=True, max_length=self.max_content_tokens,
        )
        ids, mask = batch["input_ids"], batch["attention_mask"].bool()
        if ids.shape[1] == 0 or not mask.any(dim=1).all(): raise ValueError("empty payload tokenization")
        return ids.to(self.device), mask.to(self.device)

    @staticmethod
    def _causal_padding_mask(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Qwen attention mask: 0 allowed, dtype minimum for future/padded keys."""
        _, length = mask.shape
        future = torch.triu(torch.ones(length, length, device=mask.device, dtype=torch.bool), diagonal=1)
        blocked = future.unsqueeze(0) | ~mask[:, None, :]
        result = torch.zeros((mask.shape[0], 1, length, length), device=mask.device, dtype=dtype)
        return result.masked_fill(blocked.unsqueeze(1), torch.finfo(dtype).min)

    @torch.inference_mode()
    def _compute(self, contents: list[str], method: str) -> torch.Tensor:
        ids, valid = self._tokenize(contents)
        embeddings = self.model.get_input_embeddings()(ids)
        if method == "mean0":
            result = (embeddings.float() * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True)
        elif method == "context2":
            hidden = embeddings
            attention = self._causal_padding_mask(valid, hidden.dtype)
            positions = torch.arange(ids.shape[1], device=self.device).unsqueeze(0).expand(ids.shape[0], -1)
            cache_positions = torch.arange(ids.shape[1], device=self.device)
            for layer in self.model.model.layers[:self.layer_count]:
                hidden = layer(hidden, attention_mask=attention, position_ids=positions, past_key_value=None,
                               output_attentions=False, use_cache=False, cache_position=cache_positions)[0]
            last = valid.long().sum(1).sub(1)
            result = hidden[torch.arange(ids.shape[0], device=self.device), last].float()
        else: raise ValueError(method)
        if not torch.isfinite(result).all(): raise RuntimeError("non-finite native feature")
        return result.detach().to(torch.bfloat16)

    def extract_contents(self, contents: list[str], method: str, *, batch_size: int = 16) -> torch.Tensor:
        """Return detached native features in original order, caching unique payloads."""
        unique = list(dict.fromkeys(contents)); resolved: dict[str, torch.Tensor] = {}; missing: list[str] = []
        for content in unique:
            cached = self._load_cached(self.cache_key(content, method))
            if cached is None: missing.append(content)
            else: resolved[content] = cached
        for start in range(0, len(missing), batch_size):
            chunk = missing[start:start + batch_size]
            features = self._compute(chunk, method)
            for content, feature in zip(chunk, features):
                feature = feature.cpu(); self._save_cached(self.cache_key(content, method), feature); resolved[content] = feature
        return torch.stack([resolved[content] for content in contents]).to(self.device)

    @torch.inference_mode()
    def verify_partial_parity(self, contents: list[str], *, atol: float = 0.08, rtol: float = 0.08) -> float:
        """Compare explicit two-layer execution to the unpadded Qwen hidden-state path."""
        if not contents: raise ValueError("need at least one payload")
        ids, valid = self._tokenize(contents)
        if not valid.all(): raise ValueError("parity check requires unpadded same-length payloads")
        ours = self._compute(contents, "context2").float()
        full = self.model.model(input_ids=ids, attention_mask=valid.long(), output_hidden_states=True, use_cache=False)
        hidden = full.hidden_states[self.layer_count]
        theirs = hidden[:, -1].float()
        difference = (ours - theirs).abs().max().item()
        if not torch.allclose(ours, theirs, atol=atol, rtol=rtol): raise AssertionError(f"context2 partial/full parity failed: max_abs={difference}")
        return difference


def collate_views(views: Iterable[object], feature_map: dict[str, torch.Tensor], device: torch.device, *, typed_values: bool = False, binding_relations: bool = False) -> dict[str, torch.Tensor]:
    """Pad EvidenceViews, optionally expanding safe tagged values into ordered nodes."""
    from .typed_values import FEATURE_DIM, parse_typed_nodes, typed_feature
    views = list(views)
    if not views: raise ValueError("cannot collate no views")
    if typed_values and binding_relations:
        raise ValueError("typed expansion and binding relations are separate representation ablations")
    expanded = []
    for view in views:
        entries = []
        for event in view.events:
            entries.append((event, None))
            if typed_values:
                entries.extend((event, node) for node in parse_typed_nodes(event.content))
        expanded.append(entries)
    width = max(len(entries) for entries in expanded); hidden = next(iter(feature_map.values())).shape[-1]
    contents = torch.zeros(len(views), width, hidden, dtype=torch.bfloat16, device=device)
    roles = torch.zeros(len(views), width, dtype=torch.long, device=device)
    tests = torch.zeros(len(views), width, dtype=torch.long, device=device)
    mask = torch.zeros(len(views), width, dtype=torch.bool, device=device)
    typed = torch.zeros(len(views), width, FEATURE_DIM, dtype=torch.float32, device=device) if typed_values else None
    relations = torch.zeros(len(views), width, width, dtype=torch.long, device=device) if binding_relations else None
    for batch_index, entries in enumerate(expanded):
        for event_index, (event, node) in enumerate(entries):
            contents[batch_index, event_index] = feature_map[event.content].to(device)
            roles[batch_index, event_index] = event.role_id; tests[batch_index, event_index] = event.test_id; mask[batch_index, event_index] = True
            if node is not None: typed[batch_index, event_index] = torch.tensor(typed_feature(node), device=device)
        if relations is not None:
            from .binding_encoder import SELF, relation_edges
            valid = len(entries)
            diagonal = torch.arange(valid, device=device)
            relations[batch_index, diagonal, diagonal] = SELF
            # Binding mode has no virtual nodes, so view event indices are tensor indices.
            for source, target, relation in relation_edges(views[batch_index].events):
                relations[batch_index, target, source] = relation
    result = {"content_vectors": contents, "role_ids": roles, "test_ids": tests, "event_mask": mask}
    if typed is not None: result["typed_features"] = typed
    if relations is not None: result["relation_ids"] = relations
    return result
