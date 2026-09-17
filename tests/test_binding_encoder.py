from __future__ import annotations

import json

import torch

from trace2cache.binding_encoder import LAST_OBSERVED_VALUE, NEXT, SAME_SOURCE, SELF, relation_edges
from trace2cache.latent import RoleAwareEventEncoder
from trace2cache.native_features import collate_views
from trace2cache.paired_evidence import EvidenceEvent, EvidenceView
from trace2cache.structured_runtime import tagged_value


def payload(state: dict[str, object]) -> str:
    return json.dumps({"state": {name: tagged_value(value) for name, value in state.items()}}, sort_keys=True)


def test_relations_are_auditable_observed_topology_not_variable_tokens() -> None:
    events = (
        EvidenceEvent(0, 0, 0, 1, "{}", None),
        EvidenceEvent(1, 0, 1, 5, payload({"x": 1}), 4),
        EvidenceEvent(2, 0, 2, 5, payload({"x": 1, "y": 2}), 4),
        EvidenceEvent(3, 0, 3, 5, payload({"x": 3, "y": 2}), 5),
    )
    edges = set(relation_edges(events))
    assert {(0, 1, NEXT), (1, 2, NEXT), (2, 3, NEXT)} <= edges
    assert (1, 2, SAME_SOURCE) in edges
    assert (1, 2, LAST_OBSERVED_VALUE) in edges
    assert (1, 3, LAST_OBSERVED_VALUE) in edges
    # Relations contain only indices/types: the variable spelling never becomes a feature.
    assert all(isinstance(value, int) for edge in edges for value in edge)


def test_relation_collation_and_zero_warm_start_are_padding_safe() -> None:
    view = EvidenceView("test", (
        EvidenceEvent(0, 0, 0, 1, "{}", None),
        EvidenceEvent(1, 0, 1, 5, payload({"x": 1}), 4),
    ))
    feature_map = {event.content: torch.randn(16, dtype=torch.bfloat16) for event in view.events}
    batch = collate_views([view], feature_map, torch.device("cpu"), binding_relations=True)
    assert batch["relation_ids"][0, 0, 0].item() == SELF
    assert batch["relation_ids"].shape == (1, 2, 2)
    encoder = RoleAwareEventEncoder(model_width=16, output_anchor=torch.randn(3, 16), hidden_width=16, num_roles=10, max_events=4, max_tests=2, num_layers=1, num_heads=4, relation_types=5)
    changed = batch["relation_ids"].clone(); changed[0, 1, 0] = LAST_OBSERVED_VALUE
    first = encoder(**batch)
    second = encoder(batch["content_vectors"], batch["role_ids"], batch["test_ids"], batch["event_mask"], relation_ids=changed)
    assert torch.equal(first, second)
    second.sum().backward()
    assert encoder.relation_layer.output.weight.grad is not None
