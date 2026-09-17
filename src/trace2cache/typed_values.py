"""Generic, bounded typed-value features for structured runtime evidence.

The runtime collector already serializes values as tagged JSON.  This module deliberately
only decodes that safe representation: it never evaluates a repr or source fragment.
Nested values become ordered virtual evidence nodes, so a list such as ``[3, -2]`` is not
reduced to an unordered bag or a length scalar before it reaches the trace encoder.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Iterator


KIND_NAMES = ("none", "bool", "int", "float", "str", "list", "tuple", "unknown", "exception")
KIND_INDEX = {name: index for index, name in enumerate(KIND_NAMES)}
# kind one-hot, numeric sign, integer bits, scalar fields, bool, ordered string sketch,
# structure/depth/index indicators.
FEATURE_DIM = len(KIND_NAMES) + 3 + 64 + 8 + 2 + 8 + 5
MAX_TYPED_NODES_PER_EVENT = 96


@dataclass(frozen=True)
class TypedNode:
    """One tagged runtime value, in pre-order traversal order."""

    value: dict[str, Any]
    depth: int
    child_index: int


def _tagged_objects(value: Any, *, depth: int = 0, child_index: int = 0) -> Iterator[TypedNode]:
    """Yield strictly-recognized tagged values embedded in an arbitrary JSON payload."""
    if isinstance(value, dict):
        kind = value.get("type")
        if isinstance(kind, str) and kind in KIND_INDEX:
            yield TypedNode(value=value, depth=depth, child_index=child_index)
            if kind in {"list", "tuple"} and isinstance(value.get("items"), list):
                for index, item in enumerate(value["items"]):
                    yield from _tagged_objects(item, depth=depth + 1, child_index=index)
            return
        # Payload wrappers (state, value, argument) are traversed in canonical key order.
        for key in sorted(value):
            yield from _tagged_objects(value[key], depth=depth, child_index=child_index)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _tagged_objects(item, depth=depth, child_index=index)


def parse_typed_nodes(content: str, *, limit: int = MAX_TYPED_NODES_PER_EVENT) -> tuple[TypedNode, ...]:
    """Recover bounded tagged values from a JSON evidence payload, or return no nodes.

    JSON parsing intentionally rejects non-finite constants and malformed payloads; these
    should remain represented only by the frozen native content feature.
    """
    if limit < 1:
        raise ValueError("limit must be positive")
    try:
        payload = json.loads(content, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except (TypeError, ValueError, json.JSONDecodeError):
        return ()
    nodes: list[TypedNode] = []
    for node in _tagged_objects(payload):
        nodes.append(node)
        if len(nodes) >= limit:
            break
    return tuple(nodes)


def _bounded(value: float, scale: float = 16.0) -> float:
    return math.tanh(value / scale)


def typed_feature(node: TypedNode) -> list[float]:
    """Return a deterministic fixed-width feature vector for one tagged value.

    The 64 integer bits encode the absolute magnitude; sign remains independent, making
    ``-1`` distinct from ``1`` and from ``True``.  Values outside signed int64 are treated
    as unknown rather than silently wrapping.
    """
    result = [0.0] * FEATURE_DIM
    kind = node.value.get("type")
    kind_index = KIND_INDEX.get(kind, KIND_INDEX["unknown"])
    result[kind_index] = 1.0
    sign_start = len(KIND_NAMES)
    bits_start = sign_start + 3
    scalar_start = bits_start + 64
    bool_start = scalar_start + 8
    string_start = bool_start + 2
    struct_start = string_start + 8
    raw = node.value.get("value")
    if kind == "int" and isinstance(raw, int) and not isinstance(raw, bool) and -(2**63) <= raw < 2**63:
        result[sign_start + (0 if raw < 0 else 1 if raw == 0 else 2)] = 1.0
        magnitude = abs(raw)
        for bit in range(64): result[bits_start + bit] = float((magnitude >> bit) & 1)
        result[scalar_start] = _bounded(math.log1p(magnitude))
    elif kind == "float" and isinstance(raw, (int, float)) and not isinstance(raw, bool) and math.isfinite(float(raw)):
        numeric = float(raw)
        result[sign_start + (0 if numeric < 0 else 1 if numeric == 0 else 2)] = 1.0
        result[scalar_start] = _bounded(math.log1p(abs(numeric)))
        result[scalar_start + 1] = math.tanh(numeric / 32.0)
    elif kind == "bool" and isinstance(raw, bool):
        result[bool_start + int(raw)] = 1.0
    elif kind == "str" and isinstance(raw, str):
        result[scalar_start] = _bounded(float(len(raw)))
        # Eight ordered byte buckets preserve a small, deterministic sketch of spelling.
        for index, char in enumerate(raw[:8]): result[string_start + index] = ord(char) / 255.0
    elif kind in {"list", "tuple"}:
        items = node.value.get("items")
        if isinstance(items, list): result[scalar_start] = _bounded(float(len(items)))
    result[struct_start] = min(node.depth, 8) / 8.0
    result[struct_start + 1] = _bounded(float(node.child_index))
    result[struct_start + 2] = float(node.child_index == 0)
    result[struct_start + 3] = float(kind in {"list", "tuple"})
    result[struct_start + 4] = float(node.depth > 0)
    return result
