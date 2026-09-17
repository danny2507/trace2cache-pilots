from trace2cache.structured_runtime import canonical_value, tagged_value
from trace2cache.typed_values import FEATURE_DIM, parse_typed_nodes, typed_feature


def test_ordered_list_items_and_typed_int_signs_are_preserved() -> None:
    payload = canonical_value({"argument": 0, "value": tagged_value([3, -2, True])})
    nodes = parse_typed_nodes(payload)
    assert [node.value["type"] for node in nodes] == ["list", "int", "int", "bool"]
    assert [node.child_index for node in nodes] == [0, 0, 1, 2]
    positive, negative = typed_feature(nodes[1]), typed_feature(nodes[2])
    assert len(positive) == FEATURE_DIM
    # sign features begin immediately after the nine type features.
    assert positive[9:12] == [0.0, 0.0, 1.0]
    assert negative[9:12] == [1.0, 0.0, 0.0]
    assert positive[12] != negative[12]  # low bit of |3| versus |-2|
    assert positive[13] == negative[13] == 1.0


def test_malformed_or_untyped_text_never_becomes_a_value_node() -> None:
    assert parse_typed_nodes("not JSON") == ()
    assert parse_typed_nodes('{"statement":"x = 1"}') == ()
