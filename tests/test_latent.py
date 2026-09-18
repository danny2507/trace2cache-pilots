from __future__ import annotations

import unittest
import json

import torch
from torch import nn

from trace2cache.latent import (
    NativeEventResampler,
    NativePairCodebook,
    NativePairEncoder,
    RoleAwareEventEncoder,
)
from trace2cache.native_features import collate_views
from trace2cache.paired_evidence import EvidenceEvent, EvidenceView
from trace2cache.structured_runtime import tagged_value


class LatentTests(unittest.TestCase):
    def test_role_aware_encoder_shape_and_role_signal(self):
        anchor = torch.randn(3, 16)
        encoder = RoleAwareEventEncoder(
            model_width=16,
            output_anchor=anchor,
            hidden_width=16,
            num_roles=4,
            max_events=6,
            max_tests=3,
            num_layers=1,
            num_heads=4,
        )
        contents = torch.randn(2, 5, 16)
        roles = torch.zeros(2, 5, dtype=torch.long)
        tests = torch.tensor([[0, 0, 1, 1, 1], [0, 0, 0, 1, 1]])
        mask = torch.ones(2, 5, dtype=torch.bool)
        initial = encoder(contents, roles, tests, mask)
        self.assertEqual(initial.shape, (2, 3, 16))
        self.assertTrue(torch.allclose(initial, anchor.unsqueeze(0).expand(2, -1, -1)))
        nn.init.normal_(encoder.output[-1].weight, std=0.02)
        changed_roles = roles.clone()
        changed_roles[:, 2] = 1
        self.assertFalse(
            torch.allclose(
                encoder(contents, roles, tests, mask),
                encoder(contents, changed_roles, tests, mask),
            )
        )

    def test_typed_channel_is_a_zero_perturbation_at_warm_start(self):
        encoder = RoleAwareEventEncoder(
            model_width=16, output_anchor=torch.randn(3, 16), hidden_width=16,
            num_roles=4, max_events=6, max_tests=3, num_layers=1, num_heads=4,
            typed_feature_dim=7,
        )
        contents = torch.randn(1, 5, 16)
        roles = torch.zeros(1, 5, dtype=torch.long)
        tests = torch.zeros(1, 5, dtype=torch.long)
        mask = torch.ones(1, 5, dtype=torch.bool)
        baseline = encoder(contents, roles, tests, mask, torch.zeros(1, 5, 7))
        typed = encoder(contents, roles, tests, mask, torch.randn(1, 5, 7))
        self.assertEqual((baseline - typed).abs().max().item(), 0.0)
        typed.sum().backward()
        self.assertIsNotNone(encoder.typed_projection[1].weight.grad)

    def test_parent_aggregated_typed_collation_preserves_plain_warm_start(self):
        view = EvidenceView(
            "typed-warm-start",
            (
                EvidenceEvent(0, 0, 0, 1, "plain event", None),
                EvidenceEvent(
                    1,
                    0,
                    1,
                    5,
                    json.dumps({"state": {"values": tagged_value([3, -2, True])}}),
                    4,
                ),
            ),
        )
        features = {event.content: torch.randn(16, dtype=torch.bfloat16) for event in view.events}
        plain = RoleAwareEventEncoder(
            model_width=16, output_anchor=torch.randn(3, 16), hidden_width=16,
            num_roles=10, max_events=4, max_tests=2, num_layers=1, num_heads=4,
        )
        typed = RoleAwareEventEncoder(
            model_width=16, output_anchor=plain.output_anchor, hidden_width=16,
            num_roles=10, max_events=4, max_tests=2, num_layers=1, num_heads=4,
            typed_feature_dim=99,
        )
        missing, unexpected = typed.load_state_dict(plain.state_dict(), strict=False)
        self.assertEqual(sorted(missing), ["typed_projection.0.bias", "typed_projection.0.weight", "typed_projection.1.weight"])
        self.assertEqual(unexpected, [])
        nn.init.normal_(plain.output[-1].weight, std=0.02)
        typed.output[-1].weight.data.copy_(plain.output[-1].weight.data)
        plain_batch = collate_views([view], features, torch.device("cpu"))
        typed_batch = collate_views([view], features, torch.device("cpu"), typed_values=True)
        for name in ("content_vectors", "role_ids", "test_ids", "event_mask"):
            self.assertTrue(torch.equal(plain_batch[name], typed_batch[name]), name)
        self.assertEqual(typed_batch["typed_features"].shape[:2], plain_batch["event_mask"].shape)
        self.assertTrue(torch.allclose(plain(**plain_batch), typed(**typed_batch), atol=0, rtol=0))
        typed(**typed_batch).sum().backward()
        self.assertGreater(typed.typed_projection[1].weight.grad.abs().max().item(), 0.0)
    def test_pair_encoder_is_parametric_and_native_anchored(self):
        embedding = nn.Embedding(60, 16)
        digit_ids = list(range(10, 20))
        encoder = NativePairEncoder(embedding, digit_ids, hidden_width=12)
        reference = torch.tensor([2, 7])
        buggy = torch.tensor([9, 1])
        output = encoder(reference, buggy)[:, 0]
        expected = embedding(torch.tensor([digit_ids[2], digit_ids[7]]))
        self.assertEqual((output - expected).abs().max().item(), 0.0)
        output.sum().backward()
        self.assertIsNotNone(encoder.residual[-1].weight.grad)
        self.assertIsNone(embedding.weight.grad)

    def test_pair_codebook_is_exact_reference_anchor_at_initialization(self):
        embedding = nn.Embedding(60, 16)
        digit_ids = list(range(10, 20))
        codebook = NativePairCodebook(embedding, digit_ids)
        reference = torch.tensor([2, 7])
        buggy = torch.tensor([9, 1])
        output = codebook(reference, buggy)[:, 0]
        expected = embedding(torch.tensor([digit_ids[2], digit_ids[7]]))
        self.assertEqual((output - expected).abs().max().item(), 0.0)

        output.sum().backward()
        self.assertIsNotNone(codebook.residual.weight.grad)
        self.assertIsNone(embedding.weight.grad)

    def test_resampler_shape_and_gradient(self):
        embedding = nn.Embedding(50, 16)
        adapter = NativeEventResampler(
            embedding, model_width=16, hidden_width=16, num_latents=3, num_heads=4
        )
        event_types = torch.tensor([[0, 1, 2, 0], [0, 2, 0, 0]])
        arguments = torch.randint(0, 50, (2, 4))
        states = torch.randint(0, 50, (2, 4))
        mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
        output = adapter(event_types, arguments, states, mask)
        self.assertEqual(output.shape, (2, 3, 16))
        output.sum().backward()
        self.assertIsNotNone(adapter.latent_queries.grad)
        self.assertIsNone(embedding.weight.grad)

    def test_zero_initialized_trace_output(self):
        embedding = nn.Embedding(50, 16)
        adapter = NativeEventResampler(
            embedding,
            model_width=16,
            hidden_width=16,
            num_latents=1,
            num_heads=4,
            zero_output_init=True,
        )
        output = adapter(
            torch.tensor([[0, 1]]),
            torch.tensor([[1, 2]]),
            torch.tensor([[3, 4]]),
            torch.ones(1, 2, dtype=torch.bool),
        )
        self.assertEqual(output.abs().max().item(), 0.0)

    def test_fixed_positions_support_untrained_longer_indices(self):
        embedding = nn.Embedding(50, 16)
        adapter = NativeEventResampler(
            embedding,
            model_width=16,
            hidden_width=16,
            num_latents=1,
            num_heads=4,
            max_events=12,
            fixed_positions=True,
        )
        output = adapter(
            torch.zeros(2, 10, dtype=torch.long),
            torch.randint(0, 50, (2, 10)),
            torch.randint(0, 50, (2, 10)),
            torch.ones(2, 10, dtype=torch.bool),
        )
        self.assertEqual(output.shape, (2, 1, 16))

    def test_native_sink_anchor_stays_close_to_last_state(self):
        embedding = nn.Embedding(50, 16)
        adapter = NativeEventResampler(
            embedding,
            model_width=16,
            hidden_width=16,
            num_latents=1,
            num_heads=4,
            native_sink_anchor=True,
        )
        event_types = torch.tensor([[0, 1, 2]])
        arguments = torch.tensor([[1, 2, 3]])
        states = torch.tensor([[4, 5, 6]])
        mask = torch.tensor([[1, 1, 1]], dtype=torch.bool)
        output = adapter(event_types, arguments, states, mask)[:, 0]
        sink = embedding(states[:, -1])
        self.assertEqual((output - sink).abs().max().item(), 0.0)

    def test_explicit_native_anchor_overrides_sink(self):
        embedding = nn.Embedding(50, 16)
        adapter = NativeEventResampler(
            embedding,
            model_width=16,
            hidden_width=16,
            num_latents=1,
            num_heads=4,
            native_sink_anchor=True,
        )
        event_types = torch.tensor([[0, 1, 2]])
        arguments = torch.tensor([[1, 2, 3]])
        states = torch.tensor([[4, 5, 6]])
        mask = torch.ones(1, 3, dtype=torch.bool)
        anchor_ids = torch.tensor([9])
        output = adapter(
            event_types, arguments, states, mask, anchor_token_ids=anchor_ids
        )[:, 0]
        self.assertEqual((output - embedding(anchor_ids)).abs().max().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
