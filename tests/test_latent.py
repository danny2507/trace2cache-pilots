from __future__ import annotations

import unittest

import torch
from torch import nn

from trace2cache.latent import NativeEventResampler, NativePairCodebook, NativePairEncoder


class LatentTests(unittest.TestCase):
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
