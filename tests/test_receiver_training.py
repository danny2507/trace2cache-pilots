from __future__ import annotations

import unittest
import torch
from torch import nn

from trace2cache.receiver_training import patch_nll, patch_score, patch_token_logprobs, paired_patch_loss, vector_loss, oracle_identity_loss


class ToyReceiver(nn.Module):
    def __init__(self, vocab: int = 11, width: int = 5) -> None:
        super().__init__(); self.embedding = nn.Embedding(vocab, width); self.head = nn.Linear(width, vocab)
    def get_input_embeddings(self): return self.embedding
    def forward(self, *, inputs_embeds, attention_mask, use_cache):
        return type("Output", (), {"logits": self.head(inputs_embeds)})()


class ReceiverTrainingTest(unittest.TestCase):
    def test_identity_prefers_matching_code_and_freezes_dictionary(self) -> None:
        codes = torch.eye(3).reshape(3, 1, 3).requires_grad_(True)
        latent = codes[:2].detach().clone().requires_grad_(True)
        right = oracle_identity_loss(latent, codes, torch.tensor([0, 1]))
        wrong = oracle_identity_loss(latent, codes, torch.tensor([1, 0]))
        self.assertLess(right.item(), wrong.item())
        wrong.backward()
        self.assertGreater(latent.grad.abs().sum().item(), 0)
        self.assertIsNone(codes.grad)
    def test_token_logprobs_and_masked_scores(self) -> None:
        model = ToyReceiver(); prompt = torch.randn(2, 3, 5, requires_grad=True); target = torch.tensor([[1, 2, 3], [3, 2, 1]])
        logprobs = patch_token_logprobs(model, prompt, target)
        self.assertEqual(tuple(logprobs.shape), (2, 3))
        self.assertTrue(torch.isfinite(logprobs).all())
        masked = patch_score(logprobs, torch.tensor([[1, 0, 1], [1, 1, 0]]))
        self.assertEqual(tuple(masked.shape), (2,)); self.assertGreater(patch_nll(logprobs).item(), 0)

    def test_decoder_loss_reaches_latent_through_frozen_receiver(self) -> None:
        model = ToyReceiver()
        for parameter in model.parameters(): parameter.requires_grad_(False)
        latent = torch.randn(1, 2, 5, requires_grad=True); positive = torch.tensor([[1, 2, 3]]); negative = torch.tensor([[3, 2, 1]])
        result = paired_patch_loss(model, latent, positive, negative, latent.detach().clone(), latent)
        result["loss"].backward()
        self.assertIsNotNone(latent.grad); self.assertGreater(latent.grad.abs().sum().item(), 0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_vector_loss_rejects_mismatched_shapes(self) -> None:
        with self.assertRaises(ValueError): vector_loss(torch.zeros(1, 2, 3), torch.zeros(1, 3, 2))
