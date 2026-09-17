from __future__ import annotations

import unittest
import torch

from trace2cache.native_features import FeatureCacheKey, FrozenNativeFeatureExtractor


class NativeFeatureKeyTest(unittest.TestCase):
    def test_cache_key_changes_for_semantic_cache_dimensions(self) -> None:
        base = FeatureCacheKey("model", "tokenizer", "payload", "mean0", "last_valid", 0, 128)
        changed = FeatureCacheKey("model", "tokenizer", "payload", "context2", "last_valid", 2, 128)
        self.assertNotEqual(base.digest(), changed.digest())

    def test_causal_mask_blocks_future_and_padding(self) -> None:
        valid = torch.tensor([[True, True, False]])
        mask = FrozenNativeFeatureExtractor._causal_padding_mask(valid, torch.float32)
        self.assertEqual(mask.shape, (1, 1, 3, 3))
        self.assertEqual(mask[0, 0, 1, 0].item(), 0.0)
        self.assertLess(mask[0, 0, 0, 1].item(), -1e30)
        self.assertLess(mask[0, 0, 2, 2].item(), -1e30)
