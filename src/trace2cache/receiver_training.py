"""Decoder-facing objectives for frozen-receiver latent repair experiments.

The language model is frozen, but its forward pass deliberately remains differentiable with
respect to the supplied soft states.  This is the distinction between a usable receiver
loss and accidentally training only a vector regressor.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


MARKER = "<TRACE2CACHE_REPAIR_CODE>"


def splice_prompt(model: object, tokenizer: object, rendered: str, latent: torch.Tensor | None) -> torch.Tensor:
    """Replace one textual marker by ``[B,K,H]`` states, preserving prompt token order."""
    embedding = model.get_input_embeddings()
    if latent is None:
        ids = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)["input_ids"].to(embedding.weight.device)
        return embedding(ids)
    if MARKER not in rendered: raise ValueError("rendered prompt has no latent marker")
    if latent.ndim != 3 or latent.shape[-1] != embedding.weight.shape[-1]: raise ValueError("invalid latent shape")
    before, after = rendered.split(MARKER, 1)
    before_ids = tokenizer(before, return_tensors="pt", add_special_tokens=False)["input_ids"].to(embedding.weight.device)
    after_ids = tokenizer(after, return_tensors="pt", add_special_tokens=False)["input_ids"].to(embedding.weight.device)
    before_embed, after_embed = embedding(before_ids), embedding(after_ids)
    if latent.shape[0] != before_embed.shape[0]:
        if before_embed.shape[0] != 1: raise ValueError("latent and prompt batch mismatch")
        before_embed = before_embed.expand(latent.shape[0], -1, -1); after_embed = after_embed.expand(latent.shape[0], -1, -1)
    return torch.cat((before_embed, latent.to(before_embed.dtype), after_embed), dim=1)


def patch_token_logprobs(model: object, prompt_embeds: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Target-only causal log probabilities with the alignment used by the codebook pilot."""
    if prompt_embeds.ndim != 3 or target_ids.ndim != 2: raise ValueError("expected [B,P,H] prompt and [B,T] targets")
    if prompt_embeds.shape[0] != target_ids.shape[0]: raise ValueError("batch mismatch")
    if target_ids.shape[1] < 1: raise ValueError("target must contain at least one token")
    embedding = model.get_input_embeddings()
    teacher = embedding(target_ids[:, :-1])
    inputs = torch.cat((prompt_embeds, teacher), dim=1)
    attention = torch.ones(inputs.shape[:2], dtype=torch.long, device=inputs.device)
    logits = model(inputs_embeds=inputs, attention_mask=attention, use_cache=False).logits.float()
    start = prompt_embeds.shape[1] - 1
    predicted = logits[:, start:start + target_ids.shape[1]]
    if predicted.shape[:2] != target_ids.shape: raise RuntimeError("causal target alignment failed")
    return F.log_softmax(predicted, dim=-1).gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)


def patch_score(logprobs: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
    if token_mask is None: return logprobs.mean(dim=-1)
    token_mask = token_mask.to(dtype=logprobs.dtype)
    if token_mask.shape != logprobs.shape: raise ValueError("token mask shape mismatch")
    if (token_mask.sum(-1) == 0).any(): raise ValueError("token mask excludes every token")
    return (logprobs * token_mask).sum(-1) / token_mask.sum(-1)


def patch_nll(logprobs: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
    return -patch_score(logprobs, token_mask).mean()


def vector_loss(latent: torch.Tensor, oracle_code: torch.Tensor, *, cosine_weight: float = 0.1) -> torch.Tensor:
    if latent.shape != oracle_code.shape: raise ValueError("oracle shape mismatch")
    mse = F.mse_loss(latent.float(), oracle_code.float())
    cosine = F.cosine_similarity(latent.flatten(1).float(), oracle_code.flatten(1).float()).mean()
    return mse + cosine_weight * (1 - cosine)


def oracle_identity_loss(latent: torch.Tensor, oracle_codes: torch.Tensor, labels: torch.Tensor, *, temperature: float = 0.1) -> torch.Tensor:
    """Known-vocabulary diagnostic: contrast all oracle codes using evidence-derived states.

    Labels are supervision only. This does not add a label lookup at inference and does not
    provide a transferable code dictionary for unseen MBPP programs.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    prediction = F.normalize(latent.flatten(1).float(), dim=-1)
    dictionary = F.normalize(oracle_codes.detach().flatten(1).float(), dim=-1)
    return F.cross_entropy(prediction @ dictionary.T / temperature, labels)


def paired_patch_loss(
    model: object,
    prompt_embeds: torch.Tensor,
    positive_target_ids: torch.Tensor,
    negative_target_ids: torch.Tensor,
    oracle_code: torch.Tensor,
    latent: torch.Tensor,
    *,
    margin: float = 0.1,
    vector_weight: float = 0.1,
) -> dict[str, torch.Tensor]:
    """NLL + pairwise candidate rank + auxiliary oracle vector supervision."""
    positive_logprobs = patch_token_logprobs(model, prompt_embeds, positive_target_ids)
    negative_logprobs = patch_token_logprobs(model, prompt_embeds, negative_target_ids)
    positive_score = patch_score(positive_logprobs)
    negative_score = patch_score(negative_logprobs)
    nll = patch_nll(positive_logprobs)
    rank = F.softplus(margin - positive_score + negative_score).mean()
    vector = vector_loss(latent, oracle_code)
    total = nll + rank + vector_weight * vector
    return {"loss": total, "nll": nll, "rank": rank, "vector": vector, "positive_score": positive_score.mean(), "negative_score": negative_score.mean(), "margin": (positive_score - negative_score).mean()}
