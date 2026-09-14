---
license: apache-2.0
base_model: Qwen/Qwen2.5-Coder-3B-Instruct
library_name: pytorch
tags:
  - code
  - program-repair
  - execution-trace
  - latent-representation
  - negative-result
---

# Trace2Cache MBPP role-aware trace encoders

Trainable-only role-aware trace encoders from an early Trace2Cache MBPP repair pilot. A frozen
`Qwen/Qwen2.5-Coder-3B-Instruct` receiver consumes eight continuous states generated from typed
Python runtime events. The files exclude the frozen Qwen model and all datasets.

These are research artifacts and **do not demonstrate instance-specific latent runtime
communication for program repair**. They are released to make the negative result reproducible.

## Contents

- `adapters/mbpp_initial_seed173.safetensors`: 400-microstep full-patch run.
- `adapters/mbpp_contrastive_augmented_seed173.safetensors`: 800-microstep run with expanded
  deterministic mutations, type-directed fuzzing, and a different-task shuffled-trace hinge loss.
- `results/*.json`: complete evaluation payloads, including generated patch validation.
- `manifest.json` and `SHA256SUMS`: architecture/provenance and integrity metadata.

Each adapter has 4,734,208 trainable parameters. Instantiate `RoleAwareEventEncoder` from
`scripts/run_mbpp_latent_repair.py` in the source repository, then load the matching safetensors
state dict. The frozen receiver's input-token embeddings are used to represent event text, so the
same Qwen base model and tokenizer are required.

## Result and limitation

On task-disjoint MBPP validation traces, the contrastive adapter produces lower true-latent patch
loss (0.25821) than a different-task shuffled latent (0.25886), but the gap is only 0.00065 versus
the training margin of 0.1. Generated outputs are exactly identical for seven of eight sampled
tasks. Thus the adapter mostly acts as a generic repair soft prompt; it has not passed the
true-versus-shuffled, instance-specific evidence gate.

See the [source repository](https://github.com/danny2507/trace2cache-pilots), especially
`docs/handover_2026-09-14.md`, for the next experimental controls and claim discipline.
