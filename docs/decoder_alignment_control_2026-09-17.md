# Matched decoder-alignment diagnostic, 2026-09-17

## Question

The full-data continuous encoder has a 15.83-point gap to nearest fixed-code projection
on the same 240 development views. Test whether a small receiver-facing loss improves
continuous latent decoding while preserving evidence sensitivity.

This is a **diagnostic fork**, not the official Phase-D1 matrix and not a claim of
unseen-program generalization. It starts from the committed
`full_data_vector2000_seed401.pt` checkpoint. Both branches use the same context2
features, frozen eager Qwen2.5-1.5B receiver, role-aware encoder, eight states, seed 401,
training data and 200 updates at 3e-5.

Each update samples one family then the same ordered sequence of eight paired records with
replacement. Vector control trains both A/B views with vector loss. Decoder branch trains
the same A/B views in two batched receiver forwards using full target NLL + paired positive
vs negative rank + existing 0.1 vector auxiliary. Each result records a SHA-256 trace of
the sampled pair UIDs; the two digests must match before comparing outcomes.

## Commands

```bash
.venv/bin/python scripts/train_paired_trace_encoder.py \
  --dataset .local/datasets/paired_runtime_v2/train.jsonl \
  --validation-dataset .local/datasets/paired_runtime_v2/dev.jsonl \
  --init-checkpoint checkpoints/paired_runtime_v2/full_data_vector2000_seed401.pt \
  --objective vector --sampling paired_family --steps 200 --learning-rate 3e-5 \
  --batch-size 16 --gradient-accumulation 8 \
  --checkpoint checkpoints/paired_runtime_v2/alignment_vector200_seed401.pt \
  --output artifacts/paired_runtime_v2/alignment_vector200_seed401.json

.venv/bin/python scripts/train_paired_trace_encoder.py \
  --dataset .local/datasets/paired_runtime_v2/train.jsonl \
  --validation-dataset .local/datasets/paired_runtime_v2/dev.jsonl \
  --init-checkpoint checkpoints/paired_runtime_v2/full_data_vector2000_seed401.pt \
  --objective decoder --sampling paired_family --steps 200 --learning-rate 3e-5 \
  --batch-size 16 --gradient-accumulation 8 \
  --checkpoint checkpoints/paired_runtime_v2/alignment_decoder200_seed401.pt \
  --output artifacts/paired_runtime_v2/alignment_decoder200_seed401.json
```

After confirming equal sample digests, evaluate both checkpoints on the fixed first
development pair per family (24 views), with `true_latent paired_swap`, greedy cap 192.
Do not select a best intermediate checkpoint. Expand only a promising branch to all 120
development pairs, preserving the existing test split.

## Decision

Decoder wins only if it improves executed true repair and true-minus-swap or complete
pairs over the matched vector fork, without a large collapse in nearest-code identity.
If it mainly improves syntax but not swapped behavioral control, report reader alignment
as the gain and continue with typed item/value binding. If both are weak, do not add
uncontrolled epochs: implement Phase F1 probes and representation first.
