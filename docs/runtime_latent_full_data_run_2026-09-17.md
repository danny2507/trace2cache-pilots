# Full-data continuation, 2026-09-17

## Prespecified experiment

Continue the rescue using the existing audited full v2 dataset, not an LLM-generated
replacement. Keep the contextual extractor, role-aware encoder, eight continuous soft
states, frozen eager Qwen2.5-1.5B-Instruct receiver, oracle vector loss and seed 401.
Start a fresh encoder; do not initialize from the memorized 48-view rescue.

- Train: `.local/datasets/paired_runtime_v2/train.jsonl`, 768 pairs / 1,536 views.
- Development: `.local/datasets/paired_runtime_v2/dev.jsonl`, 120 pairs / 240 views.
- Learning rates: 400 updates at 1e-3, then 800 at 3e-4, then 800 at 1e-4.
- Each continuation loads the previous encoder and resets AdamW and the sampling RNG,
  matching the earlier rescue mechanism. Batch size remains 16 views.
- Feature cache: existing project-local `.local/cache/paired_runtime_v2/features`.
- Save separate checkpoints/results named `full_data_vector{400,1200,2000}_seed401`.
- Development nearest-code accuracy/cosine is diagnostic only. Development examples
  never enter optimizer batches. Check train/development input-hash disjointness before
  loading the model and record both file hashes in each checkpoint/result.
- At the fixed final 2,000-update checkpoint, execute true and same-pair-swapped latent
  repairs on all saved development pairs, both orientations, greedy decode, cap 192.
  Report intended repairs, complete pairs, opposite repairs and caps. Keep the same
  hidden repair validation suites as the earlier pilot.

No test split is used for training or selection here. This measures transfer to new
input bundles for the same twelve toy programs, **not unseen-program repair**. The
oracle dictionary is a closed set of 24 behaviors. More data at the same update budget
also lowers exposure per example; this is not a compute-matched convergence claim.

## Reproduction

Run from the project root using `.venv/bin/python`, with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. All neural operations require CUDA.

```bash
.venv/bin/python scripts/train_paired_trace_encoder.py \
  --dataset .local/datasets/paired_runtime_v2/train.jsonl \
  --validation-dataset .local/datasets/paired_runtime_v2/dev.jsonl \
  --steps 400 --learning-rate 1e-3 \
  --checkpoint checkpoints/paired_runtime_v2/full_data_vector400_seed401.pt \
  --output artifacts/paired_runtime_v2/full_data_vector400_seed401.json

.venv/bin/python scripts/train_paired_trace_encoder.py \
  --dataset .local/datasets/paired_runtime_v2/train.jsonl \
  --validation-dataset .local/datasets/paired_runtime_v2/dev.jsonl \
  --steps 800 --learning-rate 3e-4 \
  --init-checkpoint checkpoints/paired_runtime_v2/full_data_vector400_seed401.pt \
  --checkpoint checkpoints/paired_runtime_v2/full_data_vector1200_seed401.pt \
  --output artifacts/paired_runtime_v2/full_data_vector1200_seed401.json

.venv/bin/python scripts/train_paired_trace_encoder.py \
  --dataset .local/datasets/paired_runtime_v2/train.jsonl \
  --validation-dataset .local/datasets/paired_runtime_v2/dev.jsonl \
  --steps 800 --learning-rate 1e-4 \
  --init-checkpoint checkpoints/paired_runtime_v2/full_data_vector1200_seed401.pt \
  --checkpoint checkpoints/paired_runtime_v2/full_data_vector2000_seed401.pt \
  --output artifacts/paired_runtime_v2/full_data_vector2000_seed401.json

.venv/bin/python scripts/evaluate_paired_trace_encoder.py \
  --checkpoint checkpoints/paired_runtime_v2/full_data_vector2000_seed401.pt \
  --dataset .local/datasets/paired_runtime_v2/dev.jsonl \
  --pairs-per-family 10 --conditions true_latent paired_swap \
  --output-dir artifacts/paired_runtime_v2/full_data_vector2000_dev_repair

.venv/bin/python scripts/analyze_paired_transfer.py \
  --evaluation-dir artifacts/paired_runtime_v2/full_data_vector2000_dev_repair \
  --train-dataset .local/datasets/paired_runtime_v2/train.jsonl \
  --development-dataset .local/datasets/paired_runtime_v2/dev.jsonl
```

Code/protocol must be committed before starting training. Results will be appended
after execution; no improvement is assumed in advance.

## Additional overlap audit before repair evaluation

The train/development bundle-hash overlap is zero. However, individual tests need not be
disjoint when generated from this small value domain: 23 of 376 development test
occurrences appear in training (23 unique repeated inputs). There are 99 development
pairs whose **every** individual test input is unseen in training. Save both the full
development result and this stricter subset. The subset is defined from input overlap,
not repair outcomes. Use a 2,000-resample family-cluster bootstrap with seed 401; twelve
program groups imply substantial uncertainty and limited external validity.
