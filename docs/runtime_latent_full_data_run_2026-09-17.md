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
  --development-dataset .local/datasets/paired_runtime_v2/dev.jsonl \
  --reference-dir artifacts/paired_runtime_v2/rescue_vector2000_dev_repair
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
Also compare the new run against the earlier 48-view encoder on exactly its saved
12-pair development panel, not its 24-view counts against the larger 240-view result.

## Measured training result

All three stages completed on the A100. The receiver is frozen and only the existing
4,078,848-parameter encoder is trained. The development identity score is not executed
repair accuracy.

| Total updates | Train nearest-code identity | Development nearest-code identity | Development cosine |
|---|---:|---:|---:|
| 400 | 586/1,536 (38.15%) | 94/240 (39.17%) | 0.5713 |
| 1,200 | 1,075/1,536 (69.99%) | 168/240 (70.00%) | 0.8143 |
| 2,000 | 1,170/1,536 (76.17%) | 183/240 (76.25%) | 0.8549 |

Stage training-loop times, including periodic train/development identity evaluation,
were 69.9, 128.4 and 130.6 seconds. Training feature preparation took 48.5 seconds on
the first stage and 1.5 / 0.9 seconds on later cache-pack loads. These feature times
exclude separate development feature preparation and receiver loading; they are not
end-to-end wall times. Peak allocated GPU memory was 8.72 GiB. Tests: 41 passed.

Development code identity briefly reached 193/240 at total update 1,850 but the
prespecified final checkpoint is used for repair. No best-checkpoint/test selection
is performed. Small train/development identity gaps are encouraging for known-program
evidence transfer, but cannot establish correct continuous-latent decoding.

## Executed primary development result

The prespecified final checkpoint was evaluated on all 120 saved development pairs,
both orientations. Receiver configuration, prompt and 192-token cap match the earlier
rescue evaluation. No test split was touched.

| Condition | Intended repairs | Complete correct pairs | Opposite repairs | Cap hits |
|---|---:|---:|---:|---:|
| True continuous latent | 129/240 (53.75%) | 41/120 (34.17%) | 34/240 | 10 |
| Same-pair swapped latent | 34/240 (14.17%) | 3/120 (2.50%) | 129/240 | 10 |

True-minus-swap is 39.58 percentage points. The family-cluster bootstrap 95% interval
is [16.25, 62.92] points; true-repair interval is [35.42%, 72.08%]. These intervals use
only twelve known program groups and are not evidence of unseen-program generalization.
The full numerical mechanism gate is **not passed**: repair, complete-pair success and
opposite selection remain below 75%, 60% and 60%, respectively. The gap criterion passes.
Baseline replication and runtime-vs-IO utility have not been established.

On the input-defined strict subset (99 pairs, no individual test input seen in train),
true repairs are 111/198 (56.06%), swap intended repairs 31/198 (15.66%), and true
complete pairs 36/99 (36.36%). This preserves an evidence-use signal without repeated
individual input tests, but still uses the same programs and value domain.

On exactly the earlier 12-pair/24-view dev panel, the earlier 48-view encoder repaired
11/24 with 2/12 complete pairs; the new full-data encoder repairs 12/24 with 4/12 pairs.
Nearest identity changes 16/24 -> 17/24. Thus data expansion alone produces only a small
measured downstream change on this matched panel, not a demonstrated large improvement.
Equal updates also mean much less exposure per example in the larger training set.

| Family | Correct nearest code | True repairs | Complete pairs | True-minus-swap (points) |
|---|---:|---:|---:|---:|
| aggregate | 18/20 | 19/20 | 9/10 | 95 |
| arrange | 12/20 | 8/20 | 0/10 | -20 |
| choose | 20/20 | 20/20 | 10/10 | 100 |
| extreme | 19/20 | 13/20 | 4/10 | 45 |
| filter | 15/20 | 8/20 | 1/10 | 0 |
| locate | 20/20 | 10/20 | 0/10 | 50 |
| measure | 15/20 | 7/20 | 0/10 | 30 |
| median | 11/20 | 8/20 | 2/10 | 30 |
| rotate | 5/20 | 1/20 | 0/10 | -20 |
| transform | 18/20 | 18/20 | 8/10 | 80 |
| truth | 20/20 | 17/20 | 7/10 | 85 |
| unique | 10/20 | 0/20 | 0/10 | 0 |

64/240 views have a correct nearest-code identity but fail continuous repair. Identity
must not replace executed repair. Of 111 failed true repairs, 66 reached execution and
failed tests, 22 were not parseable, and 23 were rejected by the existing conservative
sandbox policy. Policy rejections are not automatically semantic errors; the validator
was not loosened after seeing results. For example, `list(set(values))` is rejected and
does not guarantee the order-preserving semantics needed by `unique`.

Padding audit passes on all 240 views: maximum absolute state difference 3.61e-7,
RMSE 2.49e-8, and zero nearest-code disagreements. No development role-removal run,
matched no-role training baseline or claim that all eight unrestricted queries have
fixed semantic roles is established by this experiment.

## Follow-up fidelity diagnostic

After the primary true/swap repair run, evaluate `nearest_oracle_latent` and `oracle_latent`
on the full saved development panel. This is a diagnostic, not a deployed method or an
unseen-task result. The dictionary has only 24 known behaviors.

```bash
.venv/bin/python scripts/evaluate_paired_trace_encoder.py \
  --checkpoint checkpoints/paired_runtime_v2/full_data_vector2000_seed401.pt \
  --dataset .local/datasets/paired_runtime_v2/dev.jsonl \
  --pairs-per-family 10 --conditions nearest_oracle_latent oracle_latent \
  --cache-fixed-codes \
  --output-dir artifacts/paired_runtime_v2/full_data_vector2000_dev_projection

.venv/bin/python scripts/analyze_paired_transfer.py \
  --evaluation-dir artifacts/paired_runtime_v2/full_data_vector2000_dev_repair \
  --train-dataset .local/datasets/paired_runtime_v2/train.jsonl \
  --development-dataset .local/datasets/paired_runtime_v2/dev.jsonl \
  --reference-dir artifacts/paired_runtime_v2/rescue_vector2000_dev_repair \
  --projection-dir artifacts/paired_runtime_v2/full_data_vector2000_dev_projection
```

The optional decoding cache is restricted to fixed oracle/projected codes and exact
visible prompts. On every cache hit, the entire spliced receiver input tensor must be
exactly tensor-equal to its cached copy (`torch.equal`, no tolerance); otherwise evaluation aborts. Continuous true/swap
states are never cached. Generated patches are still validated against each row's
intended and opposite tests. Cache hits are saved in rows and summary. Toy-program
duplicate decoding savings must not be advertised as a deployment latency improvement.

### Measured fidelity result

| Interface | Intended repairs | Complete pairs | Opposite repairs |
|---|---:|---:|---:|
| Continuous encoder | 129/240 (53.75%) | 41/120 (34.17%) | 34/240 |
| Nearest fixed oracle code | 167/240 (69.58%) | 55/120 (45.83%) | 37/240 |
| Exact target oracle code | 220/240 (91.67%) | 100/120 (83.33%) | 0/240 |

Projection gains 48 repairs and loses 10, a net gain of 38/240 (15.83 points).
Its paired family-cluster bootstrap 95% gain interval is [7.08, 25.00] points.
This isolates a substantial fidelity limitation, **not** an unseen-program quantization
method: the inference dictionary is closed and learned against these known behaviors.
The oracle run accesses the intended label by design and is a capacity diagnostic only.
Projection also stays below the mechanism gate, so reader alignment alone is not known
to be sufficient. Input/order representation remains a separate problem.

Exact oracle failures are the same `locate_last` and `measure_even` failures as before,
repeated over ten bundles each. They must not be attributed to the trace encoder.
Projection rescues `unique` from 0/20 to 10/20, `filter` from 8/20 to 15/20, and
`extreme` from 13/20 to 19/20. `rotate` only improves 1/20 -> 5/20; its code identity
is already weak at 5/20. These distinguish fidelity from representation deficiencies.

The diagnostic used 29 distinct fixed-code/prompt combinations and 451 exact-input
cache hits across 480 rows. Every row was validated against its own intended/opposite
tests. Primary continuous decoding had no cache. Padding again passes on all 240 views.

## Handover: next implementation, not yet run

This is a single-seed data-expansion/fidelity rescue, not the completed Phase-D1 2x2
matrix from the implementation plan. Do not claim text is an inferior medium, useful
intermediate runtime semantics, disjoint-program repair or transferable semantic slots
from these results. The data is automatically generated; GPT data generation was not
needed. No changes were made systemwide and no GitHub/HF publication was attempted.

1. **Reader alignment first.** Fork the saved `full_data_vector2000_seed401.pt` into a
   conservative decoder-loss diagnostic and a matched vector-continuation control.
   Implement a shared paired-family sampler first: choose the same family and same
   eight A/B pairs per update in both forks (16 effective views). The current vector
   sampler is unpaired, so it is not yet a matched-objective control. Keep native
   features/receiver frozen, eight states and seed 401. A concrete initial diagnostic
   budget is 200 updates at LR 3e-5, with saved metadata and actual executed dev repairs.
   Use full target NLL for syntax and paired ranking for behavior; consider a separately
   named differing-token rank mask and stronger vector-fidelity weight, not an
   undocumented change to the existing `decoder` loss. Select/report on dev complete
   pairs, not nearest-code score. This lower LR/budget is a proposed rescue configuration,
   not an already measured improvement or the official D1 matrix schedule.
2. **Generic typed/item binding next if identity stays weak.** Follow Phase F1 of the
   implementation plan: signed-int64 sign/bits plus bounded magnitude, type tags,
   separate ordered sequence-item records with parent/test/item IDs, and a small value
   projection alongside native content vectors. Probe equality/order/value information
   before repair. Do not insert family-specific arithmetic, patch IDs or diagnoses.
   Current hard families are rotate, unique, median and arrange. Change one factor at
   a time with the same saved training data and matched exposures.
3. **Role and medium controls remain necessary.** Test expected/status removal and
   role-ID swap/removal on disjoint dev, then train a matched no-role baseline. Compare
   structured text and IO-only evidence before attributing gains to runtime traces.
   Many toy buggy functions are constant-return stubs with little intermediate signal.
4. **Hold external benchmarks.** Fix the known median value-new length confound before
   new-value testing; do not overwrite historical saved data. Run replication seeds,
   task-disjoint MBPP repair and direct KV interfaces only after the relevant gates.

Latest verification: 44 unit tests passed in 7.90 seconds; `git diff --check` passed.
All neural jobs finished and released
the A100. Checkpoints/base weights/caches remain project-local and ignored by Git;
code, row-level results, summaries and this report are tracked.
