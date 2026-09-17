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

## Measured result

The vector and decoder forks have the same sampled-pair trace SHA-256:
`393d9fcca6831ddb88ecf53006fed2acf323a2d84ae5866806925a4a21b4d9e9`.
Thus the objective comparison is not explained by a different family/pair stream.

| Fork, 200 updates | Dev code identity | Fixed 24-view true repair | Complete pairs | True-minus-swap |
|---|---:|---:|---:|---:|
| Vector control | 81.25% | 9/24 | 3/12 | 16.67 points |
| Decoder NLL + rank + vector | 80.42% | 18/24 | 7/12 | 62.50 points |

The decoder fork is selected by this prespecified fixed panel and then evaluated on the
complete 120-pair development set. It obtains 170/240 true repairs (70.83%), 60/120
complete pairs (50.00%), 46/240 swap-intended repairs (19.17%), and 172/240 swapped
opposite repairs (71.67%). The family-cluster bootstrap intervals are: true repair
[57.50%, 84.58%], complete pair [28.33%, 74.17%], true-minus-swap [27.92, 75.83]
points, and swap opposite [59.17%, 85.00%]. Two true runs time out; no true run hits
the 192-token cap. The predeclared repair and complete-pair gates remain unmet.

On the strict 99-pair subset with no individual test input in training: true repair is
139/198 (70.20%), swap intended repair 36/198 (18.18%), complete pairs 49/99 (49.49%),
and true-minus-swap 52.02 points. This is evidence of new-bundle use in the same known
toy programs, not task-disjoint repair. The dev panel was used for selecting decoder,
so do not use it as an unbiased final method comparison.

The gain supports the reader-alignment hypothesis: direct decoder loss repairs many
syntax/receiver-fidelity failures while preserving a behavioral swap intervention.
It does not establish the proposed model-native runtime interface as better than text,
causal trace use, role-slot semantics, or generalization to unseen programs. `arrange`,
`median`, `rotate`, `unique` and one filter direction remain weak, motivating Phase F1
typed ordered items rather than more unstructured decoder epochs.

## Typed-value continuation (negative representation control)

The Phase-F1 implementation expands the existing safe tagged JSON evidence into ordered
virtual value nodes.  It preserves list/tuple item order and uses generic type features:
type, signed-int bits and magnitude, bool, bounded string sketch, container size, depth,
and child index.  It neither evaluates a `repr` nor receives any family-specific diagnosis
or patch label.  New nodes retain the parent event role/test metadata.  The value projection
is zero-initialized, so at update zero the typed encoder is exactly the committed pre-typed
encoder; gradients alone can enable the new channel.

It uses the **same** 200-update decoder objective and matched sampling digest as the two
forks above.  Its final development code identity is lower (74.58%, versus 80.42% for the
decoder baseline).  On the selected 24-view development panel it has 18/24 true repairs,
7/12 complete pairs and 4/24 paired-swap intended repairs: a 58.33-point true-minus-swap
gap, slightly below the baseline's 62.50 points.  Thus raw generic typed features do not
improve the selected decoder mechanism.

As an out-of-selection check, one first pair per family was evaluated on `test_short`:
disjoint runtime-input bundles but the same 12 toy program families.  The baseline obtains
11/24 true, 6/24 swap-intended, 3/12 complete pairs (20.83-point gap).  Typed obtains
13/24 true, 8/24 swap-intended, 3/12 complete pairs (also 20.83 points).  Its two extra
true repairs are exactly offset by two extra swap-following repairs.  This is not evidence
of better role-aware behavioral binding, so do not scale this representation alone to
additional epochs or the full held-out split.  The next method should make the required
behavioral contrast explicit in the objective/architecture, not merely append value bits.
