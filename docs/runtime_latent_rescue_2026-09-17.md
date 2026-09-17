# Runtime latent rescue experiments, 2026-09-17

Follow-up: [the full-data run](runtime_latent_full_data_run_2026-09-17.md) trains on
1,536 views and executes the full 240-view development panel. Continuous repair reaches
53.75%, fixed-code projection 69.58%; the mechanism gate is still unmet. Read its
handover for the next reader-alignment and typed-binding steps.

This report supersedes the premature D0 no-go interpretation in
`runtime_latent_implementation_status_2026-09-17.md`. The earlier 400-update budget missed
an encoder that was still improving. Nearest-code accuracy after decoder training also
cannot substitute for executed repair evaluation: a decoder-trained representation may
depart from the oracle dictionary while remaining useful.

## Measured rescue

The existing frozen-Qwen contextual extractor and 4,078,848-parameter event encoder were
kept. All training uses the same 24 matched pairs / 48 views and seed 401. No labels, patch
sources or diagnoses are supplied as evidence to the encoder. Oracle codes remain training
supervision; this study has only 24 known behaviors across 12 known programs.

| Total vector updates | Learning-rate stages | Train code identity | Train cosine |
|---|---|---:|---:|
| 400 | 400 at 1e-3 | 33/48 (68.75%) | 0.7083 |
| 1,200 | then 800 at 3e-4 | 47/48 (97.92%) | 0.9737 |
| 2,000 | then 800 at 1e-4 | 48/48 (100%) | 0.9997 |

Each continuation loads the preceding encoder and resets AdamW, matching the existing
fork mechanism. The two 800-update continuations took 66.1 and 60.5 seconds of adapter
training, each at peak allocated GPU memory 3.58 GiB. This excludes receiver loading and
feature preparation. All neural work was on the A100; no systemwide changes were made.

On a predefined balanced subset of the training panel (one pair per family, 24 views),
the 1,200-update continuous encoder gave:

- True evidence: 18/24 repairs pass intended public and hidden tests; 7/12 complete pairs.
- Same-pair swapped evidence: 2/24 pass the original intended tests; 18/24 pass the
  opposite behavior tests. True-minus-swap is 66.7 percentage points.

This is executed behavior control, not patch-string difference. It is a training-panel
mechanism result, not unseen-input or unseen-program generalization. Residual fidelity
failures existed despite near-correct oracle classification: for example, the predicted
max code could decode into a min patch. This motivated the smaller-LR continuation.

## Implementation

`evaluate_paired_trace_encoder.py` now saves resumable JSONL rows containing response,
extracted source, intended and opposite hidden-test outcomes, cap hits and checkpoint/data
hashes. It evaluates both sides of each selected matched pair. It supports true, same-pair
swap, nearest-oracle diagnostics, no evidence, oracle, expected/status removal and no-role
conditions.

An optional `vector_identity` objective adds cosine-softmax classification over frozen
oracle codes, with labels confined to training supervision. Its gradient/dictionary tests
pass. It has **not** been used for the successful rescue reported above, and does not
establish a transferable dictionary for unseen programs.

Feature packs reduce repeated small-file reads on the network filesystem. A pack is keyed
by the ordered individual extractor cache digests; key equality, shape and finite values
are checked on load. Cache artifacts remain local and ignored by Git.

## Full training-panel repair and controls

The 2,000-update encoder was evaluated on all 24 training pairs / 48 views with greedy
generation, the same visible prompt and a 192-token output cap. Every generated patch was
validated against the intended and opposite public/hidden test suites.

| Condition | Intended repairs | Both sides correct | Opposite repairs |
|---|---:|---:|---:|
| True continuous latent | 42/48 (87.5%) | 19/24 (79.2%) | 1/48 |
| Same-pair swapped latent | 1/48 (2.1%) | 0/24 | 42/48 (87.5%) |
| Expected and status removed | 8/48 (16.7%) | 0/24 | 8/48 |
| Role IDs replaced by one ID | 6/48 (12.5%) | 0/24 | 7/48 |

True-minus-swap is 85.4 percentage points. The train-panel method uses the supplied values
and role metadata, rather than improving only a generic prompt. Metadata removal does not
erase all natural role clues: return values and source statements remain in payloads.
This is evidence that role metadata matters to this trained encoder, not exact source/
dependency binding or a guarantee of semantic meanings for the eight unrestricted queries.

One true and one swap generation hit the token cap. Remaining errors include invalid
syntax and wrong behavior despite correct nearest-oracle classification. Thus perfect
code identity alone is insufficient to establish continuous-interface repair fidelity.

On the same balanced 24-view subset, continuous true latents, nearest-oracle projection
and exact oracle latents each repair 22/24. The current eager receiver's oracle failures
are `locate_last` and `measure_even`; the historical 23/24 capacity was not reproduced in
this configuration. Thus an oracle syntax failure must not be attributed to the encoder.
The second training bundle for `extreme` accounts for two further continuous-latent
failures on the complete 48-view panel.

Padding invariance was also audited on 24 views: singleton-vs-padded-batch maximum
absolute state difference is 2.46e-7, RMSE 2.90e-8, and nearest-code disagreements 0/24.
Repair-time singleton encoding therefore matches training-panel batched encoding to
floating-point precision.

## Disjoint development evidence

One saved development pair per family was evaluated (12 pairs / 24 views). These are new
input bundles in the short training-value domain for the **same twelve programs**, not
new programs or disjoint value magnitudes. Input-hash overlap with actual training is zero.

| Condition | Intended repairs | Both sides correct | Opposite repairs |
|---|---:|---:|---:|
| True continuous latent | 11/24 (45.8%) | 2/12 (16.7%) | 3/24 |
| Same-pair swapped latent | 3/24 (12.5%) | 0/12 | 11/24 |
| Nearest-oracle projection | 14/24 (58.3%) | 4/12 (33.3%) | 4/24 |

Nearest-code identity on true development evidence is 16/24 (66.7%). Padding audit again
passes (max_abs 7.38e-7, zero code disagreements). Projection improves executed repair by
three views, isolating a continuous-interface fidelity issue, while remaining below the
mechanism gate. The dictionary contains only known behaviors and projection must not be
presented as a method for unseen program repair.

## Decision and concrete rescue directions

The claim that this encoder cannot learn was premature: the training-panel code gate and
executed evidence control are now demonstrated. The evidence-transfer gate is still unmet.
More updates on the same 48 views are insufficient evidence of generalization.

Recommended next steps, in order:

1. Train against the existing audited full training set: 768 pairs / 1,536 views, 10,588
   unique neutral payloads and maximum 62 events per view. Evaluate full saved development
   pairs, both orientations, continuous and swapped evidence. Use matched schedules/data
   exposure for baselines and select on development evidence, not test results. No LLM
   data generation is needed for this expansion.
2. If transfer stays weak, implement generic typed values and separate sequence items with
   parent/item positions. Currently an entire INPUT list is one neutral JSON payload
   pooled to one frozen contextual vector before the trainable event encoder sees it.
   Explicit item records and exact number fields can preserve the comparisons/order this
   task needs. Test equality, order and value probes before repair. Do not add family-specific
   sum/product/min/max features or label-derived evidence.
3. Improve reader alignment conservatively from the successful vector warmup. Test teacher
   logit distillation or differing-token ranking with a fidelity penalty and lower decoder
   learning rate, rather than interpreting oracle classification as repair. These are
   proposed experiments, not measured improvements. Keep the known-code projection as a
   diagnostic until task-disjoint repair removes the closed behavior dictionary.

Do not spend replication seeds or start MBPP/KV writing on this small-panel validation
result. IO-only/output-only controls are also needed before attributing gains to intermediate
runtime semantics rather than communicated behavioral specifications. Existing toy buggy
functions often have little informative intermediate execution.

Before a main matrix, fix a separate sampler defect: current median new-value generation
also changes its item count from (4,6) to (8,10). Thus the existing median value-new split
does not isolate the value axis. Do not use that split to claim pure new-value robustness.
Neither the rescue training nor short/in-range development evaluation uses that split.

## Reproduce the continuations

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python \
  scripts/train_paired_trace_encoder.py --objective vector --steps 800 \
  --learning-rate 3e-4 \
  --init-checkpoint checkpoints/paired_runtime_v2/overfit_context2_vector_seed401.pt \
  --feature-cache .local/cache/paired_runtime_v2/features_overfit_context2 \
  --checkpoint checkpoints/paired_runtime_v2/rescue_vector1200_seed401.pt \
  --output artifacts/paired_runtime_v2/rescue_vector1200_seed401.json

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python \
  scripts/train_paired_trace_encoder.py --objective vector --steps 800 \
  --learning-rate 1e-4 \
  --init-checkpoint checkpoints/paired_runtime_v2/rescue_vector1200_seed401.pt \
  --feature-cache .local/cache/paired_runtime_v2/features_overfit_context2 \
  --checkpoint checkpoints/paired_runtime_v2/rescue_vector2000_seed401.pt \
  --output artifacts/paired_runtime_v2/rescue_vector2000_seed401.json
```

Historical checkpoints and artifacts are preserved. These experiments were committed
before training. No GitHub/Hugging Face publication was performed in this rescue task.

## Saved artifacts

- `artifacts/paired_runtime_v2/rescue_vector1200_seed401.json`
- `artifacts/paired_runtime_v2/rescue_vector2000_seed401.json`
- `artifacts/paired_runtime_v2/rescue_vector1200_train_repair/{rows.jsonl,summary.json}`
- `artifacts/paired_runtime_v2/rescue_vector2000_train_repair/{rows.jsonl,summary.json}`
- `artifacts/paired_runtime_v2/rescue_vector2000_oracle_diagnostic/{rows.jsonl,summary.json}`
- `artifacts/paired_runtime_v2/rescue_vector2000_dev_repair/{rows.jsonl,summary.json}`

Local ignored trainable-only checkpoints are
`checkpoints/paired_runtime_v2/rescue_vector{1200,2000}_seed401.pt`. Receiver, feature caches
and frozen model weights are not included in these checkpoints.
