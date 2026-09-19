# Stage C execution plan — 2026-09-18

## Purpose

Determine whether the paired-runtime gain comes from intermediate execution events, from I/O
behavioral specification alone, or from an unfair text interface. Do this before interpreting the
known-family diagnostic as a general program-repair method.

## Existing evidence and its limits

On the fixed 24-view panel, the parent-aggregate full-runtime latent model reaches 19/24 intended
repairs and 2/24 paired-swap repairs. The separately trained I/O-only latent model reaches 15/24
and 2/24. Frozen raw structured JSON text reaches 2/24 in both conditions.

These numbers are directional only. The full-runtime and I/O-only models start from a checkpoint
already trained on full runtime, and the raw JSON text uses numeric role IDs without an explanatory
text interface. Neither comparison alone isolates the desired mechanism.

## Phase C1 — Fix and measure the text interface

1. Serialize roles as names (`INPUT`, `ACTUAL`, `EXPECTED`, `STATUS`, `BRANCH`, and so on), not
   numeric IDs.
2. Add a compact behavioral text form that explicitly groups each test's input, observed output,
   expected output, status, and ordered runtime events.
3. Evaluate no evidence, true text, and paired-swap text under identical prompts, greedy decoding,
   output limit, policy evaluator, and fixed panel.
4. Record input-token counts and failure categories.

Decision: if named/compact text becomes evidence-sensitive, retain it as the primary text baseline.
If it remains weak, report that the frozen receiver does not reliably consume this textual runtime
serialization; do not call that a general latent-over-text result.

## Phase C2 — Train matched full-runtime and I/O-only models from the same initialization

1. Use the same untrained/plain encoder initialization, dataset, 200-update budget, seed, pair
   sampler, optimizer, decoder objective, and frozen receiver for both conditions.
2. Full-runtime sees all events. I/O-only sees only test identity, inputs, actual output, expected
   output, and status throughout training, including warm-up.
3. Run three seeds after a one-seed smoke. Evaluate a fixed development panel and a held-out
   task/source panel with true and paired-swap evidence.
4. Report task-cluster confidence intervals, exact output agreement, Repair@1, complete-pair rate,
   true-minus-swap, event/token counts, and end-to-end runtime.

Decision: if I/O-only matches full runtime, narrow the claim to compressed behavioral-specification
communication. If full runtime wins consistently, proceed to C3.

## Phase C3 — Test runtime bindings, not merely extra tokens

1. Keep I/O events bit-identical while replacing intermediate runtime payloads and source lines
   with a deterministic corruption marker.
2. Add a binding permutation: retain each intermediate payload but permute its source/test
   association within an execution while preserving event count and role histogram.
3. Train/evaluate full-runtime, corrupted-runtime, and permuted-binding conditions independently.
4. Audit that I/O hashes are equal and that only declared runtime fields changed.

Implementation note (2026-09-19): `runtime_test_bindings_permuted` keeps every I/O event
bit-identical and keeps every runtime row's event id, test id, step and role. Within each runtime
role, it interleaves events by per-test rank and rotates their `(payload, source_line)` packets.
This preserves runtime-packet multiplicity and role/event histograms while breaking the association
between a runtime packet and the receiving test execution. Across the 1,536 training views,
15,026/15,032 intermediate packets change; the remaining six have no same-role cross-test donor.

Follow-up (2026-09-19): `runtime_temporal_bindings_permuted` is the stronger within-test control.
It retains each test's exact multiset of intermediate `(payload, source_line)` packets, plus all
I/O and row metadata, but rotates packets between runtime steps. All 15,032 training runtime
packets change receiver step under this transform.

Decision: a full-runtime advantage that disappears under either intervention supports use of
runtime associations. If it survives both, investigate generic trace-format or prompt effects.

## Phase D — Task/source-disjoint repair

1. Move to mutated MBPP with every mutant, fuzz input, and augmentation from a task held in one
   split only.
2. Remove closed 24-behavior codebook supervision. Train from verified canonical patch targets and
   task-disjoint evidence.
3. Reserve original MBPP test IDs and MBPP+ test-only IDs for final hidden-test evaluation.
4. Compare full-runtime latent, I/O-only latent, best text baseline, and no evidence across three
   seeds. Validate all patches on held-out tests.

The project can claim general program-repair utility only after this phase shows a replicated
task-disjoint gain with an evidence-corruption sensitivity check.
