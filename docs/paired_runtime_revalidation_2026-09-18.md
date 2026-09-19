# Paired-runtime evaluator revalidation — 2026-09-18

This report re-scores saved responses only.  It does not load a checkpoint, generate a new token,
edit a patch, alter a test suite, or overwrite a historical artifact.

## Policy

The historical evaluator applied a single three-second parent-process deadline covering Python
startup, worker import, candidate compilation, and candidate execution.  The revalidation runner
uses an explicit ready handshake before candidate source is sent to the worker, then applies a
five-second startup allowance and a three-second candidate wall/CPU budget.  Candidate CPU,
memory, file-size, and file-descriptor limits remain enabled in the Linux worker.

The new evaluator returns mutually distinguishable outcomes:

- `extraction_failure` — no requested function could be parsed from the saved response;
- `policy_rejection` — extracted source violates the existing patch policy;
- `semantic_failure` / `semantic_pass` — worker completed every supplied test;
- `candidate_timeout` / `candidate_wall_timeout` — source exceeded execution budget after startup;
- `infrastructure_failure` — startup/protocol/worker fault.

An extracted patch is retained even for policy and execution failures.  The immutable input hashes,
execution policy and evaluator revision are recorded in
`artifacts/paired_runtime_v2_revalidation_20260918/manifest.json`; per-row response, source-row,
patch and test-suite hashes are stored next to each source run.  The aggregate is
`artifacts/paired_runtime_v2_revalidation_20260918/summary.json`.

## Coverage and result

All 2,136 historical response rows from 14 paired-runtime artifact directories were revalidated
once under this policy.  There were no `infrastructure_failure`, `candidate_timeout`, or
`candidate_wall_timeout` outcomes.  Thus the earlier combined parent deadline, rather than a
measured candidate resource violation, explains the affected historical timeout rows.

The matched 24-view development comparison is unchanged:

| Branch | True intended repair | Paired-swap intended repair |
| --- | ---: | ---: |
| Vector control | 9/24 | 5/24 |
| Decoder alignment | 18/24 | 3/24 |

On the saved full-dev decoder rows, true intended repair changes from 170/240 to 172/240; paired-
swap intended repair remains 46/240.  The two changed true rows are now `semantic_pass` under the
same saved response and test suite.  This is a revalidation result, not a fresh generation or a
newly selected branch.

On the saved `test_short` panel, true intended repair changes from 11/24 to 15/24, while
paired-swap intended repair remains 6/24.  The four changed true results were historical combined-
deadline timeouts that complete under the separated policy.  `test_short` still holds out only
input bundles, not source/program families, so it remains insufficient for a task-general repair
claim.

Historical extraction and policy failures remain in all denominators; they were not dropped or
converted to a successful repair.  For example, the fixed 24-view decoder panel contains two
policy rejections in each condition, leaving the executed result 18/24 versus 3/24 rather than a
filtered rate.

## Control repair

The typed-value path previously expanded each event into virtual nodes.  Even with its typed
projection zeroed, that changed event count, event positions and attention.  It therefore was not
a valid warm-start-equivalent ablation.

`parent_aggregate` is now the default typed mode.  It parses the same bounded typed items but
averages them into a side feature attached to the original parent event.  The base content, roles,
test identity, event mask, positions and attention graph remain identical.  A regression test
copies a nonzero trained-like plain encoder into the typed architecture and verifies bitwise equal
output at zero residual plus nonzero gradient into the typed projection.  `legacy_virtual_nodes`
remains available only to reproduce pre-fix checkpoints and is explicitly selected for those
checkpoint configurations.

Before the following controlled run, no typed, binding, or decoder model was retrained after this
repair.  The next GPU action was a small, predeclared plain-vs-parent-aggregate gradient/parity
smoke followed by a matched rerun; it did not reuse old virtual-node results as an architecture
conclusion.

## Post-repair parent-aggregate typed smoke and matched rerun

The planned gate was then run on the real frozen receiver and
`full_data_vector2000_seed401.pt` warm-start.  Eight development views had identical base event
inputs under plain and parent-aggregate collation, exactly zero latent difference at update zero,
and a nonzero first-update typed-projection gradient (`2.61e-4` maximum absolute value).  The GPU
parity artifact is `artifacts/paired_runtime_v2/typed_parent_aggregate_gpu_parity.json`.

A single matched decoder fork then used the same 768 training pairs, 200 updates, seed 401,
decoder NLL/ranking/vector objective, and sampling digest as the existing alignment comparison:

`393d9fcca6831ddb88ecf53006fed2acf323a2d84ae5866806925a4a21b4d9e9`.

Its predeclared first-development-pair-per-family evaluation used the immutable manifest generated
by the repaired evaluator:

| Branch | True intended repair | Paired-swap intended repair | Complete true pairs |
| --- | ---: | ---: | ---: |
| Plain decoder, revalidated | 18/24 | 3/24 | 7/12 |
| Parent-aggregate typed decoder | 19/24 | 2/24 | 8/12 |

The new run has no cap hits, two policy rejections in each condition, and no execution or
infrastructure timeouts.  Its singleton-versus-batched encoder audit has zero nearest-code
disagreements (`max_abs=1.97e-7`).  Result files are
`artifacts/paired_runtime_v2/typed_parent_aggregate_decoder200_seed401.json` and
`artifacts/paired_runtime_v2/typed_parent_aggregate_decoder200_dev_repair/`.

This is a one-seed, 12-known-family diagnostic.  The +1 true repair / -1 swap repair difference
is directional only and is not evidence that typed values generalize, improve unseen-program
repair, isolate intermediate runtime trace utility, or beat matched text.  The next valid action
is a predeclared task/source-disjoint evaluation with the Stage-C I/O and text medium controls,
not additional selection on this development panel.

## Stage-C I/O-only latent baseline

An independently trained I/O-only encoder was run with the same warm start, 200 decoder updates,
seed, pair sampler and sampling digest as the full-runtime parent-aggregate encoder. It receives
only test identity, inputs, actual output, expected output and derived status; it never receives
CALL/LINE/BRANCH/DEFINITION/RETURN events. Its fixed-panel executed result is `15/24` true,
`2/24` paired-swap and `6/12` complete pairs, compared with full runtime's `19/24`, `2/24` and
`8/12`. Both have two policy rejections per condition; no candidate/infrastructure timeout occurs.

Thus intermediate runtime events supply a +4 true-repair directional signal at unchanged swap
repair on this narrow panel. This does **not** isolate a latent-medium advantage: a structured-text
baseline trained/evaluated with matched evidence and a task/source-disjoint final split remain
required. Artifacts: `artifacts/paired_runtime_v2/io_only_decoder200_seed401.json` and
`artifacts/paired_runtime_v2/io_only_decoder200_dev_repair/`.

The frozen structured-JSON prompt baseline, using the identical full-runtime events and greedy
decoder but no latent states, obtains `2/24` true and `2/24` paired-swap repair (zero complete
pairs). It is therefore not evidence-sensitive on this panel. This makes the current ordering
full-runtime latent > I/O-only latent >> frozen structured text, but it is not a final claim that
latent beats text: the text receiver was not separately SFT/alignment-trained, and all results are
still one-seed known-family diagnostics. Artifact:
`artifacts/paired_runtime_v2/structured_text_dev_repair/`.

## Stage-C1 named and compact text interface

The original JSON serializer exposed numeric role IDs, so the frozen receiver had no textual role
legend.  C1 evaluates two stricter replacements using the same 24 views, greedy decoding and
policy evaluator: a lossless JSON serializer with names such as `EXPECTED` and `BRANCH`, and a
test-grouped compact form with the same event payloads.  The compact form lowers mean prompt size
from 1,592 to 1,108 tokens (30.4%); neither form produces evidence-sensitive repair:

| Condition | Intended repair | Paired-swap repair | Complete pairs |
| --- | ---: | ---: | ---: |
| No evidence | 4/24 | 4/24 | 0/12 |
| Named lossless text | 3/24 | 3/24 | 0/12 |
| Compact behavioral text | 2/24 | 3/24 | 0/12 |

This is a frozen-receiver text control, not a claim about what text could achieve after text-SFT.
It removes the specific numeric-schema objection to the earlier baseline and justifies moving to
the matched-from-scratch full-runtime versus I/O-only latent comparison.  Immutable generation
artifact: `artifacts/paired_runtime_v2/named_text_compact_dev_repair_v2/`.

## Stage-C2 matched from-scratch gate

The first C2 run uses a plain, untrained role-aware encoder for both evidence views (seed 401),
the same 768-pair decoder objective, 200 updates, sampler trace
`393d9fcca6831ddb88ecf53006fed2acf323a2d84ae5866806925a4a21b4d9e9`, frozen receiver and codebook.
Both initial encoder state hashes are exactly
`001a5b74bac2f31f5ed4a88c2b9d5740a623daf33877f1c25b32c5c5c4d274c8`.

This is a capacity/optimization gate failure, rather than a runtime-information comparison: both
branches finish at 1/24 nearest repair-code accuracy on train and validation (full-runtime cosine
0.145, I/O-only 0.173).  The earlier useful decoder used a full-runtime vector-aligned warm start;
200 decoder-only updates cannot learn the receiver alignment from a random adapter.  Therefore no
repair generation was run and no claim is made that I/O equals or exceeds full runtime.  Next: use
a *representation-neutral* pretraining stage or a shared text/latent alignment objective before
re-running this matched comparison.  Artifacts:
`artifacts/paired_runtime_v2/stage_c_from_scratch_full_seed401.json` and
`artifacts/paired_runtime_v2/stage_c_from_scratch_io_seed401.json`.

The first matched vector-only warm-up checkpoint (400 updates at 1e-3, paired sampler) also has
identical initialization and sampling hashes across views.  It remains below the identity gate:
full runtime reaches 0.341 cosine / 1/24 nearest-code accuracy and I/O-only 0.370 / 1/24.  This
shows that the historical `full_data_vector2000` warm start cannot be treated as a from-scratch
baseline: it was a continuation checkpoint.  The remaining predeclared warm-up schedule is 800
updates at 3e-4 and 800 at 1e-4 for **each** view, then the 200-update decoder stage; do not use
the 400-step numbers to compare runtime utility.  Artifacts:
`artifacts/paired_runtime_v2/stage_c_matched_full_vector400_seed401.json` and
`artifacts/paired_runtime_v2/stage_c_matched_io_vector400_seed401.json`.

After completing the predeclared 2,000-update vector schedule (400 @ 1e-3, 800 @ 3e-4,
800 @ 1e-4), the branches separate on the fixed code-identity probe:

| Evidence view | Train nearest code | Dev nearest code | Dev cosine |
| --- | ---: | ---: | ---: |
| Full runtime | 72.5/240 (30.2%) | 73/240 (30.4%) | 0.523 |
| I/O-only | 25.2/240 (10.5%) | 26/240 (10.8%) | 0.421 |

This is the first matched-from-scratch indication that intermediate events improve the encoder's
ability to recover the pre-existing repair-code targets. It is a one-seed representation probe,
not repair accuracy and not a latent-versus-text result. The next predeclared operation is the
same 200-update decoder alignment objective from each final vector checkpoint, followed by the
fixed true-versus-paired-swap repair evaluation. Artifacts:
`artifacts/paired_runtime_v2/stage_c_matched_full_vector2000_seed401.json` and
`artifacts/paired_runtime_v2/stage_c_matched_io_vector2000_seed401.json`.

The subsequent matched 200-update decoder stage and fixed first-pair-per-family repair
evaluation give the following one-seed result. Each row uses the same frozen receiver, decoder
loss, greedy cap, evaluator, development panel and true-versus-paired-swap intervention.

| Evidence view | True intended repair | Paired-swap intended repair | True − swap | Complete true pairs |
| --- | ---: | ---: | ---: | ---: |
| Full runtime | 9/24 | 6/24 | +3 | 2/12 |
| I/O-only | 4/24 | 8/24 | −4 | 0/12 |

The full-runtime adapter is padding-stable (`max_abs=1.27e-7`, zero nearest-code disagreements)
and the I/O adapter independently passes the same audit (`8.94e-8`, zero disagreements). Thus the
result is not explained by batching/padding. Relative to the matched I/O-only branch, full runtime
adds five true repairs, reverses the swap direction by seven repairs, and supplies the only
complete-pair successes. This supports a narrow **known-family, one-seed** claim that intermediate
runtime events matter to this learned latent adapter after matched from-scratch training. It does
not establish a text-medium advantage after text alignment, causal source/value binding, or
task/source-disjoint program-repair generalization. Artifacts:
`artifacts/paired_runtime_v2/stage_c_matched_full_decoder200_{seed401.json,dev_repair/}` and
`artifacts/paired_runtime_v2/stage_c_matched_io_decoder200_{seed401.json,dev_repair/}`.

## Stage-C3 runtime corruption control

The independently trained corruption branch preserves exact I/O rows and every runtime row's
position, test identity and role, but replaces each runtime payload with the same `CORRUPTED`
marker and removes its source line. It uses the same 2,000-vector plus 200-decoder schedule,
seed and evaluation panel as Stage C2. Despite reaching 29.0% train / 29.2% dev code identity
(close to full runtime's 30.2% / 30.4%), its repair intervention is `7/24` true and `7/24`
paired-swap, with zero complete pairs. Full runtime is `9/24` true, `6/24` swap and two complete
pairs. Thus generic runtime-event layout is enough for much of the vector probe, but it does not
reproduce the full branch's positive true-minus-swap signal (+3 versus 0). This is directional
one-seed evidence that payload/source content matters; the binding-permutation branch remains
necessary to distinguish content presence from correctly bound content. Artifacts:
`artifacts/paired_runtime_v2/stage_c_corrupted_decoder200_{seed401.json,dev_repair/}`.
