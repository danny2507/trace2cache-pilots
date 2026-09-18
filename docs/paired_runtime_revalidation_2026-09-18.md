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
