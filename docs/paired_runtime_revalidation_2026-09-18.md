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

No typed, binding, or decoder model has been retrained after this repair.  The next GPU action is
a small, predeclared plain-vs-parent-aggregate gradient/parity smoke followed by a matched rerun;
it should not reuse old virtual-node results as an architecture conclusion.
