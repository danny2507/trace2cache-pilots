# Trace2Cache handover: MBPP generalization — 2026-09-19

## Objective and current decision

The next experiment is program repair on **MBPP tasks unseen during adapter training**. The
question is whether runtime evidence sent as eight model-native continuous states improves
held-out patch correctness over the same frozen CodeLM with no evidence, a well-prompted textual
trace, and I/O-only evidence. Keep a runtime-corruption intervention as the evidence-use check.

Stage C on 12 curated toy-program families is finished. Do not spend more runs tuning its fixed
24-view development panel. It is useful for debugging interventions but too small to establish a
mechanism or statistical significance. The next model should build the larger task-level
evaluation and report uncertainty across independent MBPP tasks.

## What is established, and what is not

The receiver in Stage C is frozen `Qwen/Qwen2.5-1.5B-Instruct`. A roughly 4M-parameter
role-aware trace encoder produces **eight continuous soft states** that replace a marker in
`inputs_embeds`. This is model-native latent communication through the receiver's prefill; **it
does not directly write per-layer KV cache entries**. All neural work uses the project-local
`.venv` and CUDA BF16 on the A100. Check available GPU memory before a run; other processes may
use the card.

The matched Stage C seed-401 schedule starts each branch from the same random encoder state:
400 vector updates at 1e-3, 800 at 3e-4, 800 at 1e-4, then 200 decoder-alignment updates at
3e-5. The full-runtime and I/O-only branches have matching initialization and sampling hashes.
The 24 evaluation views comprise only **12 independent source families**, two views per family.

| Branch, fixed 24 views | True repair | Paired-swap repair | Complete true pairs |
| --- | ---: | ---: | ---: |
| Full runtime latent | 9/24 | 6/24 | 2/12 |
| I/O-only latent | 4/24 | 8/24 | 0/12 |
| Runtime payload/source corrupted, I/O retained | 7/24 | 7/24 | 0/12 |
| Runtime packets moved between tests | 7/24 | 5/24 | 1/12 |
| Runtime packets moved between steps of the same test | 10/24 | 8/24 | 1/12 |
| Prompt-only debugger text, no adapter training | 2/24 | 1/24 | 1/12 |

The full latent branch has a small true-versus-swap effect. Erasing runtime payloads removes that
effect. The two packet-permutation branches retain much of it, so Stage C **does not show
fine-grained test, temporal, source-line, or def-use binding**. These are one-seed directions, not
significance claims. The text result is one explicit frozen prompt; it does not establish latent
superiority over all possible text prompts. The text prompt averaged 1,158 tokens; the latent
condition used about 103 text tokens plus eight continuous states.

Detailed evidence, exact artifacts and caveats: `docs/paired_runtime_revalidation_2026-09-18.md`.
Control definitions: `src/trace2cache/evidence_controls.py`. Stage C plan:
`docs/stage_c_execution_plan_2026-09-18.md`.

## Existing MBPP assets and their limits

- Local data: `.local/datasets/mbpp/mbpp.jsonl`,
  `.local/datasets/mbppplus/MBPPPlus-v0.2.0.jsonl.gz`. These files are project-local and ignored
  by Git; verify hashes and task IDs in the new manifest. The official ID split implemented in
  `src/trace2cache/mbpp.py` is train 601–974 (374 tasks), validation 511–600 (90 tasks), test
  11–510 (500 tasks), and prompt 1–10.
- `src/trace2cache/mbpp.py` implements deterministic AST mutations, call execution, tracing,
  and an isolated worker. `scripts/audit_mbpp.py` contains the first-order mutation audit. It
  found 142 train tasks, 36 validation tasks and 221 test tasks with at least one mixed
  pass/fail mutant; these are **candidate-yield counts**, not a final evaluation cohort.
- `scripts/run_mbpp_latent_repair.py` already trains an open-ended repair adapter on MBPP and
  has task-level train/validation splits. Reuse useful mutation/tracing pieces, but review its
  data and evaluation paths before using it for the final study: `_fuzzed_calls` derives calls
  from `test_list`; the current evaluator validates generated patches on the task's visible
  tests; defaults cap evaluation at 36 examples and generation at eight. It does not provide
  an untouched hidden-test Repair@1 estimate over 100+ independent tasks.
- Earlier 3B Coder MBPP pilot checkpoints and results are in `checkpoints/mbpp_repair/` and
  `artifacts/mbpp_repair/`. The 800-step contrastive run trained on 178 tasks and evaluated
  teacher-forced loss on 36 examples, but seven of eight generated outputs were identical for
  true versus different-task shuffled latent. Treat these checkpoints as negative diagnostics,
  not as a proven initialization for the new experiment.
- The Stage C encoder/checkpoints use the 1.5B receiver and a closed 24-behavior repair
  codebook. They cannot be dropped into open-ended 3B Coder MBPP repair or counted as MBPP
  task-transfer results. Remove closed codebook labels from the new training objective.

## Phase D execution protocol

1. **Freeze dataset and evaluation first.** Choose one receiver for every MBPP condition
   (recommend the locally cached `Qwen/Qwen2.5-Coder-3B-Instruct` used by earlier MBPP pilots).
   Fix train/validation/test task IDs, mutation operators, seeds, evidence call generation,
   public failing/pass examples and hidden tests in a versioned manifest. Every mutant, fuzz
   call, trace and augmentation from one task must remain in that task's split. Do not select
   the final test cohort after seeing model outputs.
2. **Make hidden tests genuinely hidden from the model input.** The existing builder uses MBPP
   `test_list` to derive calls. For final test tasks, define public evidence and a separate
   held-out call/assertion set before training; audit exact call/input overlap and keep original
   MBPP assertions plus applicable MBPP+ tests for final validation only. If a task cannot
   supply a disjoint, executable evidence/hidden-test partition, exclude it with a recorded
   reason. Canonical source may serve as an offline oracle for expected outputs and supervised
   training targets, but must never appear in evaluation prompts or adapter training for test
   tasks. Validate canonical source against the selected hidden suite before inclusion.
3. **Audit yield and leakage.** Save one row per candidate mutant: task ID, mutation kind,
   source hash, public evidence IDs, hidden test IDs, canonical status, mutant status,
   exception/timeout category and selection reason. Report independent task counts and mutant
   counts separately. Target at least 100 eligible test tasks; if fewer qualify, report the
   actual count rather than relaxing hidden-test criteria after seeing results.
4. **Build matched conditions on the same examples.** No evidence; prompt-only I/O summary;
   prompt-only compact full trace with explicit debugger instruction; latent I/O-only; latent
   full-runtime; and full-runtime with payload corruption. Give each condition the same buggy
   program, public test, receiver, decoding budget and evaluator. Train separate latent branches
   with matched initialization, sample order, update budget and decoder objective. Also swap or
   corrupt evidence **at inference** for the trained full encoder as an additional sensitivity
   check; distinguish this from separately retrained controls.
5. **Use open-ended repair supervision.** Train toward verified canonical patches on train
   tasks only. Do not use the toy 24-codebook/oracle identity metric as the primary MBPP goal.
   A small smoke run should first verify nonzero gradient into the adapter, frozen receiver
   parameters, plausible generated source and valid patch extraction. Then run three predeclared
   training seeds (suggest 1001, 1002, 1003) without selecting a favorite on the final test.
6. **Evaluate every eligible test task.** Greedy Repair@1 is the primary metric: extract one
   replacement patch and require it to pass every hidden test under the same timeout/resource
   policy. Count syntax, extraction, policy, runtime and timeout failures in the denominator.
   Report results both per mutant and aggregated per task; use task as the uncertainty unit.
   Record exact-output agreement between full and corrupted/swap conditions, prompt token
   counts, latent slot count, latency and GPU memory. Use paired task-level bootstrap intervals
   and a paired randomization or McNemar test for the full-versus-control deltas.
7. **Predeclare the decision.** A generalization claim needs a replicated positive full-runtime
   minus no-evidence/text/I/O gap on held-out tasks, plus reduced performance when runtime
   evidence is corrupted or swapped. If full and corrupted are statistically indistinguishable,
   report a generic soft-prompt or I/O/layout effect. Do not claim direct KV-cache injection or
   fine-grained causal binding from this architecture.

Before a long GPU run, commit the protocol, dataset-builder/evaluator code and tests; verify CUDA
and free A100 memory. Use only `.venv`, `.local`, `checkpoints/`, `artifacts/` and other paths
inside this project. Preserve existing files. Commit generated manifests/results and push the
scoped changes to GitHub after verification. Checkpoint `.pt` files are ignored by Git; record
their hashes and upload them to the existing private Hugging Face project only when packaging
the final trained artifacts, with credentials kept out of the repo. No API generation is needed
for the first deterministic MBPP pilot.

## Suggested first concrete deliverable

Implement and test the **frozen MBPP cohort builder plus hidden-test evaluator** before training:
one manifest with at least 100 eligible test task IDs if available, disjoint public/hidden call
hashes, reasons for exclusions, and a no-evidence versus prompt-only text baseline on the same
cohort. This gives the next model a trustworthy denominator and a meaningful reference before
spending GPU time on new latent branches.

## Workspace state at handover

Git HEAD when this handover was drafted: `3315942` (`Record within-test temporal binding
control`). The working tree also has a pre-existing untracked directory
`artifacts/paired_runtime_v2/named_text_compact_dev_repair/`; leave it untouched unless its
provenance has been checked. Local Stage C checkpoints are under
`checkpoints/paired_runtime_v2/` and are ignored by Git. The latest committed Stage C temporal
results are `artifacts/paired_runtime_v2/stage_c_temporal_decoder200_dev_repair/`.
