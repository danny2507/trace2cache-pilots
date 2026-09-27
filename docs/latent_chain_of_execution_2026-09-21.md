# Latent chain-of-execution (output prediction)

Working title:

> **Don't Simulate in Tokens: A Fixed-Rate Execution Channel for Frozen CodeLMs**

Repair Gate 0 failed on MBPP, Refactory, and RunBugRun (3B and 7B). The splice
is faithful; the failing test is already in the prompt, so traces are unused.
This experiment changes the **task**, not the channel.

The decoder must emit stdout of a **gold** program on a given stdin. The
message is intermediate execution, not a debug hint beside a failing assert.

## Gate 0 (text only, no encoder)

Frozen `Qwen/Qwen2.5-Coder-3B-Instruct`, greedy, seed 1001. Panel:
`execution_sim_runbugrun_v1`, the same 128 RunBugRun test-split programs as
the repair cohort, with `target_source` in the prompt. The repair
`test.jsonl` is not modified.

| Condition | Evidence |
| --- | --- |
| `code_input` | gold program + stdin (CruxEval analogue) |
| `trace_gist` | that, plus last CALL/BRANCH/LINE/STATE |
| `trace_text` | that, plus compact intermediate events |

TEST/PASS/FAIL/RETURN are dropped. Gold stdout (and its lines of length ≥ 2)
are redacted from events. Specification omitted.

Metric: exact match on normalized stdout (not hidden-test Repair@1).

Decision: train a K-slot encoder **only if** `trace_gist` or `trace_text`
beats `code_input` by McNemar. Do not weaken `code_input` by hiding stdin.

## Result (seed 1001, 128 programs)

`trace_text` 25, `trace_gist` 14, `code_input` 12. Mean prompt tokens
2326 / 467 / 300. Peak 14.88 GiB. Frozen repair test sha `ddae5cf9…`
unchanged.

McNemar: trace_text vs code_input **15–2 p = 0.0023**; gist vs
code_input 5–3 p = 0.73.

Gate 0 **pass** for the compact intermediate trace. Gist is not a win.
Train K=8 of `trace_text` events on a disjoint RunBugRun train split,
then eval `true_latent` vs `trace_text` vs `code_input` vs shuffle on
this frozen 128. Do not train on the 128.
`artifacts/mbpp_generalization/execution_sim_text_seed1001/`.

## K=8 encoder (seed 1001)

Train: 320 official RunBugRun train bugs, excluding frozen test
`task_id`s and gold `source_hash`es. Loss: teacher-forced NLL of
fenced stdout + relu(margin + nll_true − nll_shuffled) + optional
reconstruction. No cosine-separation term. Slot roles:
`SIM_SLOT_ROLES`. Frozen 3B receiver. Best checkpoint step 140;
ranking stayed at the 0.1 margin; true/shuffled cosine stayed ≈ 1.0.

| Condition | Exact-match |
| --- | ---: |
| `trace_text` | 25 |
| `true_latent` | 14 |
| `shuffled_latent` | 14 |
| `code_input` | 12 |

McNemar: true vs code 4–2 p = 0.69; true vs text 5–16 p = 0.027;
true vs shuffle 0–0 p = 1.0 (128/128 identical).

Call: **latent_fail**. The compact intermediate *text* is used. The
K-slot splice of the same events is not. Do not claim a channel.
`artifacts/mbpp_generalization/execution_sim_latent_seed1001/`.

## Native event splice (no encoder)

K=8 was a protocol lock from repair (Haque: length hurts). Gate 0 on this
task inverted that: compact intermediate text won at ~2326 tokens. The
trained compressor did not bind. This run drops `RoleAwareEventEncoder`.

For each example, mean-pool the frozen input embedding of each compact-trace
line (`ROLE: content`, same strings as `trace_text`, no TEST/END_TEST
wrappers) and splice all of them at `MARKER`. Variable K, cap 24, mean
~19. No training. Shuffle splices another task's event vectors.

| Condition | Evidence |
| --- | --- |
| `code_input` | gold program + stdin (copied from Gate 0) |
| `trace_text` | compact intermediate events (copied from Gate 0) |
| `native_events` | same events as mean-pooled native embeddings |
| `shuffled_native` | another task's event vectors |

Win: `native_events` ≥ `trace_text` at this K **or** `native_events` >
`code_input`, **and** shuffle drops. Do not claim a channel if shuffle
does not drop. Do not hide stdin. Do not train on the frozen 128.

## Result (seed 1001, 128 programs)

No encoder. 2238 unique event strings, mean 19.17 slots, cap 24. Peak
7.43 GiB. Trainable parameters 0. Frozen sim test sha `074dd6ce…` and
repair test sha `ddae5cf9…` unchanged. Gate 0 text rows copied.

| Condition | Exact-match | mean prompt tokens |
| --- | ---: | ---: |
| `trace_text` | 25 | 2326 |
| `code_input` | 12 | 300 |
| `native_events` | 11 | 327 |
| `shuffled_native` | 11 | 327 |

McNemar: native vs shuffle **0–0 p = 1.0** (pass/fail identical on
128/128; 100/128 predictions identical); native vs code 1–2 p = 1.0;
native vs text **3–17 p = 0.0026**.

Call: **native_fail**. The same events help as tokens and are unused as
mean-pooled embeddings. Do not claim a channel. Do not train ICAE of
this splice. Do not raise K of an unused prefix.
`artifacts/mbpp_generalization/execution_sim_native_seed1001/`.

## Trace-grounded latent execution (distill the tracer)

The splice family is closed. This run unfreezes a LoRA on the same 3B
and distills interpreter events into a K=16 latent rollout.

Train on the disjoint 320. Never on the frozen 128. Prompt is always
`code_input` (gold program + stdin). Traces never enter the text.
Stdin is not hidden.

| Objective | Train | Test |
| --- | --- | --- |
| `code_sft` | LoRA SFT of fenced stdout | greedy from the prompt |
| `latent` | that, plus teacher-forced resampled native event vectors and a projector that writes them | `rollout`: K projected states, no tracer; `teacher`: gold event vectors (oracle); `code`: 0 extra states |

Matched LoRA rank 16 on q/k/v/o, 1280 steps (one epoch at accum 4),
alignment weight 1.0. Shuffle is a separate objective, trained only if
`latent_rollout` beats `code_sft` by McNemar.

Win: `latent_rollout` > `code_sft` by McNemar. Then shuffled-teacher
training is required before claiming the tracer was distilled rather
than extra compute. `teacher` beating `code_sft` without a rollout win
is `reader_pass` (LoRA can read embeddings; it cannot simulate).
`artifacts/mbpp_generalization/execution_sim_distill_seed1001/`.

## Result (seed 1001, 128 programs)

LoRA rank 16 on q/k/v/o. Projector LayerNorm+Linear, identity-init.
`code_sft` 7.37M params; latent 11.57M. 1280 steps each, one epoch
at accum 4. Alignment cosine 0.057 → 0.929. Teacher-forced stdout
NLL collapsed. Peak 8.84 GiB. Elapsed 2612 s. Frozen sim test sha
`074dd6ce…` and repair test sha `ddae5cf9…` unchanged. Train 320
disjoint. Mean test rollout cosine 0.878.

| Condition | Exact-match | mean prompt tokens |
| --- | ---: | ---: |
| Gate 0 `trace_text` | 25 | 2326 |
| Gate 0 `code_input` | 12 | 300 |
| `code_sft` | 11 | 300 |
| `latent_code` | 8 | 300 |
| `latent_teacher` | 5 | 316 |
| `latent_rollout` | 4 | 316 |

McNemar: rollout vs code_sft **1–8 p = 0.039** (rollout loses);
teacher vs code_sft 1–7 p = 0.070; code vs code_sft 2–5 p = 0.45;
rollout vs teacher 0–1 p = 1.0. Pass/fail identical rollout vs
teacher on 127/128; predictions identical on 81/128. Versus
`code_sft`, predictions identical on 16/128.

Call: **distill_fail**. The projector writes teacher-like vectors
and greedy exact-match falls. Oracle gold event embeddings also
fall. Alignment is not a channel. Do not train shuffled. Do not
claim the tracer was distilled. Compact intermediate *text* is
still the only win on this panel (25).
`artifacts/mbpp_generalization/execution_sim_distill_seed1001/`.

## Vocab-snap native splice (on-manifold)

Native mean-pooled event vectors sit off the token manifold.
This run keeps the frozen 3B, no encoder, no LoRA. Each native
vector is replaced by the cosine-nearest row of the input
embedding table. The spliced vector is that raw vocab row.
Variable K, cap 24. Shuffle splices another task's snapped
vectors. Gate 0 text rows copied.

| Condition | Evidence |
| --- | --- |
| `code_input` | gold program + stdin (copied from Gate 0) |
| `trace_text` | compact intermediate events (copied from Gate 0) |
| `vocab_snap` | same events as nearest vocab rows |
| `shuffled_snap` | another task's snapped vectors |

Win: `vocab_snap` beats `code_input` by McNemar **and** shuffle
drops. Do not claim a channel if shuffle does not drop. Do not
hide stdin. Do not train on the frozen 128.
`artifacts/mbpp_generalization/execution_sim_snap_seed1001/`.

## Result (seed 1001, 128 programs)

No encoder. 2238 unique event texts collapse to **14** unique
vocab ids under cosine snap. Mean snap cosine 0.75, min 0.41.
Mean 19.17 slots, cap 24. Peak 9.07 GiB. Trainable parameters 0.
Elapsed 366 s. Frozen sim test sha `074dd6ce…` and repair test
sha `ddae5cf9…` unchanged. Gate 0 text rows copied.

| Condition | Exact-match | mean prompt tokens |
| --- | ---: | ---: |
| `trace_text` | 25 | 2326 |
| `code_input` | 12 | 300 |
| `shuffled_snap` | 11 | 327 |
| `vocab_snap` | 10 | 327 |

McNemar: snap vs code 4–6 p = 0.75; snap vs shuffle **3–4 p = 1.0**;
snap vs text **5–20 p = 0.004**. Pass/fail identical snap vs
shuffle on 121/128; predictions identical on 75/128.

Call: **snap_fail**. On-manifold nearest-vocab rows of the same
events are unused. The splice family is closed. Compact
intermediate *text* is still the only win on this panel (25).
Do not train ICAE. Do not raise K. Do not hide stdin. Do not
claim a channel.
`artifacts/mbpp_generalization/execution_sim_snap_seed1001/`.
