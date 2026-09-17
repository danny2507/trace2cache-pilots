# Pilot results — 2026-09-13

These are directional diagnostics, not publication-level estimates.

## 0. MBPP data-feasibility audit

The original 974-task MBPP release was downloaded only into the project-local ignored data
directory. Running each canonical solution against its three published assertions in an isolated
Python subprocess gave 973/974 passing tasks. Task 123 timed out because its reference amicable-
number implementation is unusually slow; all 374 official training tasks passed.

A deterministic first-order AST mutator was then applied to the 374 training tasks, with at most
12 candidates per task. Comparison, boolean, arithmetic, and integer-boundary mutations produced:

| Quantity | Count |
|---|---:|
| Candidate mutants executed | 1,997 |
| Mutants with both passing and failing tests | 508 |
| Training tasks with at least one mixed mutant | 142 |
| Training tasks with at least two mixed mutants | 104 |
| Mutants failing every published test | 1,033 |
| Equivalent/surviving mutants | 410 |
| Timed-out mutants | 46 |

This is enough data for an initial cross-program trace encoder without an LLM data-generation API.
Splits must remain task-disjoint: only task IDs 601--974 may produce training mutants. MBPP+ is
reserved for evaluation and contributes 39,841 extended inputs over 378 curated tasks; because it
also contains some IDs from the original train split, only official test IDs 11--510 are eligible
for the final MBPP+ evaluation.

The identical mutation audit on unseen tasks found 151 mixed mutants across 36/90 validation tasks
and 762 mixed mutants across 221/500 test tasks. Thus the deterministic operator family supplies a
usable task-disjoint evaluation set as well as training data. The exact reports are stored under
`artifacts/data/mbpp_{train,validation,test}_mutation_audit.json`.

### Open-ended MBPP latent repair

A frozen `Qwen2.5-Coder-3B-Instruct` receiver was paired with a 4.73M-parameter role-aware trace
encoder producing eight soft states. Training used 241 mixed mutants from 139 official MBPP train
tasks. Validation used one mutant from each of 36 distinct official validation tasks; shuffled
controls always came from a different task. The run performed 400 microsteps (100 optimizer
updates with four-step gradient accumulation). Peak GPU allocation across training and generation
was 11.85 GiB.

| Validation condition | Mean teacher-forced patch loss | Repair@1 (8-task sample) |
|---|---:|---:|
| No runtime evidence | 0.48276 | 1/8 |
| Text trace | 0.46882 | 5/8 |
| True latent trace | 0.22887 | 7/8 |
| Different-task shuffled latent | 0.22906 | 7/8 |

True and shuffled latent conditions had identical pass/fail outcomes on all eight generated tasks,
and their generated text was exactly identical on seven. Therefore the apparent latent gain is not
instance-specific runtime communication. The encoder learned a generic continuous repair prompt
that steers the frozen receiver toward shorter, valid source. This run fails the shuffled-control
gate despite improving raw repair accuracy.

The next training objective must make trace identity necessary. Suitable interventions are
same-prompt counterfactual repairs or an explicit margin requiring the correct trace to assign
higher likelihood to its patch than a different-task trace. Merely adding more MBPP mutants to the
current patch loss is unlikely to fix the shortcut.

### Automatic augmentation plus task-level contrastive loss

The mutation family was expanded to cover alternative comparisons and arithmetic operators,
augmented assignments, condition negation, boolean changes, and assignment deletion. A
type-directed fuzzer perturbed literal MBPP arguments and used the canonical implementation as the
output oracle. This increased the project-local cache to 1,173 bundles over 183 train tasks (599
base-test and 574 fuzz-derived bundles), plus 297 bundles over 47 validation tasks.

The contrastive run sampled 768 balanced bundles from 178 train tasks, including 387 fuzz-derived
bundles. For the same buggy-code prompt and target patch, it optimized the true-trace patch loss
plus a hinge requiring that loss to beat a different-task trace by a margin of 0.1. Qwen remained
frozen. The run used 800 microsteps (200 optimizer updates) and peaked at 12.09 GiB.

| Validation condition | Mean teacher-forced patch loss | Repair@1 (8-task sample) |
|---|---:|---:|
| No runtime evidence | 0.57602 | 4/8 |
| Text trace | 0.55831 | 5/8 |
| True latent trace | 0.25821 | 6/8 |
| Different-task shuffled latent | 0.25886 | 5/8 |

One task was repaired only with its true latent, providing a weak directional hint. However, seven
of eight true/shuffled generations were exactly identical, and the mean validation loss gap was
only 0.00065, far below the requested 0.1 margin. Logged per-example ranking losses remained near
0.1 throughout training. Therefore automatic data scaling did not remove the generic-prefix
shortcut, and this run still fails the instance-specific communication gate. The next objective
needs direct semantic supervision (for example trace-fact reconstruction/matching) or same-prompt
counterfactual targets; simply increasing the hinge weight is not well motivated by this result.

### Role-factorized hard-counterfactual repair pilot (2026-09-16)

The next pilot replaced eight anonymous latent slots with fixed evidence budgets: failing return,
causal branch, last state update, local state context, passing reference behavior, pass/fail delta,
control-to-data link, and one integration slot. It also trained against three same-example hard
counterfactuals, all of which preserve buggy code, target patch, event count, and event vocabulary:
`role_swap` exchanges branch/state identities, `value_swap` moves event text/value bundles between
dynamic positions, and `trace_reassigned` assigns dynamic events to the opposite pass/fail test.

The frozen receiver was again Qwen2.5-Coder-3B-Instruct. The 4.73M-parameter encoder trained for
400 microsteps on 512 bundles from 172 official MBPP train tasks (262 fuzz-derived) and evaluated
on 24 task-disjoint validation tasks. Peak GPU allocation was 19.55 GiB.

| Validation condition | Mean teacher-forced patch loss | Repair@1 (4-task decode sample) |
|---|---:|---:|
| No runtime evidence | 0.52121 | 2/4 |
| Text trace | 0.50886 | 3/4 |
| True latent trace | 0.26016 | 2/4 |
| Different-task shuffled latent | 0.26246 | 2/4 |
| Role-swapped latent | 0.26244 | 2/4 |
| Value-swapped latent | 0.26026 | 2/4 |
| Pass/fail-reassigned latent | 0.26006 | 2/4 |

The role-swap loss gap (0.00228) and shuffled gap (0.00230) are directional but tiny; value swap
is effectively tied, and pass/fail reassignment is marginally *better* than true evidence. More
decisively, all four true-versus-role-swap generated responses were exactly identical. Therefore
the role-factorized architecture and patch-ranking loss do not by themselves make causal roles
necessary for repair. The result rejects this version of the method rather than validating a
role-aware real-code channel.

The next gate should be an auxiliary semantic task with direct labels (branch taken, last
definition, expected/observed return, and pass/fail delta) followed by repair, or a benchmark whose
same visible prompt admits different gold patches selected only by hidden behavioral evidence.

### Semantic trace pretraining gate (2026-09-16)

Rather than ranking a long gold patch directly, a role-factorized encoder was trained on explicit
same-example trace controls. A frozen Qwen2.5-Coder-3B input embedding table supplied native event
vectors; Qwen itself was unloaded before adapter training. Each batch contained a valid execution
and three views with the same code, event vocabulary, length, and target: `role_swap`, `value_swap`,
and `trace_reassigned`. The objective combined true-versus-corrupt validity ranking, four-way view
classification, mutation-family prediction, and per-event role reconstruction.

The encoder trained for 1,500 steps over 1,024 examples from official MBPP train tasks and was
evaluated on 192 task-disjoint validation traces. It has 4.75M encoder parameters; the peak
allocation was below 0.3 GiB after the one-time frozen embedding-table construction.

| Held-out metric | Result |
|---|---:|
| True validity exceeds role swap | 78.65% |
| True validity exceeds value swap | 99.48% |
| True validity exceeds pass/fail reassignment | 100.00% |
| Four-way true/corruption macro accuracy | 91.28% |
| Event-role reconstruction accuracy | 85.71% |
| Mutation-family accuracy | 32.29% |

This is the first task-disjoint real-MBPP evidence that the encoder can preserve the role/value/test
structure of an execution trace. It is not yet evidence that a frozen decoder uses this semantic
state for repair: the validity and corruption heads are trainable. The next experiment initializes
the repair encoder from this checkpoint and retains the semantic losses as an anti-collapse
regularizer.

### Frozen semantic encoder / Qwen-Coder interface check (2026-09-16)

The semantic-pretrained encoder was inserted unchanged into frozen Qwen2.5-Coder-3B-Instruct and
evaluated on 36 task-disjoint MBPP validation examples (`steps=0`). This isolates whether the
receiver is sensitive to the learned continuous states before patch fine-tuning can overwrite them.

| Condition | Mean teacher-forced patch loss | Repair@1 (4-task decode sample) |
|---|---:|---:|
| No runtime evidence | 0.57602 | 2/4 |
| Text trace | 0.55831 | 3/4 |
| True semantic latent | 0.57566 | 3/4 |
| Different-task shuffled latent | 0.57648 | 3/4 |
| Role-swapped latent | 0.57757 | 3/4 |
| Value-swapped latent | 0.58310 | 4/4 |
| Pass/fail-reassigned latent | 0.57882 | 3/4 |

The true latent is better than all three hard corruptions in mean patch likelihood, especially the
value swap (gap 0.00745). This is weak but real receiver sensitivity to the learned representation.
It does not pass the repair gate: generation remains largely unchanged and the value-swap control
is paradoxically best in the four decoded tasks. Thus external semantic heads and native soft-state
insertion are insufficient interface alignment; train a frozen-Qwen short semantic answer task
before attempting open-ended repair transfer.

### Frozen-Qwen latent interface alignment (2026-09-16)

The semantic encoder was next trained through a completely frozen Qwen2.5-Coder-3B-Instruct
decoder. The prompt disclosed no trace text and asked only whether the internal latent state was a
coherent role/value/test binding; target tokens were `VALID` for the original trace and `INVALID`
for each same-example corruption. This directly aligns encoder output states to the receiver's
native token likelihood geometry. Training used 1,024 examples from 182 train tasks for 600 steps;
evaluation used 192 examples from 46 task-disjoint validation tasks.

| Held-out decoder metric | Result |
|---|---:|
| Four-view `VALID`/`INVALID` macro accuracy | 87.37% |
| Correct original trace classified valid | 70.83% |
| Role swap classified invalid | 79.17% |
| Value swap classified invalid | 99.48% |
| Pass/fail reassignment classified invalid | 100.00% |
| Original score exceeds role swap | 71.88% |
| Original score exceeds value swap | 98.96% |
| Original score exceeds reassignment | 100.00% |

This is the first real-MBPP result where the **frozen target decoder**, rather than an external
probe head, uses the latent interface to distinguish role-aware execution evidence. The task is
still a synthetic trace-coherence question, so it is an interface gate rather than a repair claim.
The next transfer freezes this interface-aligned encoder while measuring whether repair likelihood
and generated patches retain true-versus-corruption sensitivity.

### Frozen interface-aligned encoder repair transfer (2026-09-16)

The 600-step interface-aligned encoder was inserted unchanged into frozen Qwen-Coder repair
(`steps=0`) and evaluated on 36 task-disjoint validation traces. It slightly changed repair behavior
relative to the semantic-head-only encoder but did not preserve every causal control.

| Condition | Mean teacher-forced patch loss | Repair@1 (4-task decode sample) |
|---|---:|---:|
| No runtime evidence | 0.57602 | 2/4 |
| Text trace | 0.55831 | 3/4 |
| True interface latent | 0.57156 | 4/4 |
| Different-task shuffled latent | 0.57032 | 2/4 |
| Role-swapped latent | 0.57469 | 3/4 |
| Value-swapped latent | 0.57218 | 4/4 |
| Pass/fail-reassigned latent | 0.57177 | 4/4 |

True evidence obtains a promising `4/4` versus shuffled `2/4` decode direction, and it repairs one
task that shuffled evidence leaves mixed. However, true patch likelihood is not better than
shuffled, while value-swap and pass/fail-reassignment decode equally well. The sample is too small
and the controls are inconsistent, so this does not establish role-aware repair. The next method
must retain decoder-interface supervision jointly with patch loss, rather than allowing repair-only
updates to discard it.

### Joint decoder-interface and repair training (2026-09-16)

The interface-aligned encoder was initialized from the frozen-Qwen `VALID`/`INVALID` checkpoint and
then trained for 400 repair microsteps on 512 bundles from 172 MBPP train tasks. Each update retained
one true-interface label and one randomly selected same-example corruption label with weight 0.5,
alongside patch likelihood and patch-ranking losses. Evaluation uses 24 task-disjoint validation
traces.

| Condition | Mean teacher-forced patch loss | Repair@1 (4-task decode sample) |
|---|---:|---:|
| No runtime evidence | 0.52121 | 2/4 |
| Text trace | 0.50886 | 3/4 |
| True joint latent | 0.26133 | 3/4 |
| Different-task shuffled latent | 0.26292 | 3/4 |
| Role-swapped latent | 0.37344 | 3/4 |
| Value-swapped latent | 0.60446 | 2/4 |
| Pass/fail-reassigned latent | 0.60080 | 3/4 |

This is the strongest real-code role-aware likelihood result so far: true evidence beats role swap
by 0.11211, value swap by 0.34313, and pass/fail reassignment by 0.33947, while retaining a small
0.00160 advantage over different-task shuffled evidence. The decoder is therefore not merely
receiving a generic latent repair prefix; its gold-patch likelihood depends strongly on correctly
bound role/value/test structure.

However, generated Repair@1 remains insensitive except for value swap (true 3/4 versus value 2/4).
This is a valid representation/interface result, not yet a reliable open-ended repair improvement.
The next evaluation should increase generation to at least 24--36 tasks and use paired ambiguous
repair targets, where runtime evidence is required to choose among distinct valid patches.

## 1. Synthetic repair harness

Model: frozen `Qwen2.5-Coder-3B-Instruct`, BF16, greedy decoding.

After correcting an evaluator artifact that rejected safe defensive `ValueError` statements, both
test-only and compact-text trace conditions repaired all 6/6 cases. Compact traces increased mean
input length from 134 to 460 tokens without measurable benefit. Conclusion: the toy set has a hard
ceiling and should not be used to judge the method.

## 2. Refactory real student submissions

Subset: first 20 failing submissions from Refactory question 1 (sequential search). Each patch was
validated on the shown failing test and ten held-out instructor tests.

The first run exposed a serious generation-budget confound: 12/20 test-only outputs hit the
128-token cap because the model added explanations, while none of the trace outputs did. Re-running
test-only at 256 tokens removed one apparent trace win. Fair directional results:

| Condition | Repair@1 | Mean input tokens | Mean output tokens | Mean generation seconds |
|---|---:|---:|---:|---:|
| Test only (256 max) | 11/20 (55%) | 187 | 158 | 11.26 |
| Compact text trace | 11/20 (55%) | 423 | 56 | 3.83 |
| Full text trace | 12/20 (60%) | 599 | 58 | 3.97 |
| Structured JSON trace | 13/20 (65%) | 777 | 57 | 4.15 |
| Shuffled JSON trace | 12/20 (60%) | 777 | 61 | 4.88 |

True versus shuffled JSON agreed on 19/20 bugs: 12 both correct, 7 both wrong, and only 1 true-only
correct. Thus most of the apparent JSON gain is compatible with generic prompt/format steering,
not instance-specific use of runtime evidence.

Only four bugs changed outcome across the main conditions:

| Submission | Test | Compact | Full | JSON | Shuffled JSON |
|---|---:|---:|---:|---:|---:|
| wrong_1_006 | 0 | 0 | 0 | 1 | 1 |
| wrong_1_009 | 0 | 0 | 1 | 0 | 0 |
| wrong_1_013 | 1 | 0 | 0 | 1 | 0 |
| wrong_1_019 | 0 | 1 | 1 | 1 | 1 |

Submissions 009 and 013 contain nearly identical predicates and produce similar failing behavior,
yet text traces flip them in opposite directions. This is evidence of instability, not reliable
causal trace use.

### Directional complexity split

Using target trace event count only to form post-hoc groups:

| Condition | Short (<=7 events, n=11) | Long (>7 events, n=9) |
|---|---:|---:|
| Test only | 7/11 | 3/9 |
| Compact text | 9/11 | 2/9 |
| Full text | 9/11 | 3/9 |
| JSON | 9/11 | 4/9 |

Long-trace programs are also harder without traces, so this does not identify a causal length
effect. It does show why raw accuracy versus trace length is insufficient; the experiment must vary
trace length within the same bug or use a paired complexity intervention.

## 3. Latent runtime-value sufficiency

Model: frozen `Qwen2.5-1.5B-Instruct`. Training traces contain 2–6 modular arithmetic events;
evaluation contains unseen longer traces of 8–12 events. Target is the final accumulator digit.

### Failed generic interface

Four arbitrary projected soft states prepended before the chat header failed after 400 steps:

- latent: 10.5%;
- shuffled latent: 10.2%;
- no trace: 11.3%;
- text trace: 53.1%.

Moving the state into a semantic slot inside the user message made an exact native digit embedding
immediately usable: 83.2–85.9% depending on evaluation sample, while text reached 99%+.

### Native sink anchor

A one-state message anchored at the frozen receiver's embedding of the sliced sink value achieved,
without training:

| Channel | Accuracy (n=256) |
|---|---:|
| Native-anchored latent | 83.2% |
| Shuffled anchored latent | 10.2% |
| No trace | 9.8% |
| Text trace | 99.6% |

This clean true-versus-shuffled gap validates an instance-specific latent channel. It does **not**
yet validate learned compression: the probe target is directly represented by the sink anchor.

### Non-copy semantic probe

To test whether the receiver merely copied the hidden digit, the target was changed to the binary
question “is the final accumulator greater than 4?” The message still contained only the native
embedding of the final digit and the receiver remained frozen:

| Channel | Threshold accuracy (n=256) |
|---|---:|
| Native-anchored latent | 100.0% |
| Shuffled anchored latent | 44.5% |
| No trace | 52.7% |
| Text trace | 91.0% |

Thus a native state inserted in the correct semantic slot can be consumed for a downstream decision,
not only copied. Still, an exact token embedding is mathematically close to transmitting one hidden
text token. The result validates the interface and placement principle, not a beyond-text compression
claim.

## New design hypothesis

An unconstrained projector forces the adapter to learn both runtime semantics and the receiver's
usable activation manifold. The pilots suggest a better design:

\[
z_k = \sum_j \alpha_{kj} e_{LM}(f_j) + g_k\,\Delta z_k,
\]

where `f_j` are native tokenizable event fields, the weights select causal evidence, and the
continuous residual gate starts at zero. This **native-anchor plus learned residual** should be
compared against unconstrained soft prefixes and hard token summaries.

The next decisive probe must require combining multiple runtime facts rather than copying the sink:
branch cause, last definition, or pass/fail divergence. It should vary the query, use learned causal
selection, hold out program templates, and retain shuffled and hard-token-anchor controls.

## 4. One-slot multi-fact communication

The next probe pairs a correct reference execution with a buggy execution. Both are independently
generated modular-arithmetic traces. One query-independent latent state must support four questions:
the reference sink, buggy sink, their modulo-10 delta, and whether buggy is greater than reference.
Training traces have 2–6 events per run; evaluation traces have 8–12.

### End-to-end trace resampler: anchor shortcut

With the native reference sink as anchor and a gated learned residual, 800 training steps produced:

| Question | True latent | Shuffled latent |
|---|---:|---:|
| Reference sink | 91.0% | 8.6% |
| Buggy sink | 9.0% | 10.9% |
| Delta | 13.3% | 10.9% |
| Greater | 68.8% | 54.7% |
| **Macro** | **45.5%** | **21.3%** |

The adapter learned the anchored first fact but did not pack the second. This is an anchor shortcut,
not successful multi-fact compression.

### Oracle pair codebook: the one-slot channel has capacity

To separate receiver capacity from trace encoding, an oracle codebook assigns each of the 100
possible `(reference_digit, buggy_digit)` pairs one learned vector. Every vector starts exactly at
the receiver's native embedding for the reference digit. Only 153,600 code parameters train; Qwen
remains frozen. This representation cannot see the question, and the same vector is used for all
four tasks.

After 1,200 steps with three training paraphrases per task:

| Question | One latent | Shuffled | Compact labeled text | Full trace text |
|---|---:|---:|---:|---:|
| Reference sink | 68.0% | 9.8% | 100.0% | 19.9% |
| Buggy sink | 65.6% | 11.7% | 100.0% | 41.0% |
| Delta | 83.6% | 9.8% | 41.8% | 12.1% |
| Greater | 85.9% | 44.1% | 53.9% | 53.9% |
| **Macro** | **75.8%** | **18.8%** | **73.9%** | **31.7%** |

This is the first positive result that transmits more information than one hidden native token: a
single vector supports recovery of two values and derived relations, with a large shuffled-message
gap. The long-trace collapse relative to compact labeled text also reproduces the text-noise issue
inside this controlled task.

On one held-out question paraphrase per task, latent macro accuracy drops to 41.6%, versus 16.6% for
shuffled latent and 62.8% for compact text. The held-out delta wording yields almost no correct
first-token answers even for compact text, so part of this drop is a receiver/prompt evaluation
failure rather than latent information loss. Nevertheless, the generalization gap is real.

### Claim boundary and next gate

The oracle codebook sees all 100 value pairs during training. It establishes channel capacity and
query-dependent decoding, but it can memorize a code per pair; it does **not** establish a
generalizing trace compressor. The next gate is therefore a parametric pair encoder trained with
held-out value pairs, followed by distillation from full traces into its successful codes. Direct KV
injection remains premature until this soft-state representation passes that gate.

Artifacts: `artifacts/latent_probe/multifact_k1_800.json` and
`artifacts/latent_probe/multifact_codebook_k1_aug1200.json`.

## 5. Compositional held-out-pair gate

The codebook was replaced with a 1.58M-parameter MLP over the receiver's native embeddings of the
two sink values. Its output is one vector, zero-initialized as the exact reference-digit anchor.
The split contains 80 training pairs and 20 held-out pairs. Holdout construction is balanced so
every reference digit, buggy digit, and modulo-10 delta occurs exactly twice; therefore it tests new
combinations without withholding an input or output class. Three prompt variants per task are used
for training. Qwen remains frozen.

Seed 29 after 1,200 steps:

| Evaluation | Latent | Shuffled latent | Compact labeled text | Full trace text |
|---|---:|---:|---:|---:|
| Seen pairs | 100.0% | 19.7% | — | — |
| **Unseen pairs** | **76.8%** | **19.7%** | **74.5%** | **32.5%** |
| Unseen pairs, held-out prompt | 62.5% | 19.6% | 62.5% | — |

Unseen-pair task breakdown:

| Question | Latent | Shuffled | Compact labeled text |
|---|---:|---:|---:|
| Reference sink | 100.0% | 9.8% | 100.0% |
| Buggy sink | 96.5% | 10.2% | 100.0% |
| Delta | 27.7% | 9.8% | 42.2% |
| Greater | 82.8% | 49.2% | 55.9% |

Two exact-hyperparameter replications give the following three-seed unseen-pair results:

| Metric | Seed 29 | Seed 41 | Seed 53 | Mean ± sample SD |
|---|---:|---:|---:|---:|
| Latent macro | 76.8% | 57.5% | 76.6% | **70.3 ± 11.1%** |
| Shuffled macro | 19.7% | 21.1% | 17.8% | **19.5 ± 1.7%** |
| Held-out-prompt latent | 62.5% | 48.9% | 60.3% | **57.2 ± 7.3%** |

The gate passes for multi-fact recovery in aggregate: the one-slot parametric representation beats
its shuffled control by 50.8 points on average and seed 29 matches the compact-text macro score on
combinations absent from training. Per-task means show the boundary clearly: reference 81.6%, buggy
88.8%, greater 88.5%, but exact delta only 22.1% with 19.9-point SD. Therefore recovery and ordinal
comparison generalize; exact arithmetic composition does not yet generalize reliably.

An earlier smoke split was discarded because its algebraic selection rule accidentally withheld
delta classes 0 and 5 rather than only value combinations. The balanced split above fixes that
confound and asserts its marginals at runtime.

Artifacts: `artifacts/latent_probe/compositional_pair_encoder_balanced_1200.json`,
`artifacts/latent_probe/compositional_pair_encoder_balanced_seed41.json`, and
`artifacts/latent_probe/compositional_pair_encoder_balanced_seed53.json`.

## 6. Full-trace-to-code distillation

A 4.48M-parameter event Transformer was trained to map two structured traces into the successful
pair-encoder teacher's one-vector code. The frozen Qwen receiver was not used in the distillation
loss; training minimized MSE plus cosine distance in latent space. Training used short traces (2–6
events per run) and the same 80 value pairs. Evaluation jointly held out the other 20 balanced pairs
and used longer traces (8–12 events per run, 16–26 combined events).

### Raw sequential events fail length transfer

With fixed sinusoidal positions, 3,000 steps fit seen-short traces almost exactly (cosine 0.9998 and
100% receiver macro accuracy), but unseen-long accuracy was 21.5%, indistinguishable from shuffled
at 21.2%. Unseen-long cosine fell to 0.708. Thus replacing learned positions did not prevent the
resampler from learning a length/position shortcut.

### Semantic sink markers recover transfer

The runtime representation was then augmented with distinct event types for the reference and buggy
sink events. These markers expose event role, not its value: both values still enter only as fields
of the full event sequence. After 2,000 distillation steps:

| Evaluation | Student latent | Shuffled | Teacher ceiling |
|---|---:|---:|---:|
| Seen pairs, short traces | 100.0% | 20.4% | — |
| **Unseen pairs, long traces** | **68.5%** | **18.4%** | **79.0%** |
| Unseen-long, held-out prompt | **63.0%** | **18.1%** | **65.9%** |

Representation similarity on unseen-long traces reached cosine 0.977, RMSE 2.11, and relative L2
error 0.261. Task accuracy was reference 82.0%, buggy 84.0%, delta 12.9%, and greater 94.9%.

This provisionally passes the trace-to-code gate and isolates an important mechanism: **semantic
event roles, rather than positional sequence learning, are necessary for length transfer**. The
student compresses 16–26 events into one receiver-readable state and retains a 50-point
true-versus-shuffled gap. However, because the current queries depend primarily on the two marked
sinks, this does not yet show that a causal path or multiple intermediate events are retained. The
next structural gate must make answers depend on marked sink plus earlier last-definition/branch
events and add irrelevant marked/unmarked distractors.

Artifacts: `artifacts/latent_probe/trace_distillation_fixedpos_3000.json` (raw failure) and
`artifacts/latent_probe/trace_distillation_sinkmarked_2000.json` (semantic-marker success).

## 7. Causal-path selection with heavy distractors

To prevent a sink-value shortcut, each reference and buggy run now contains a false candidate, true
candidate, and branch-selector event. The selected candidate determines the final value, while the
sink value given to the encoder is masked. Candidate, selector, run, distractor, and sink roles are
explicit structural event types. The training distribution contains 0–6 distractors; evaluation has
16–40 distractors (24–48 total events) and the 20 unseen output pairs. No LLM-generated data is used.

Seed 83 after 3,000 latent-distillation steps:

| Evaluation | Latent macro | Shuffled | Teacher ceiling |
|---|---:|---:|---:|
| Seen pairs, 0–6 distractors | 98.5% | 20.3% | — |
| **Unseen pairs, 16–40 distractors** | **65.8%** | **18.5%** | **80.0%** |
| Unseen + held-out prompt | **60.3%** | **18.8%** | **64.6%** |

The unseen-heavy-distractor latent has cosine 0.965 to the teacher, RMSE 2.57, and relative L2 error
0.323. Task accuracy is reference 82.0%, buggy 70.3%, delta 18.0%, and greater 93.0%.

For a causal intervention, both branch-selector states were flipped while the original targets were
retained. Macro accuracy collapsed from 65.8% to 16.3%; reference and buggy recovery each fell to
1.2%. This is strong evidence that the student uses the selector-to-definition path rather than a
masked-sink, event-position, or majority shortcut.

Two exact-hyperparameter replications and test-time causal corruptions give:

| Unseen-pair, 16–40 distractor metric | Seed 83 | Seed 97 | Seed 109 | Mean ± sample SD |
|---|---:|---:|---:|---:|
| Typed causal latent | 65.8% | 46.0% | 62.1% | **58.0 ± 10.5%** |
| Shuffled latent | 18.5% | 19.1% | 21.2% | **19.6 ± 1.4%** |
| Held-out prompt | 60.3% | 45.7% | 59.6% | **55.2 ± 8.2%** |
| Branch-selector flip | 16.3% | 24.9% | 16.5% | **19.2 ± 4.9%** |
| False/true candidate-role swap | 16.2% | 25.2% | 16.6% | **19.3 ± 5.1%** |

Mean unseen latent-to-teacher cosine is 0.936 ± 0.049. Seed 97 is a genuine optimization failure
relative to the other runs and is retained in all aggregates.

For a train-from-scratch removal ablation at seed 83, false and true candidates were assigned the
same coarse candidate role while preserving run identity and the branch event. Accuracy fell from
65.8% typed to 32.9% coarse, while shuffled was 20.3%. The residual performance is expected because
the correct value remains one of two visible candidates, but the representation no longer contains
enough information to resolve branch polarity.

The result supports a more precise method hypothesis: a fixed-rate latent channel works when runtime
evidence is represented as semantically typed causal events and distilled into a receiver-readable
code. The three-seed true-versus-shuffled gap is 38.4 points; both branch flips and candidate-role
swaps erase the gain. This remains controlled synthetic evidence, but semantic-role dependence is
now directly supported rather than inferred.

Artifacts: `artifacts/latent_probe/causal_path_seed83_ablation_eval.json`,
`artifacts/latent_probe/causal_path_distillation_seed97.json`,
`artifacts/latent_probe/causal_path_distillation_seed109.json`, and
`artifacts/latent_probe/causal_path_remove_roles_seed83.json`.

## 8. Executed Python trace corpus smoke test

A deterministic local generator executes the six curated Python repair cases and emits source code,
the failing assertion, expected/actual output, raw line events, changed-state trace, and heuristic
semantic roles (`call_input`, `branch_predicate`, `loop_control`, `state_definition`, and
`return_sink`). All 6 public tests fail as intended and produce 79 runtime events in total. No LLM or
API is used to create either programs or traces.

This corpus is not yet consumed by the latent encoder. The semantic annotations are currently
line/state heuristics rather than a sound def-use analysis. The next integration step is a mapper
from these real events plus AST information into branch, definition, dependency, and assertion-sink
roles, followed by code-and-test textual context plus latent runtime evidence at the receiver.

Artifacts: `artifacts/real_traces/python_bug_traces.jsonl` and
`artifacts/real_traces/python_bug_traces.md`.

## 9. Real-code repair with non-learned event pooling

The first end-to-end repair integration keeps the buggy source and failing test as text, selects at
most eight changed-state/control/boundary events from an actual Refactory execution, and replaces
every selected event with the mean of its frozen CodeLM token embeddings. Thus each variable-length
event occupies one model-native input slot. The frozen receiver is
`Qwen2.5-Coder-3B-Instruct`; generated functions are checked on the shown test plus ten held-out
instructor tests.

| Condition | Repair@1 | Mean total input slots | Mean generation seconds |
|---|---:|---:|---:|
| Test only | 12/20 (60%) | 200.3 | 6.07 |
| Selected event text | 9/20 (45%) | 351.4 | 4.54 |
| Mean-pooled event latent | 14/20 (70%) | 202.2 | 6.67 |
| Shuffled mean-pooled latent | 14/20 (70%) | 202.2 | 6.65 |

The selected trace summaries contain 155.1 tokens on average. Pooled latent avoids all three cases
where selected text breaks a correct test-only repair, and changes two test-only failures into
successful repairs. However, the same two wins occur with trace vectors from a different program.
True and shuffled latent have identical correctness on all 20 programs and byte-identical model
responses on 18/20. Therefore the apparent gain is generic prompt/activation steering, not use of
instance-specific execution evidence.

This is a useful negative gate: arithmetic mean pooling remains near the native embedding manifold
and is less disruptive than trace text, but destroys the binding among event role, source, and
value. The next repair experiment should first establish that a small learned latent code can steer
a frozen receiver to produce a complete validated patch, with shuffled-code and no-message
controls. Only then should a role-aware trace encoder be distilled into that code.

Artifact: `artifacts/repair_latent/refactory_q1_event_pool.jsonl`.

## 10. Full-patch latent-channel capacity

The mean-pooling result leaves two possible explanations: either the receiver cannot use a small
latent message for full program repair, or the pooling operation destroys event semantics. A
capacity gate separates them using four paired tasks. Within every pair, the buggy code and public
failing test are byte-identical, but hidden behavior differs: first versus last matching index, sum
versus product, maximum versus minimum, and even-value versus positive-value count. Consequently a
deterministic no-message receiver must emit the same patch for both sides and cannot solve both.

Eight latent slots are initialized from the frozen receiver's embeddings of a short behavioral
diagnosis. A 98,304-parameter residual codebook is then optimized for 240 steps through the frozen
`Qwen2.5-1.5B-Instruct` receiver, using full corrected-function teacher forcing. Generated patches,
not token loss, determine success on public plus hidden tests.

| Condition | Repair@1 (8 paired tasks) | Held-out prompt wording |
|---|---:|---:|
| No behavioral evidence | 2/8 | 2/8 |
| Diagnosis as text | 6/8 | 6/8 |
| Untrained native diagnosis anchors | 6/8 | 6/8 |
| **Trained latent code** | **8/8** | **8/8** |
| Paired opposite-behavior code | **0/8** | **0/8** |

Training took 165.9 seconds on one A100 while the receiver stayed frozen. The paired-swap result is
the first actual-repair evidence that the latent content controls which complete program is emitted:
generic activation steering cannot explain opposite codes selecting opposite hidden semantics.
It also establishes that the earlier real-trace failure is an encoder/representation problem, not
an eight-slot receiver-capacity problem.

The claim remains deliberately narrow. The codebook is indexed by the oracle behavior label and is
evaluated on the same eight programs (with a held-out instruction wording); it is not a learned
trace compressor and does not establish program generalization. The next gate must encode multiple
test executions—including inputs, expected/actual outputs, and typed causal events—into these
successful repair codes, with held-out evidence bundles and expected-value/trace-shuffle controls.

Artifact: `artifacts/repair_latent/repair_codebook_seed131.json`.

### Expanded paired-ambiguity capacity gate (2026-09-17)

We expanded the benchmark from four to twelve behavior pairs (24 repairs): ordering, length
selection, Boolean aggregation, duplicate selection, lower/upper median, rotation direction,
numeric transform, and filtering join the original locate, aggregate, extreme, and measure pairs.
Within each pair, buggy code and the public test remain identical, while hidden tests require
opposite repairs. The 98,304-parameter oracle behavior codebook was trained for 1,200 steps with
the receiver frozen.

| Condition | Train instruction | Held-out instruction wording |
|---|---:|---:|
| No behavioral evidence | 5/24 | 5/24 |
| Diagnosis text / native anchor | 15/24 | 15/24 / 14/24 |
| **Trained latent code** | **23/24** | **21/24** |
| Paired opposite-behavior code | **0/24** | **1/24** |

Thus the frozen 1.5B receiver can reliably use an eight-state latent message to select among
opposite full-function repairs, and the behavior-to-patch association survives an instruction
wording change. The one persistent difficult behavior is `locate_last`; held-out wording additionally
misses `truth_any_positive` and `unique_last`. This is still an *oracle-code capacity gate*: it shows
that a successful latent receiver exists, not that a trace encoder can infer those codes from runtime
events. The next experiment distills a role-aware trace encoder into this stronger 24-way target and
must recover a true-evidence versus paired-swap gap.

Artifact: `artifacts/repair_latent/expanded_codebook_seed313.json`.

## 11. Role-aware runtime-to-repair distillation

A 4.08M-parameter role-aware Transformer was trained to replace the oracle lookup. Its input is a
bundle of actual buggy executions with separate roles for test boundary, input, call, executed line,
branch, state definition, return, actual output, expected output, and test status. Event contents
enter through means of the frozen receiver's native token embeddings; role, test identity, and
event position remain separate. Eight cross-attention queries produce the eight repair-code slots.

Training used 512 generated bundles containing 2–4 tests per bundle and small value ranges. The
encoder was distilled for 1,200 steps into the successful frozen repair codebook. Evaluation used
128 new bundles with 5–8 tests, longer traces, and disjoint/wider value ranges.

Training-batch cosine reached approximately 0.93, but joint held-out performance did not pass:

| Joint held-out metric | Result |
|---|---:|
| Nearest repair-code accuracy | 47.7% |
| Target-code cosine | 0.698 |
| Full repair with true evidence | 3/8 |
| Full repair with paired behavior swap | 3/8 |
| Full repair with expected values removed | 2/8 |

The absence of a true-versus-paired-swap repair gap means the end-to-end trace encoder claim is not
yet supported. A factorial transfer audit isolates the failure:

| Evidence split | Nearest code | Cosine |
|---|---:|---:|
| New short bundles, in-range values | 78.9% | 0.903 |
| Long bundles, in-range values | 75.8% | 0.889 |
| Short bundles, OOD values | 47.7% | 0.721 |
| Long bundles, OOD values | 51.6% | 0.734 |

Length transfer is therefore not the main bottleneck: increasing from 2–4 to 5–8 tests costs only
3.1 points when values remain in range. Numeric/relational extrapolation is the failure. In the OOD
audit, max/min remain 81–100% accurate, whereas first-index, sum/product, and even-count degrade
sharply. Mean native embeddings preserve lexical value identity but do not give the small event
Transformer a reliable inductive bias for arithmetic, parity, or index relations outside the
training range.

The next intervention should add structured scalar and relational fields rather than more steps or
more latent slots: normalized scalar values, equality/delta features, branch outcomes, and explicit
data-dependence edges. It must retain the same paired-swap and role-removal controls. This result
also argues against treating arbitrary text-embedding averages as a sufficient "model-native"
runtime representation.

Artifacts: `artifacts/repair_latent/role_aware_repair_seed151.json` and
`artifacts/repair_latent/role_aware_transfer_audit_seed151.json`.
