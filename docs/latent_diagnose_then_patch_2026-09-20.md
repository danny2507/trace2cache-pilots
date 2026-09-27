# Latent Diagnose-then-Patch (LDP)

Working title:

> **Don't Say the Sketch: Latent Diagnose-then-Patch with a Frozen CodeLM**

One frozen CodeLM communicates an *edit sketch* to itself in native states,
then applies it. Not traces. Not a second agent. Not KV negotiation.

This supersedes Trace2Cache / DeltaCache as the paper idea. Keep the MBPP
hidden-test protocol, the frozen 3B, and the shuffle control. Change the
sender and the payload.

## One-sentence claim

A small frozen CodeLM can *apply* a one-line `REPLACE` it cannot *discover*
as a patch; communicating that sketch through its own diagnosis states — without
decoding it — is the latent-communication question that this setting can
actually answer.

## Why the previous idea is the wrong paper

Trace2Cache asked: can eight native states carry a *runtime trace* so that a
frozen 3B repairs better than well-prompted text?

On the frozen 128-task panel (`cohort_v1`, seed 1001, hidden-test Repair@1):

| Signal | Repair@1 | Evidence use |
| --- | ---: | --- |
| no public extra evidence | 68 | — |
| gold compact trace (~869 tokens) | 64 | text; net-negative |
| 8-slot const prefix | 72 | none (`true = shuffled`) |
| gist-of-trace latent | 54 | bound, harmful |
| edit-sketch ICAE from traces | 59 | bound, harmful (66/128 shuffle disagree); loses const floor |
| oracle `REPLACE` **in the prompt** | 114 | text upper bound, not protocol |

The decoder is not the bottleneck. The public `assert f(x) == y` already leaks
the intended value. First-order mutants leave little for LINE/STATE/RETURN to
add. Gold traces *hurt*. Binding the wrong payload *hurts more*. A generic
soft-prompt explains the only numerical beat of text.

So “latent runtime channel from an executor” is the wrong object on this
cohort. The object that moved the decoder is a **one-line edit sketch**.

Stage 0 is the existence proof: 114/128 when the prompt contains
`REPLACE \`old\` WITH \`new\`` from `ast.unparse` of canonical vs buggy.
Canonical is a train-only target and a Stage 0 diagnostic. It never enters a
test-split prompt.

## What this paper is

**Single-model latent diagnose-then-patch.**

```
Pass D (diagnosis)     Pass R (repair)
buggy + failing test   buggy + failing test + MARKER
frozen CodeLM          frozen CodeLM
do not decode Z   →    splice / continue Z → patch
```

- **Sender:** the same frozen `Qwen/Qwen2.5-Coder-3B-Instruct` during a
  diagnosis forward. Not a debugger. Not a peer agent.
- **Message:** an edit sketch (`REPLACE` / `DELETE` / `INSERT` / `NO_EDIT`),
  the thing Stage 0 showed the decoder can apply.
- **Channel:** native states of that diagnosis (learned gist tokens, or the
  diagnosis prefix KV of the *same* sequence). Not natural language. Not
  another model's cache.
- **Receiver:** the same frozen decoder generating a patch under the existing
  splice at `<TRACE2CACHE_REPAIR_CODE>`.
- **Supervision:** teacher-forced oracle sketch + edit-span patch NLL on the
  train split. Oracle sketch is derived offline from canonical vs buggy. Eval
  never sees it.

The research question is not “can agents agree in a latent social space.”
It is:

> If the useful intermediate is a short sketch this model can apply in text,
> does *not decoding* that sketch preserve or beat decoding it?

That is latent communication with a paired text control that the field
usually skips.

## What this paper is not

Do not overlap the multi-agent KV proposal (LaMAR-shaped: peer exchange,
sycophancy, CodeBERT+AST alignment of caches).

| Out of scope | In scope |
| --- | --- |
| Two or more coding agents | One frozen 3B, two passes |
| KV injection into another model | Gist splice into the same decoder, or continue the same sequence's KV |
| Social space / negotiation / sycophancy | Hidden-test Repair@1 |
| CodeBERT or AST similarity as a loss | Sketch NLL + shuffle ranking |
| Runtime traces as the payload | Edit sketch as the payload |
| Weakening the text prompt to fake a win | Self-sketch text is a required baseline |

Do not retune Stage C. Do not claim per-layer KV injection into a *different*
model. Canonical source never enters a test-split prompt or adapter update.

## Method

Frozen receiver, BF16, greedy decode, official MBPP ID split, public vs hidden
tests as in `docs/mbpp_generalization_protocol_2026-09-19.md`.

Sketch language (already implemented as `edit_sketch`): after `ast.unparse`,
a short `REPLACE \`old\` WITH \`new\`` (128/128 panel sketches are one line).
This is the communication *object*, not a loss decoration.

### Pass D — diagnosis

Chat: buggy program, public failing test, instruction to write only a
REPLACE/DELETE/INSERT line. **No canonical. No patch. No decoded sketch at
eval.**

Append `K` learned gist tokens (start with `K = 8` because that is the
channel we already isolated; Gate 1 may force `K` up or a different medium).

Read last-layer hidden at those `K` positions:

```
Z = W h_gist + b
```

Do **not** add a frozen const prefix. The const floor is a generic soft-prompt.
A residual on const already left that floor and hurt Repair@1. If LDP works,
it must work as a sketch carrier, not as a perturbation of skiphide.

### Pass R — repair

Same repair prompt as the protocol (specification omitted; public test kept
at eval; hidden at train). Splice `Z` at the marker. Generate the patch.
Hidden tests decide Repair@1.

### Losses

```
L = L_edit
  + λ L_sketch(Z → teacher-force oracle REPLACE)
  + μ relu(m + L_sketch(Z_true) − L_sketch(Z_other))
```

- `L_edit`: edit-span patch NLL (tokens outside the shared affix with the
  buggy program).
- `L_sketch`: teacher-forced oracle sketch through the splice. Short,
  instance-specific, the same class of target that Stage C could bind.
- `Z_other`: Pass D on **another task's** buggy program and failing test.
  Not this program with shuffled debugger events. The ICAE run showed that
  distinction matters once the encoder reads program text.

No vector reconstruction toward pooled embeddings. No cosine-separation
loss (skipsep collapsed Repair@1 to 49).

Train on in-budget train-split mutants. Canonical is the sketch and patch
*target only*.

### Why this can work when ICAE-from-trace did not

ICAE-from-trace asked the gist tokens to *infer* a REPLACE from a debugger
gist. Gold traces as text already lose to no_evidence, so that inference
target is the wrong sufficient statistic.

LDP asks the gist tokens to *hold* a diagnosis the same LM is computing
from buggy+assert — the inputs Stage 0 used — and to skip serializing it.
If the 3B's diagnosis hidden states already point at the edit (they must,
in the 114 tasks where a written REPLACE is enough), compressing those
states is a different problem from compressing LINE/STATE/RETURN.

## Baselines (all required)

Same buggy program, public test, decoder, token budget. Specification off.

| Condition | What is communicated |
| --- | --- |
| `no_evidence` | nothing |
| `compact_text` | gold runtime trace |
| `io_text` | failing assertion + pass/fail |
| `self_sketch_text` | Pass D **decoded** to a REPLACE line, then Pass R reads that line |
| `oracle_sketch` | Stage 0 upper bound; **not** in the protocol table |
| `true_latent` | Pass D states for this task |
| `shuffled_latent` | Pass D states for another task |
| `corrupted_latent` | Pass D on this task with the failing test / gist destroyed |

`self_sketch_text` is the paper-making control. Latent communication that
loses to its own decoded message is a soft-prompt or a compression failure.
Report it that way.

Do not drop `compact_text` to make latent look good. Gold traces are allowed
to lose; that is a finding, not a license to hide them.

## Decision rule

A latent *communication* claim needs all of:

1. Task-level Repair@1(`true_latent`) > Repair@1(`no_evidence`)
2. Repair@1(`true_latent`) > Repair@1(`compact_text`)
3. Repair@1(`true_latent`) > Repair@1(`self_sketch_text`)
4. A drop under `shuffled_latent` or `corrupted_latent`
5. No canonical / oracle REPLACE in any eval prompt

If (1)–(2) hold but (3) fails: the channel is not beating text of the same
message. If (4) fails: generic prefix, same as const 72. If ranking never
leaves the margin: the decoder is not reading `Z`.

Uncertainty unit is the task. McNemar on the 128-task panel.

## Experiments, in order

Do not train Gate 2 if Gate 1 fails. Do not reopen trace encoders.

### Gate 1 — channel capacity (oracle sketch as *encode input*)

Encode **only** the oracle REPLACE line (plus EOS). No buggy source, no
trace, no public test in the encoder. Train `Z` to reconstruct that line
and to help edit-span repair. Shuffle = another task's REPLACE.

This asks: can `K` spliced states *hold* a sketch the decoder can apply?

- **Fail** (no beat of `no_evidence`, or no shuffle drop): `K = 8`
  `inputs_embeds` cannot carry this payload. Stop splicing eight residual
  tokens. Go to Gate 3 or drop latent on this decoder.
- **Win** + shuffle drop: the channel is sufficient. The remaining problem
  is producing `Z` without the oracle. Go to Gate 2.

**Ran 2026-09-24, seed 1001. Call `ldp_g1_fail`.** Frozen 128, encode =
oracle REPLACE only, no const residual. Repair@1: `true_latent` 64,
`shuffled_latent` 66, `corrupted_latent` 64, `no_evidence` 68, `io_text`
67, `compact_text` 64. McNemar true vs no_evidence 14 vs 18, p=0.597.
True vs shuffled 4 vs 6, p=0.754 (shuffle did not drop). True vs
corrupted 2 vs 2. Cosine true/shuffled stayed ~0.99. Kill Gate 1. **Do
not train Gate 2.** Artifacts:
`artifacts/mbpp_generalization/ldp_gate1_seed1001/`. The ICAE encode was
buggy+test+trace; this run was REPLACE-only and still failed.

### Gate 2 — latent diagnosis (the actual method)

Pass D encode input = buggy + public failing test. No trace. No oracle.
`K ∈ {8, 16, 32}` only if 8 is tight at Gate 1. No const residual.

Eval against the full baseline table, including `self_sketch_text`.

Optional leakage ladder, same panel: full `assert f(x)==y` vs hide the
expected value vs traceback-only. Hypothesis: self-sketch and LDP both
move more when the prompt is not already a spec. If traces suddenly win
on the hidden-value prompt, that is a *different* paper (executor channel);
do not mix it into the LDP table.

### Gate 3 — same-sequence continuation (only if Gate 1 fails)

Do not inject KV into another model. Continue the *same* frozen decoder:

teacher-force or replace the sketch token embeddings with continuous
states (Coconut / pause-token style), then generate the patch. Shuffle
replaces that segment with another task's diagnosis segment.

This is still one model and still a sketch payload. It is the fallback
medium if eight spliced slots are too small.

## Implementation notes (this repo)

Reuse, do not rebuild:

- Frozen model, splice, hidden-test eval: `scripts/run_mbpp_generalization.py`
- Sketch + ICAE encode helpers: `edit_sketch`, `render_sketch_chat` in
  `src/trace2cache/mbpp_generalization.py`
- Protocol conditions vs `oracle_sketch` split (`PROTOCOL_CONDITIONS`)
- Cohort `artifacts/mbpp_generalization/cohort_v1/`
- Predeclared 128-task panel, ids 11–177

Change:

- Encode input for Gate 1 = oracle sketch text only
- Encode input for Gate 2 = buggy + public test, **no** `gist_text(events)`
- Drop frozen const residual
- Add `self_sketch_text` as a first-class condition (two greedy forwards)
- Shuffled encode = other example's Pass D, always

Keep: no Stage C retune, no killing other GPU jobs, `.venv/bin/python`,
commit only when asked.

## Related work, positioning

- **Text traces for APR** ([Haque et al. 2025](https://arxiv.org/abs/2505.04441)):
  raw traces rarely help. We reproduce that on hidden tests and show a
  *sketch* is the intermediate that does.
- **Self-debugging / reflexion:** decode a diagnosis, then a patch. LDP is
  the latent analogue with a mandatory decoded-sketch control.
- **ICAE / gist / Coconut / pause tokens:** compression and continuous
  thought *inside one model*. That is the family. The payload is an APR
  edit sketch, the metric is hidden-test Repair@1, the control is shuffle
  plus self-sketch text.
- **KVCOMM / LatentMAS / LaMAR-style multi-agent KV:** peer cache exchange
  for speed or consensus. Different sender, different claim, different
  supervision. Do not occupy that slot.
- **Trace2Cache / DeltaCache (this repo):** negative result on executor
  payloads. Cite as motivation for changing the message, not as the method.

## Risks

- **Gate 1 fails.** Then eight spliced states are the wrong medium for a
  sketch. That is a result. Do not hide it by weakening `self_sketch_text`.
- **Gate 2 equals const.** Diagnosis hidden states collapse to a task-
  independent prefix. Shuffle will catch this.
- **Gate 2 beats no_evidence but loses to self-sketch.** Honest: the sketch
  is the right message and text is a better encoding of it for this K.
  Still publishable as “apply ≠ discover” plus a failed latent channel;
  not a communication win.
- **Self-sketch already near 114.** Then latent has little headroom. Report
  the split: discover vs apply. The paper still holds without a latent win.
- **Leakage.** Any eval path that includes the oracle REPLACE is Stage 0
  again. Keep `oracle_sketch` out of default `full` mode.

## Frozen numbers this idea depends on

Do not rerun these to “set up” LDP. They are already in
`artifacts/mbpp_generalization/RESULTS_2026-09-19.md`.

- no_evidence 68, compact 64, io 67, const 72
- gist 54, sketchicae 59
- oracle_sketch 114 (Stage 0)
- 47 tasks unsolved by NE ∪ compact ∪ const; oracle solves 35 of them

LDP is aimed at those 35: communicate the sketch the decoder can apply,
without writing it down, without a second agent, without a trace.
