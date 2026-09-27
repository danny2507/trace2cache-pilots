# Latent execution channel for CodeLM repair

Working title:

> **Don't Prompt the Trace: A Fixed-Rate Native Runtime Channel for Small Code Models**

This is the latent analogue of Haque, Babkin, Farmahinifarahani, and Veloso,
[*Towards Effectively Leveraging Execution Traces for Program Repair with Code
LLMs*](https://arxiv.org/html/2505.04441v1) (arXiv:2505.04441). They put
execution traces **in the prompt**. We put the same evidence into **K native
states** of a frozen CodeLM.

Not multi-agent. Not a gold `REPLACE`. Not a second model. Sender = the
program executor (failing test + trace). Receiver = frozen CodeLM. Channel =
`inputs_embeds` splice, not text.

## What that paper actually showed

Models: GPT-3.5 Turbo and GPT-4. Data: Refactory (~2k student Python),
RunBugRun (1k CodeNet Python), HumanEval-Java.

Traces: PySnooper (assignments, calls, returns, lines), timestamps stripped.
**Error prompt** = buggy code + failing test (`Expected 6 but got None`).
**Trace prompt** = error prompt + raw trace. **OPT** = GPT-4-32k rewrites the
trace into a shorter prompt for APR. **Collated** = trace comments on source
lines. Fine-tune baseline: deepseek-coder-1.3b on <600 examples.

Numbers that matter:

- Raw traces are **not** a reliable upgrade over the error prompt. GPT-3.5
  drops on all three datasets (e.g. Refactory CFA 0.525 → 0.509). GPT-4 drops
  on Refactory (0.791 → 0.737), gains on HumanEval-Java (0.493 → 0.511) and
  RunBugRun (0.529 → 0.558).
- Longer traces and more assignments correlate with *worse* repair. 5–10% of
  trace prompts truncate (traces > 10k events exist).
- OPT (GPT-4 as a **text compressor**) is the only consistently decent trace
  variant.
- They still conclude traces contain facts GPT-4 does not infer from code +
  error alone (collating probe 45–88% vs 15–50% from code).

Their leftover problem, which they solve with another LLM in the prompt: **the
trace is useful in principle and harmful as raw tokens.**

That is exactly a fixed-rate channel problem.

## Claim

For a frozen small CodeLM, map a failing execution to `Z ∈ R^{K×d}` in the
model's embedding space and splice `Z` at a named marker. Compare this to
Haque's prompt stack on the **same** buggy program, failing test, decoder, and
hidden tests.

```
executor:  run public failing test → events T
encoder:   T ↦ Z  (K slots, trained; CodeLM frozen)
decoder:   p(patch | buggy, failing test, Z)
```

RQ1. Where a matched **text** trace already beats the error prompt, can `Z`
match or beat that text at fixed K (and beat raw traces that truncate)?

RQ2. Can `Z` replace Haque's GPT-4 OPT rewriter — i.e. compress the trace
without a second frontier LM in the prompt?

RQ3. Does the decoder *use* `Z` (shuffle / corruption drop), or is this
another generic prefix?

If text traces do not beat the error prompt on a cohort, **stop**. Do not
encode a signal the decoder does not want as text. That is the MBPP
first-order lesson (compact 64 < no_evidence 68).

## Why this is latent communication

Communication is executor → CodeLM, not model → model. The message is
runtime evidence that never becomes tokens in the repair prompt (except the
failing test, which Haque also shows). K is fixed while |T| varies. That is
the point of OPT, without paying GPT-4-32k per bug.

A 3B is the right receiver: Haque's GPT-4 already infers a lot from
code+error, so traces are marginal. A 3B infers less; context flood from a
869–10k token trace is worse. Their 1.3B fine-tune lost to prompting. We
keep the 3B **frozen** and only train the encoder, so a win is a channel win
not a new repair model.

## Method

Frozen `Qwen/Qwen2.5-Coder-3B-Instruct` BF16, greedy decode, splice at
`<TRACE2CACHE_REPAIR_CODE>`. Encoder is the existing role-aware event
encoder (or a thin ICAE over **trace text**, not over a gold diff). Trainable
parameters stay in the encoder.

Events stay executor-native: TEST / FAIL / PASS, LINE, STATE, RETURN, CALL,
BRANCH — the PySnooper analogue this repo already builds. Do not invent a
REPLACE language. Do not put canonical source in any eval prompt.

### Prompt / channel conditions (matched)

Haque's stack, plus latent, plus evidence-use controls:

| Condition | Evidence |
| --- | --- |
| `error` | failing test only (their Error Prompt; our `io_text` / `no_evidence` split: keep the failing assert, drop the trace) |
| `raw_trace` | full public execution trace in the prompt (their Trace Prompt; our `compact_text`) |
| `opt_text` | a **frozen** compressor's short text (either a local 3B rewrite of the trace, or a rule-based slice; do **not** require GPT-4-32k). This is the fair OPT analogue. |
| `true_latent` | K states from this run's events |
| `shuffled_latent` | K states from another program's run |
| `corrupted_latent` | I/O kept, intermediate payloads destroyed |

Optional: `collated_text` if cheap (trace as comments on lines). Do not add
it if it blows the schedule.

Repair@1 = hidden tests, all of them. Shown failing test is public evidence,
never the metric.

### Losses

Repair NLL on the train-split patch (edit-span if the bug is local).
Contrastive ranking on **repair or short runtime readout** so that
`Z_true` is better for *this* fail than `Z_other`. No cosine-separation
loss (already shown to wreck Repair@1). No gold-diff reconstruction.

If ranking on full-patch NLL saturates (it did on MBPP because the public
assert already determines most tokens), rank on a short **runtime gist of
the trace itself** (failing assert, observed value, last state) — as a
*binding* loss, not as a repair instruction. The repair prompt still does
not dump the raw trace.

### Decision

Latent communication of the trace requires:

1. `true_latent` > `error`
2. `true_latent` ≥ `raw_trace` (strictly > if raw traces truncate or are
   long; ≥ if they are short)
3. shuffle or corruption drop
4. report vs `opt_text`: beating it is the strong claim (replace GPT-4 OPT);
   matching it without a rewriter is already a result

If `raw_trace` ≤ `error` on the cohort, do not interpret a latent number.
Change the cohort.

## Cohort (this is the actual design choice)

Do **not** reuse the 128 first-order MBPP panel as the headline. There,
gold traces lose to no extra evidence. Encoding them cannot be a win.

**Gate 0 — text first, on a dataset where Haque says traces can matter.**

In order:

1. **Refactory beyond Q1 n=20.** Frozen as `refactory_cohort_v1`: all five
   questions, public = first failing instructor assert, hidden = the rest,
   reference stored as `target_source` only. Gate 0 panel = first 25 eligible
   test-split examples per question. If `gist_text` still ≈ `error` after a
   fair token budget, Refactory is the wrong headline (Haque's GPT-4 also
   *lost* on Refactory traces). Then RunBugRun.
2. **RunBugRun Python sample.** Haque's GPT-4 *gained* here (0.529 → 0.558).
   Best external match to their paper.
3. **MBPP only as a stress test**, and only with the expected value hidden
   (`assert f(x) == y` → fail without `y`, or traceback only). If traces
   still lose, first-order mutants are the wrong bugs, not just the wrong
   prompt.

Headline numbers come from the first cohort where Gate 0 passes
(`raw_trace` > `error` by McNemar, not a 1-point blip). Latent is then
trained and evaluated on that cohort with a disjoint split. Hidden tests
stay hidden.

## What we already know not to redo

- First-order MBPP + public `== y` in the prompt: traces hurt; 8-slot const
  prefix matches a trained encoder; binding a gist of the trace hurts.
- Cosine separation as a loss: Repair@1 49.
- n=20 Refactory Q1: most JSON vs shuffled-JSON agreement; do not publish
  n=20.
- Soft prefix vs KV: still an interface ablation *after* the encoder binds
  instance-specific evidence. Not the first experiment.

Keep: frozen 3B, K=8 as the starting rate (Haque's whole point is length
hurts; 8 is a hard compression), shuffle/corruption, hidden-test Repair@1,
no canonical in prompts, no Stage C retune.

Raise K (16/32) only if Gate 0 passes and K=8 binds but underfits OPT-text.

That raise-K clause did not trigger. Execution-sim Gate 0 passed for
compact intermediate *text*; K=8 of that payload failed to bind
(`latent_fail`); one native embedding per event also failed
(`native_fail`, shuffle 0–0). Do not revive K=8 or a trained compressor
on this splice. See `docs/latent_chain_of_execution_2026-09-21.md`.

## Gate 0 on this machine (2026-09-20)

`compact_text` (raw 869-token traces) already lost 64 vs 68 `no_evidence`.
Haque's actual positive text finding is **OPT**, not raw traces. The missing
measurement is prompting the short gist (I/O + last LINE/STATE/RETURN), which
was never a text condition on the 128 panel.

Conditions:

| Condition | Haque analogue |
| --- | --- |
| `no_evidence` | error prompt without dumping extra I/O events |
| `io_text` | error prompt + TEST/FAIL events |
| `compact_text` | raw Trace Prompt |
| `gist_text` | rule-based OPT (deterministic slice, not GPT-4) |
| `collated_text` | collated comments on source lines |

Default `PROTOCOL_CONDITIONS` is unchanged. These two extra text views are
`HAQUE_TEXT_CONDITIONS` and only run when named.

```
.venv/bin/python scripts/run_mbpp_generalization.py \
  --mode text \
  --conditions no_evidence io_text compact_text gist_text collated_text \
  --output-dir artifacts/mbpp_generalization/haque_text_seed1001
```

Decision: train a K-slot trace encoder **only if** `gist_text` or
`collated_text` beats `no_evidence` and `io_text` on this panel. If they
lose, do not retrain gist/skip/ICAE here.

### Result (seed 1001, 128-task panel)

`gist_text` 63, `collated_text` 60, `compact_text` 64, `io_text` 67,
`no_evidence` 68. Mean prompt tokens 296 / 452 / 869. McNemar gist vs
no_evidence 5–10 (p = 0.30); collated vs no_evidence 4–12 (p = 0.077);
gist vs compact 4–5 (p = 1.0). Overlapping this-run baselines agreed with
`const_seed1001` on 16/16 `no_evidence`, 16/16 `io_text`, 15/15 `compact_text`.

Gate 0 **fails** on this first-order MBPP panel. A short text trace is not
an upgrade over the error prompt, so eight native states of the same
events cannot be a communication win here. Do not retrain the gist or
ICAE encoder on this cohort. Next measurement belongs on a cohort where
text traces already help (Haque's RunBugRun Python gain), not on another
encoder variant of these 128 mutants.

`artifacts/mbpp_generalization/haque_text_seed1001/`.

### Result (Refactory, all five questions, 2026-09-20)

Frozen panel `refactory_cohort_v1`: 119 student programs (25/25/25/25/19).
Same 3B, hidden instructor tests, specification omitted.

`no_evidence` 53, `io_text` 45, `gist_text` 44, `compact_text` 42,
`collated_text` 39. Mean prompt tokens 199 / 244 / 303 / 720 / 481.
McNemar vs no_evidence: gist 2–11 p = 0.0225; collated 2–16 p = 0.0013;
compact 3–14 p = 0.0127. Q4 (`sort_age`) 14 vs 7 gist.

Gate 0 **fails** on Refactory. Text traces are a significant loss, same
direction as Haque's GPT-4 (Refactory error 0.791 > trace 0.737). Do not
train a K-slot encoder here. Next measurement is RunBugRun Python, where
their GPT-4 gained (0.529 → 0.558).

`artifacts/mbpp_generalization/refactory_text_seed1001/`.

### Result (RunBugRun Python, 2026-09-20)

Frozen panel `runbugrun_cohort_v1`: 128 official test-split Python bugs (43
problems, language=5). Same 3B, hidden stdin/stdout tests, specification
omitted. Worker records expected vs got; gist leads with `DISCREPANCY`.
`eval(input())` patches re-scored after generation (29 rows, 11 new passes).
Remaining extraction failures are truncated SyntaxError (28 tasks fail all
five conditions at `max_new_tokens=256`). Peak 15.6 GiB.

`gist_text` 28, `compact_text` 26, `io_text` 24, `no_evidence` 23,
`collated_text` 21. Mean prompt tokens 573 / 2464 / 456 / 344 / 911.
McNemar: gist vs no_evidence 6–1 p = 0.125; gist vs io_text 4–0 p = 0.125;
compact vs no_evidence 6–3 p = 0.508; collated vs no_evidence 3–5 p = 0.727.

First cohort where a short discrepancy note is *ahead* of the error prompt,
same direction as Haque GPT-4 on RunBugRun (0.529 → 0.558). Not McNemar.
Gate 0 **fail**. Do not train a K-slot trace encoder. The six gist-only
wins are expected-vs-got uses (float `1500.0`, extra newline, printed dict,
NameError, rectangle layout, cipher `None`).

`artifacts/mbpp_generalization/runbugrun_text_seed1001/`.

Fallback if a trace encoder is off the table: do not dump traces into K
slots. Encode a residual Δ of (expected, got) onto the last prefill state,
train on the tasks `no_evidence` misses, and use shuffle as the
evidence-use control.

Implemented 2026-09-21 as `DiscrepancyResidual`: last hidden of the repair
prompt before the marker, plus a zero-init MLP on `[h, e, g, e-g]`, K=8
slot offsets. Train split is `runbugrun_cohort_v1_residual/train.jsonl`
(527 official-train bugs, 78 problems, zero overlap with the frozen 128
test panel). `--train-misses-rows` keeps only no_evidence failures
(356 in-budget misses; train no_evidence 152/527). Shuffle keeps this
program's prefill and swaps the I/O pair. Corruption swaps expected and
got. Do not overwrite `runbugrun_cohort_v1`.
`src/trace2cache/discrepancy_residual.py`.

Result (`runbugrun_residual_seed1001`, seed 1001, 800 steps, best 680):
`true_latent` 15, `shuffled_latent` 16, `corrupted_latent` 16 vs Gate 0
`gist_text` 28 and `no_evidence` 23. McNemar true vs no_evidence 2–10
p = 0.0386; vs gist 1–14 p = 0.00098; vs shuffle 0–1 p = 1.0. Ranking
never left the margin (ema 0.099); cosine stayed 1.0; 126/128 patches
identical under shuffle. `h + Δ` was dominated by the broadcast prefill.
Residual-Δ **fail**. Not evidence use, and it hurts versus the public test.

v2 (2026-09-21): drop the prefill add. `Z_k = α · RMSNorm(MLP([e, g, e-g]))
+ offset_k`. Same 356 misses, same frozen 128. Best step 380, ranking
0.065, cosine 0.888 (step-1 cosine 0.64). `true_latent` 21,
`shuffled_latent` 21, `corrupted_latent` 20. McNemar vs no_evidence 2–4
p = 0.6875; vs gist 2–9 p = 0.065; vs shuffle 1–1 p = 1.0. Sources
119/128 identical under shuffle (v1 was 126/128). Recovered 0/6 gist-only
wins. Predeclared call: **prefix fail**. Geometry made Δ visible; Repair@1
still does not use this-bug I/O. Stop this line.
`artifacts/mbpp_generalization/runbugrun_residual_v2_seed1001/`.


## Positioning vs Haque (the related-work paragraph)

Haque et al. study *how to write traces into a prompt* for GPT-3.5/4. Raw
traces are inconsistent; LLM-optimized traces work better; length hurts;
truncation is real. We study the same evidence as a **native, fixed-rate
channel into a frozen small CodeLM**. If K states match OPT-text, the
GPT-4 rewriter is unnecessary. If they beat raw traces, the truncation /
length penalty is the thing being removed. Shuffle is the control their
prompt study does not have.

Not a multi-agent KV paper. Not an oracle-diff paper.

## Risks

- Gate 0 never passes on any Python cohort we can run. Then traces are the
  wrong payload for repair, including Haque's OPT (which may be GPT-4
  *reasoning* tagged as a trace rewrite). Stop and write that, with their
  table and ours. That is still a result; it is not latent communication.
- OPT-text with a local 3B rewriter is weak compared to their GPT-4-32k.
  Say so. The fair claim is “vs a matched local rewrite,” not “vs GPT-4.”
- Encoder collapses to a constant prefix. Shuffle catches it; we have seen
  this.
- 3B cannot use traces even when GPT-4 can (Haque GPT-3.5 often degraded).
  Then the paper becomes “small CodeLMs need a different runtime encoding”
  — still the channel paper, but Gate 0 must use the **3B** error vs raw
  trace, not GPT-4's.

## First week

1. Implement Haque's `error` vs `raw_trace` on Refactory (all available
   questions) and a RunBugRun Python slice, frozen 3B, hidden tests, no
   training.
2. Pick the cohort where `raw_trace` beats `error`.
3. Only then train the K-slot encoder on that train split.

Do not train on the 128-task MBPP panel, Refactory, or RunBugRun until a
text condition beats `no_evidence` and `io_text` by McNemar. All three
Gate 0s are done. RunBugRun is the only non-negative gist result (28 vs
23, 6–1 p = 0.125). Residual-Δ v1 failed (15 vs no_evidence 23, shuffle
0–1). Residual-Δ v2 (unit-RMS, no prefill add) is a prefix fail: true 21,
shuffle 21, vs no_evidence 23 and gist 28, shuffle 1–1. Do not train a
K-slot trace encoder. Do not claim either residual as a channel. Stop
this residual-Δ line.

Failing-line Gate 0 (`line_text`, 2026-09-21): last executed source
statement as one extra text line, no I/O restatement. `line_text` 20 vs
`no_evidence` 23 (2–5 p = 0.453) and `gist_text` 28 (2–10 p = 0.0386).
Recovered 0/6 gist-only wins. Gate 0 **fail**. Do not train a
failing-span residual. The frozen 3B does not use traces as tokens, as
K prefix states, as `(e, g)` residual, or as a named failing line.
`artifacts/mbpp_generalization/runbugrun_line_text_seed1001/`.

Discrepancy one-liner then embed splice (2026-09-21):
`discrepancy_text` 23 vs `no_evidence` 23 (3–3 p = 1.0), recovers 3/6
gist-only wins. Zero-train splice of those token embeddings:
`discrepancy_embed` 23, `shuffled_embed` 24, `corrupted_embed` 22.
Embed vs text **0–0**; vs shuffle 1–2 p = 1.0. Splice is faithful (the
3B can read its own embeddings at the marker). Repair@1 does not use
this-bug I/O except 2035118 (`// 2` vs `/ 2`). Prefix fail. Do not
compress into K=8. `artifacts/mbpp_generalization/runbugrun_discrepancy_embed_seed1001/`.

7B Gate 0 (2026-09-21): frozen `Qwen/Qwen2.5-Coder-7B-Instruct`, same
128, seed 1001, greedy, spec omitted, public test kept. Peak 16.35 GiB.
`io_text` 42, `gist_text` 41, `no_evidence` 40. gist vs no_evidence 5–4
p = 1.0; gist vs io 3–4 p = 1.0; io vs no_evidence 4–2 p = 0.6875. Vs
3B: no_evidence 40 vs 23 (22–5 p = 0.0015). Five of six 3B gist-only
wins now pass 7B no_evidence; 2135533 passes no_evidence and **fails**
gist. Gate 0 **fail**. Do not train a 7B encoder. Bigger model infers
more from code+error; traces add nothing. Do not start 14B unless asked.
`artifacts/mbpp_generalization/runbugrun_text_7b_seed1001/`.

Execution-sim Gate 0 (2026-09-21): same 128 gold programs, predict
stdout from code+stdin, not a patch. Frozen 3B. `trace_text` 25 vs
`code_input` 12 (15–2 p = 0.0023). `trace_gist` 14 (5–3 p = 0.73).
First McNemar win for runtime text. The message is intermediate
execution of gold code, which is necessary for the answer.

K=8 of that payload on a disjoint 320-example train split
(2026-09-21): `true_latent` 14, `shuffled_latent` 14, `code_input` 12,
`trace_text` 25. True vs shuffle 0–0 p = 1.0 (identical on all 128).
True vs text 5–16 p = 0.027. Call **latent_fail**. Compact
intermediate text is used; eight slots of it are not. Do not claim a
channel. Do not train gist. Do not train on the 128.
`docs/latent_chain_of_execution_2026-09-21.md`.
`artifacts/mbpp_generalization/execution_sim_latent_seed1001/`.
