# Idea exploration: from execution trace to a native runtime channel

## Bottom line

The strongest paper is not "compress debugger text into KV." That version is vulnerable to the
objection that the gain comes from learned summarization or extra parameters. The stronger object of
study is a **fixed-rate runtime channel** between a program executor and a frozen CodeLM:

\[
M_\phi(C, Q, T_{1:n}) \rightarrow Z \in \mathbb{R}^{K\times d}, \qquad
p_\theta(\text{patch}\mid C,Q,Z), \quad \theta\ \text{frozen}.
\]

Here `C` is buggy code, `Q` is the failure query/assertion, and `T` is one or more executions. `K`
is fixed while raw trace length varies. A soft prefix and direct KV states are two implementations of
the same channel, not interchangeable terminology.

The most promising payload is not a single raw trace. It is a **query-conditioned behavioral delta**:

> What runtime behavior differs between passing and failing executions, restricted to dependencies
> that can reach the failed assertion?

Working method name: **DeltaCache**. Working paper title:

> **Beyond Debug Logs: Fixed-Rate Latent Runtime Feedback for Small Code Models**

`Trace2Cache` is descriptive but collides semantically with the established CPU term “trace cache.”

## Why this variant is more defensible

Prior work already covers execution-trace embeddings, execution-aware neural architectures, textual
trace prompting, trace-guided repair, and generic cache-to-cache model communication. A defensible
gap needs all of these qualifiers together:

- external runtime evidence rather than another LM's hidden state;
- an arbitrary pretrained decoder that remains frozen;
- fixed communication capacity as raw evidence grows;
- multiple passing/failing executions and their causal contrast;
- controlled separation of slicing, compression, and injection medium;
- downstream repair verified by hidden tests.

Dropping any one qualifier makes the idea much closer to existing work.

## Proposed representation: Query-Conditioned Trace Delta (QCTD)

### 1. Build runtime events

Each event contains:

```text
(test_id, outcome, time, event_type, source_span,
 variable_id, value_summary, control_parent, data_parents)
```

Object identity and source spans should be stable within an execution. Values need typed summaries
(integer sign/magnitude, string length/hash, collection shape/sample), with raw values retained only
when small. This avoids teaching the encoder incidental `repr` formatting.

### 2. Query-conditioned dynamic slice

Begin at the failed assertion's observed value and walk data/control predecessors. This is an
information-selection baseline, not part of the latent claim. The same selected events must be used
for slice-text and slice-latent conditions.

### 3. Align pass and fail behavior

Align events primarily by source span and variable role, secondarily by temporal order. For every
source span derive delta features such as:

- executed only on pass / only on fail;
- visit-count difference;
- branch-outcome difference;
- last-definition value difference;
- distance to failure;
- whether the span lies on the backward dynamic slice.

This yields a delta graph rather than two long sequences. When no useful passing test exists, add a
learned `NO_REFERENCE` node and encode the failing slice alone.

### 4. Resample to K latent messages

A 2–4 layer event Transformer or graph Transformer processes the delta graph. `K` learned queries
cross-attend over event states (Perceiver-style resampling), producing a fixed number of messages.
Condition the queries on the failed assertion and buggy code so the bottleneck preserves relevant
facts rather than globally summarizing execution.

## Make the representation model-native

Do not learn every event symbol from scratch. Reuse the frozen CodeLM's input embedding table:

- tokenize source spans and variable names, then pool their native embeddings;
- initialize event-type fields from canonical words such as `call`, `branch`, `return`, `expected`;
- encode small scalar values with their native token embeddings plus typed numeric features;
- project only structural/numeric fields that have no textual equivalent.

This gives “model-native” a measurable definition. Ablate native initialization against randomly
initialized event embeddings.

## Injection ladder

### A. Soft cache tokens (first)

Project the K messages to model width and concatenate them to ordinary prompt embeddings. Run the
frozen decoder normally. The model itself turns these virtual positions into valid per-layer K/V
states, so RoPE, GQA head shapes, masks, and cache updates remain correct.

This is the cleanest feasibility test. It is a learned input-embedding interface and must be called a
soft prefix, even though its states subsequently live in the KV cache.

### B. Cache compilation (second)

Once A works, distill its per-layer prefix cache into a cheap projector that emits selected-layer
K/V blocks directly. Train with:

\[
\mathcal{L}=\mathcal{L}_{patch}
+\lambda_{KL} KL(p_{soft}\Vert p_{KV})
+\lambda_h \sum_l \|h^l_{soft}-h^l_{KV}\|_2^2.
\]

Use a zero-initialized gate per injected layer. This turns direct KV injection into a **compiled
approximation of a known-good latent interface**, rather than asking one mapper to solve semantics,
alignment, and cache mechanics simultaneously.

### C. Native KV channel (paper endpoint)

At inference, the executor emits K messages, the compiler emits layer K/V blocks, and the CodeLM
decodes the patch without any trace tokens. Measure encoder plus transfer plus generation latency;
excluding encoder cost would make the efficiency claim invalid.

## Training curriculum

### Stage 0: oracle-use test

Before training, provide a concise oracle diagnosis in text (for example, “the accumulator never
updates on all-negative inputs”). If the frozen SLM cannot repair from that, the downstream target is
too hard or the prompt/decoder is wrong.

### Stage 1: elementary trace probes

Train the adapter on questions whose labels are computed directly from execution:

- value after a source line;
- branch taken and visit count;
- last definition of a variable;
- exception origin;
- observed versus expected assertion value;
- which pass/fail execution first diverges.

Split by program template/problem, not by trace, and test on longer traces than training.

### Stage 2: causal sufficiency probes

Mask one delta component and ask for it, predict membership in the failure slice, and retrieve the
source span causally closest to failure. These prevent the bottleneck from learning only labels such
as bug family or pass/fail.

### Stage 3: repair

Use synthetic mutations for scale, then evaluate on a real benchmark. Train only the event encoder,
resampler, and projector. Keep the CodeLM frozen for the main table; a LoRA receiver can be a
secondary upper bound.

## The experiment that can make the paper

Construct matched evidence from the exact same sliced events and vary both raw event count `L` and
channel budget `K`:

| Condition | Evidence selection | Medium | Budget |
|---|---|---|---|
| Test only | none | text | 0 |
| Full text | full | text | grows with L |
| Slice text | causal slice | text | grows with L |
| Budgeted slice text | causal slice | text | K-token matched |
| Full latent | full | soft/KV | fixed K |
| Slice latent | causal slice | soft/KV | fixed K |
| Delta latent | pass/fail delta | soft/KV | fixed K |
| Shuffled latent | mismatched trace | soft/KV | fixed K |

The key result is a condition-by-length interaction on held-out Repair@1. Aggregate accuracy alone
cannot support “text is the wrong interface.”

Three especially diagnostic comparisons are:

1. `slice latent > slice text`: evidence is held fixed; medium differs.
2. `delta latent > full latent`: behavioral contrast matters beyond compression.
3. `true latent > shuffled latent`: the adapter carries instance-specific evidence.

Add a **causal intervention test**: change one runtime value while keeping code and all other event
fields fixed. The predicted diagnosis/patch should change in the direction implied by that value.
This is stronger evidence than a linear probe.

## Failure modes likely to fool us

- **Bug-family shortcut:** synthetic mutations reveal the fix from code alone. Require hard cases
  where identical code patterns produce different needed fixes under different tests/specifications.
- **Expected-output leakage:** the public assertion alone may reveal the patch. Include paired cases
  with the same failing output but different causal paths.
- **Extra-capacity confound:** compare against learned prefix tokens without trace and adapters with
  shuffled traces; report trainable parameters.
- **Slicer confound:** use identical selected events in text and latent conditions.
- **Length confound:** position or truncation, rather than noise, may hurt text. Report whether source
  code/test remain in context and include position-matched padding controls.
- **KV mechanics artifact:** compare direct KV against the soft-prefix teacher and unit-test logits
  when the direct cache is constructed from genuine token prefixes.
- **Plausible-patch inflation:** hidden tests and semantic deduplication are mandatory.
- **Encoder cost omission:** include runtime collection, slicing, encoding, transfer, prefill, and
  decode in end-to-end latency.

## Fast 72-hour decision plan

### Day 1

- Run environment/unit smoke tests.
- Evaluate 30–50 curated bugs with test-only, oracle diagnosis, compact trace, and long trace.
- Stop if oracle diagnosis does not move the SLM or compact trace contains no usable signal.

### Day 2

- Generate 5k–20k deterministic trace-probe examples from program templates.
- Overfit 64 examples with K=8 soft tokens; then train a template-held-out split.
- Stop if the adapter cannot near-perfectly overfit elementary value/branch probes.

### Day 3

- Compare K in {4, 8, 16, 32}, token-budget-matched text, and shuffled traces.
- Only if H2 passes, start repair-loss training and direct-KV cache compilation.

## Literature position checked in September 2026

- Haque et al., KnowledgeNLP 2025: textual traces give inconsistent repair gains and degrade with
  trace complexity.
- Wang et al., Findings EMNLP 2025: broad study finds limited usefulness from current execution-trace
  integration for SFT and test-time use.
- Gupta et al., NeurIPS 2020: explicit execution-state embeddings inside a specialized neural
  synthesis/repair architecture; this blocks any “first trace embedding” claim.
- Dynamic Neural Program Embedding (2017), Test2Vec (2022), TraceFixer (2023), TRACED (2023), and
  NExT (2024) occupy adjacent execution-representation or execution-reasoning territory.
- Cache-to-Cache, ICLR 2026: generic learned KV projection/fusion between LMs; it motivates the
  medium but does not itself establish an external program-runtime-to-CodeLM interface.
- CausalRepair (August 2026): dual slicing produces compact causal repair context, making slice-text
  a mandatory strong baseline rather than a novelty claim.

No exact-match result for the phrase/name `Trace2Cache` appeared in a targeted web search, but that
is not a novelty guarantee. A proper systematic review and citation graph search is still needed
before making a first-of-kind claim.
