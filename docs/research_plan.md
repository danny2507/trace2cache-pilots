# Trace2Cache research plan

## Claim boundary

The defensible novelty target is not "embedding execution traces." Neural trace embeddings predate
large code models. The target is a **model-native, fixed-budget interface that injects runtime
evidence into a frozen pretrained CodeLM**, with controlled evidence that the communication medium
matters beyond trace filtering.

The project must distinguish three effects:

1. **Information selection:** full trace versus dynamic slice or pass/fail delta.
2. **Compression:** variable-length evidence versus a fixed K-state bottleneck.
3. **Interface:** text tokens versus input embeddings versus per-layer KV states.

Calling a learned soft prompt "KV communication" would blur (2) and (3), so results must retain
separate names.

## Hypotheses and gates

### H1: text complexity phenomenon

At fixed code, test, decoding, and model, repair accuracy or trace-probe accuracy declines as the
serialized trace grows or accumulates irrelevant variable updates.

**Gate:** reproduce a statistically visible length/complexity interaction on at least two seeds or
two SLM sizes. If compact text never helps over test-only, runtime evidence or prompt design is the
first problem; do not train a latent adapter.

### H2: fixed-budget latent sufficiency

A K-token trace message answers held-out state, branch, last-definition, and failure-cause probes
better than token-budget-matched textual evidence.

**Gate:** K in {8, 16, 32}; frozen decoder; adapter-only training; split by program template so the
encoder cannot memorize surface forms. Require gains on value and dependency probes, not only
pass/fail classification.

### H3: downstream repair

Latent runtime messages improve held-out Repair@1 and robustness to trace length for a frozen SLM.

**Gate:** compare test-only, raw text, structured text, dynamic-slice text, soft prefix, full-trace
KV, and slice/delta KV. Match evidence, generation budget, and candidate validation.

## Pilot sequence

### Pilot 1: no training

Run deterministic generation on small synthetic repair cases first. This is infrastructure and
directional evidence, not a paper result. Record:

- hidden-test Repair@1;
- input token count;
- generated token count;
- wall-clock generation time;
- runtime event count and serialized trace bytes.

Then migrate the same harness to a published repair benchmark. Avoid drawing conclusions from fewer
than 100 bugs.

### Pilot 2: trace probes with a soft-prefix encoder

Represent each event as fields: event type, source location, variable/object identity, abstracted
value, control predecessor, data predecessor, test identity, and outcome. A small event Transformer
cross-attends from K learned queries and projects to the target model embedding width. Prepend the K
vectors through `inputs_embeds`; train only the event encoder/projector.

Required controls:

- K ordinary learned prompt tokens without trace input;
- K token budget of text selected by the same slicer;
- shuffled trace-to-example pairing;
- values masked, dependencies masked, and event order shuffled;
- trainable parameter and FLOP accounting.

### Pilot 3: KV injection

Only after soft-prefix sufficiency is established, map the K bottleneck to keys and values at
selected decoder layers. Start with one middle layer and a zero-initialized gate, then ablate layer
sets. Validate cache shape, rotary-position semantics, attention masks, beam expansion, and
generation cache updates with unit tests before training.

## Dataset and leakage rules

- Split by canonical correct program or problem ID before generating mutations.
- Generate public failing tests and hidden validation tests independently.
- Deduplicate buggy and fixed code across splits by normalized AST.
- Never expose expected outputs for hidden tests.
- A patch is plausible if it passes shown tests; it is correct only if it passes all held-out tests.
- Execute generated Python only in a restricted subprocess with AST checks and resource limits.

## Primary analysis

Fit a mixed-effects logistic model (or bootstrap per bug if sample size is small) for repair success
with condition, log trace length, and their interaction; include bug and model as grouping factors.
The central evidence is a better length slope for latent than for matched text, not merely a higher
aggregate mean.

Report paired bootstrap confidence intervals, McNemar tests for paired conditions, accuracy versus
trace-length quartile, and Pareto curves for correctness versus input tokens/latency.

## Immediate stop conditions

- The target SLM cannot use a concise oracle failure explanation.
- Soft prefixes fail elementary state probes despite overfitting a tiny training set.
- Gains vanish against dynamic-slice text at matched information and budget.
- KV results depend on incorrect position/cache handling or disappear under held-out tests.

