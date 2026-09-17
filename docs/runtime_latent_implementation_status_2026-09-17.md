# Trace2Cache v2 implementation status

Date: 2026-09-17. This is a factual handover for the implementation described in
[the v2 plan](implementation_plan_runtime_latent_2026-09-17.md), not a claim of repair
generalization.

Update: the later [rescue experiments](runtime_latent_rescue_2026-09-17.md) overcome the
400-update code-identification failure with a longer, decaying-LR vector schedule. Read
that report before following this note's old no-go/typed-branch recommendation.

## Completed phases

Phase A is implemented and audited.

- `structured_runtime.py` provides a separate trusted-source collector with typed JSON
  values, type-aware equality (`bool` is distinct from `int`), callback-time snapshots and
  a typed return from the same execution.
- `paired_evidence.py` makes A/B records with identical buggy source, input bundle,
  runtime events, actual value and terminal presentation order. The only allowed view
  differences are the supplied expected values and derived truthful status.
- Full data at `.local/datasets/paired_runtime_v2/` has 1,368 paired records / 2,736
  views: 768 train pairs and 120 pairs in each of five evaluation splits. All 1,368 input
  hashes are unique across splits. The committed manifest/audit report zero invariant
  errors. This data is local by design; Git contains its hash manifest and audit only.

Phase B is implemented and GPU-verified.

- `native_features.py` supplies frozen-receiver `mean0` and two-layer Qwen `context2`
  features, keyed local feature cache, typed padded collator and explicit causal/padding
  mask.
- On the A100 with frozen `Qwen/Qwen2.5-1.5B-Instruct` BF16, `mean0` and `context2` return
  `[N,1536]`, the evidence collator returns `[B,E,1536]`, and the explicit two-layer result
  matches the full Qwen hidden-state path on the unpadded parity probe (`max_abs=0.0`).

Phase C is implemented and audited.

- `receiver_training.py` has marker splicing, target-only causal log-probability alignment,
  masked patch scores, NLL, ranking and auxiliary vector objectives.
- A real frozen-Qwen backward pass gave latent gradient norm `1.188`, zero receiver
  parameter gradients and `4.31 GiB` peak allocated memory. Thus decoder supervision has a
  working gradient path into soft states without fine-tuning Qwen.

## Fixed-panel D0 results

All runs use the same 48-view (`2 pairs x 12 families`) training-only smoke panel, 4,078,848
trainable adapter parameters, eight soft states, seed 401 and the frozen 1.5B receiver.
Nearest-oracle-code accuracy is a representation diagnostic, **not** Repair@1.

| Run | Updates | Final nearest-code | Final cosine | Interpretation |
|---|---:|---:|---:|---|
| `mean0` + vector | 400 | 31.25% | 0.552 | weak fixed-panel code recovery |
| `context2` + vector | 400 | 68.75% | 0.708 | contextual payload features materially help, but miss 90% overfit gate |
| `context2` + decoder from mean anchor | 400 | 4.17% | 0.029 | fresh decoder objective fails to leave generic anchor |
| `context2` vector warmup then decoder | 400 + 400 | 31.25% | 0.098 | decoder continuation degrades code fidelity; no D0 pass |

The correct-oracle check rules out one tempting explanation: fixed oracle repair codes yield
mean positive-vs-opposite teacher-forcing margin `+0.442` and a positive margin for 23/24
behaviors. Therefore candidate ranking is a valid *receiver diagnostic* here; the observed
failure is adapter optimization/representation mismatch around the mean anchor, not an
absence of score signal in the frozen receiver.

Raw JSON artifacts are in `artifacts/paired_runtime_v2/`:

- `overfit_context2_vector_seed401.json`
- `overfit_mean0_vector_seed401.json`
- `overfit_context2_decoder400_seed401.json`
- `overfit_context2_vector_then_decoder_seed401.json`
- `oracle_teacher_forcing_margin.json`

## Decision and next action

The predeclared D0 code-identification threshold was 90%. It was not reached. Do **not** run
the seed-401 2x2 matrix, held-out repair evaluation, MBPP transfer or direct KV writing from
these checkpoints. They would not establish latent runtime control.

The justified next implementation is plan Phase F1/F2, one factor at a time:

1. Add generic typed value fields (signed bits/magnitude/type, sequence item positions and
   explicit string structure) and value/order/equality probes on the saved paired data.
2. Add audited source-location and variable-version/def-use bindings for the restricted
   trusted Python subset; do not call guessed links dynamic dependencies.
3. Re-run the same 48-view overfit gate with the best-understood vector loss before spending
   GPU budget on decoder loss. Advance only if code identity reaches the gate, then perform
   the preregistered matrix and evidence controls.

The evidence for continuing is narrow but real: `context2` improves fixed-panel oracle-code
identification by 37.5 percentage points over `mean0` under otherwise matched training.
It does not yet demonstrate behavior switching, repair success, role awareness, OOD evidence
transfer or a benefit of direct KV writing.

## Reproduction

```bash
.venv/bin/python -m unittest discover -s tests -v

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python \
  scripts/train_paired_trace_encoder.py \
  --objective vector --steps 400 --seed 401 --feature-method context2 \
  --feature-cache .local/cache/paired_runtime_v2/features_overfit_context2 \
  --checkpoint checkpoints/paired_runtime_v2/overfit_context2_vector_seed401.pt \
  --output artifacts/paired_runtime_v2/overfit_context2_vector_seed401.json
```

Before any new neural run, recheck free GPU memory and commit only scoped source/config/data
manifest changes. Model and feature caches remain project-local; no systemwide install or
base-model upload was performed.
