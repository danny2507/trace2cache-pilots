# Trace2Cache: concrete implementation and experiment handover

Date: 2026-09-17. Workspace: `/workspace/hf_cache/test/my-project/codelm_exp2`.
Audience: a smaller coding model taking over implementation and GPU experiments.

This document specifies work to implement. New files, configuration keys and command-line
flags below do not exist yet unless explicitly marked **existing**. Finish each phase's
acceptance checks before executing dependent phases. Read this document and the linked
literature review before changing the project.

## 0. Objective, scope and current evidence

Objective: learn a runtime-evidence encoder whose compact continuous message makes a frozen
small language model choose the repair required by the supplied evidence, including on new
evidence values and eventually new programs.

The immediate question is why the receiver reads oracle behavior codes successfully, while
our runtime encoder does not recover useful repair control. Build a controlled comparison of
content representation and supervision, then improve typed bindings if needed.

Verified results:

| Experiment | Supported result | Limitation |
|---|---|---|
| Expanded oracle codebook, seed 313 | Correct latent solves 23/24, paired opposite latent 0/24; held-out instruction wording 21/24 vs 1/24 | Same known behavior labels and programs; this is receiver capacity |
| Expanded role-aware encoder, seed 317 | OOD nearest code 40.1%, cosine 0.583; true/swap/expected-removed each repair 6/24 | Longer bundles, longer inputs and value changes are entangled |

Equal repair counts do not imply identical generation. True-versus-swap extracted patch
strings differ in 16/24 cases; the repair advantage is absent, not all sensitivity.

Main sources: [focused review](literature_review_runtime_latent_2026-09-17.md). BLIP-2,
ICAE, xRAG, C2C and LatentPress motivate decoder-facing supervision. CLRS and Neural Execution
Engines motivate typed values, positions and references. These are precedents, not proof of
our method. The official ASE 2026 LaMAR listing overlaps latent KV communication for repair;
its full method remains unverified. Avoid first-trace-embedding or first-latent-repair claims.

## 1. Operational rules and exact technology

- Keep all installs, caches, tools and outputs inside this project. No systemwide installation,
  CUDA/driver upgrades, global pip packages, shell-profile edits or changes to another job.
- Existing `.venv` works. Do not rebuild it by default. Use `.venv/bin/python` for execution.
  If a dependency is missing, install it with project-local `uv pip --python .venv/bin/python`.
  Do not run the README's environment-creation command over an existing environment.
- Verified stack: PyTorch `2.4.1+cu118`, PyTorch CUDA runtime `11.8`, Transformers `4.44.2`.
  GPU: NVIDIA A100 PCIe 40 GB; observed driver `525.147.05`. Recheck live free memory before
  each neural job; the earlier free-memory reading is not a reservation.
- Neural training and native contextual encoding run on CUDA. Fail explicitly if CUDA or
  BF16 support is absent. CPU is used only for Python execution, dataset serialization,
  subprocess test validation and small non-neural unit checks.
- Use PyTorch, Python stdlib `dataclasses`, `json`, `hashlib`, `ast`, `difflib`, and the
  existing tokenizer. No new graph library, PEFT dependency, vector DB or LLM API is needed
  for the immediate pilots. JSON configs avoid adding a YAML dependency.
- Receiver for the entire ambiguity study: **`Qwen/Qwen2.5-1.5B-Instruct`**, frozen BF16,
  `eval()`. Existing oracle codes were trained for this receiver. This is not the Coder-3B
  receiver used by earlier MBPP experiments. Do not mix their width/checkpoints/tokenizers.
- Preserve any preexisting working-tree changes. Before training, commit the scoped code,
  config and dataset manifest. Never commit credentials, frozen model weights, `.local/`
  caches or large `.pt` files. `git add` explicit files, never blindly `git add .`.
- Run one neural experiment at a time on this A100. Monitor another user's processes without
  terminating them. Log PID, GPU peaks, elapsed time and optimizer updates.
- Continue reporting progress with counts and completed stages; report negative results.

Existing environment checks:

```bash
.venv/bin/python scripts/check_env.py
nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv,noheader
.venv/bin/python -m unittest discover -s tests -v
git status --short
```

Use CUDA allocator configuration per command:
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
Do not add FlashAttention, vLLM, quantization, `torch.compile`, RL or a new CUDA toolkit to
make this pilot work. They add unrelated moving parts.

## 2. Repository map: reuse these components

| Existing file | Reusable components / caution |
|---|---|
| `src/trace2cache/ambiguous_repair.py` | `get_ambiguous_cases()` returns 24 behaviors grouped into 12 adjacent pairs; `AmbiguousRepairCase`, correct source and diagnosis |
| `src/trace2cache/runtime.py` | `execute_with_trace()` for trusted curated source; `TraceEvent`, `TraceResult`, bounded representations; line events show state **before** that line executes |
| `src/trace2cache/latent.py` | `RoleAwareEventEncoder`, eight queries, hidden width 256, two event Transformer layers, four heads; role embeddings and test/event sinusoidal positions |
| `scripts/run_role_aware_repair.py` | `expected_for()`, native content averaging, original distillation; old sampling couples axes and always writes `test failed` |
| `scripts/run_repair_codebook.py` | `MARKER`, `render()`, `splice()`, `target_ids()`, `NativeRepairCodebook`, `initial_native_codes()` |
| `scripts/run_repair_latent_pool.py` | `greedy_generate()`; existing deterministic custom `inputs_embeds` decoding |
| `src/trace2cache/sandbox.py` | `extract_function()`, `evaluate_patch()` for curated functions; validator rejects many Python constructs; a rejection is an evaluation outcome, not semantic evidence |
| `scripts/evaluate_role_aware_representation.py` | Existing four-split audit structure; sampler is still the old coupled sampler |
| `src/trace2cache/mbpp.py` | Official splits, AST mutation, fuzzing, isolated execution and test tracing for later MBPP phase |
| `scripts/run_mbpp_latent_repair.py` | Existing MBPP pipeline, role-factor slots and negative controls; later integrate new encoder here or through a shared trainer |

Important old checkpoints (local, ignored by Git):

- `checkpoints/repair_latent/expanded_codebook_seed313.pt`
- `checkpoints/repair_latent/expanded_role_aware_ood_seed317.pt`

Oracle checkpoint currently contains a residual `[24, 8, 1536]` tensor and result metadata,
not saved anchor vectors or behavior ordering. Reconstruct anchors using the exact model,
tokenizer and `get_ambiguous_cases()` order, then add the residual. Assert shape and behavior
order compatibility. The residual contains **294,912** parameters.

Do not overwrite these artifacts or historical scripts to silently change old results.
Create a new v2 pipeline and use old code as reference. Existing checkpoint loading must
still work after changes to shared encoder classes.

## 3. Planned files and responsibilities

Create these modules in dependency order:

```text
src/trace2cache/
  paired_evidence.py       # schema, matched-pair sampler, splits, validation, controls
  structured_runtime.py    # trusted v2 collector with typed actual outputs/snapshots
  native_features.py       # mean0 and frozen-Qwen context2 feature extractors, cache keys
  receiver_training.py     # prompt splicing, patch token scores, NLL and paired ranking
  paired_metrics.py        # pair success, semantic switch, clustered confidence intervals
  typed_values.py          # later: generic typed scalar/sequence encoding
  binding_encoder.py       # later: typed fields and relations before latent resampling

scripts/
  prepare_paired_evidence.py
  audit_paired_evidence.py
  train_paired_trace_encoder.py
  evaluate_paired_trace_encoder.py
  run_paired_matrix.py     # sequential orchestration, resume, no hidden training changes
  summarize_paired_matrix.py

configs/paired_runtime_v2/
  smoke.json
  overfit.json
  mean_vector.json
  mean_decoder.json
  context2_vector.json
  context2_decoder.json

tests/
  test_paired_evidence.py
  test_native_features.py
  test_receiver_training.py
  test_paired_metrics.py
  test_typed_values.py     # only when the typed branch is implemented
```

These new filenames are implementation targets, not existing commands.
Shared training functions belong in `src/`, not imports from a sibling script where avoidable.

## 4. Phase A: build and audit genuinely matched paired data

### 4.1 Data objects

Use immutable dataclasses with a `schema_version` field. Keep model evidence and supervision
in separate objects so labels/reference patches cannot accidentally reach the encoder.

```python
@dataclass(frozen=True)
class EvidenceEvent:
    event_id: int
    test_id: int
    step: int
    role_id: int
    content: str                  # neutral ordered payload, no role words or case labels
    source_line: int | None

@dataclass(frozen=True)
class EvidenceView:
    view_uid: str
    events: tuple[EvidenceEvent, ...]

@dataclass(frozen=True)
class PairedEvidenceRecord:
    schema_version: int
    pair_uid: str
    family_id: str                # bookkeeping only
    split: str
    buggy_source: str
    function_name: str
    public_test: dict
    evidence_a: EvidenceView
    evidence_b: EvidenceView
    label_a: int                  # supervision only, never input to encoder/prompt
    label_b: int
    target_source_a: str          # teacher-forcing target only
    target_source_b: str
    tests_metadata: tuple[dict, ...]
```

`family_id`, case ID, label, diagnosis, correct source, intended operation names, checkpoint
keys and filenames are forbidden encoder fields. Expected values themselves are allowed:
they are the supplied behavioral specification. The shared decoder context is buggy source,
the public failure, the fixed marker and the same instructions for both sides.

Roles for immediate v2 remain the existing ten IDs:
`TEST_START, INPUT, CALL, LINE, BRANCH, DEFINITION, RETURN, ACTUAL, EXPECTED, STATUS`.
For content use neutral strings such as `argument=0; item=2; value=-3` or just an output's
canonical typed value. Do not write `expected output` or `actual output` in payload text;
those lexical cues would survive a metadata-role ablation.

Also counterbalance the presentation order of ACTUAL/EXPECTED records, using a saved random
choice per test shared by A/B. Give these terminal records the same semantic time; do not
encode their role through a special timestamp or ID. Keep execution events chronological.
Otherwise the model can recover roles from fixed event positions after role IDs are removed.
Natural clues such as an actual value also appearing in RETURN can remain; report metadata
role ablation as a measured contribution, not proof that all role information disappeared.

### 4.2 Matched-pair generation algorithm

For each family pair `(A, B)`:

1. Assert same buggy source, function name and public test.
2. Sample an input bundle once, with a local `random.Random(data_seed)`.
3. Compute expected A/B outputs using `expected_for()` or trusted reference functions.
   Cross-check these two implementations on generated examples before relying on them.
4. Reject a bundle if there is no test distinguishing A from B. Prefer all tests to
   distinguish behavior for the initial mechanism pilot. Bound resampling attempts and
   report exclusions, rather than looping forever.
5. Execute buggy source **once per input**. Build shared input/trace/actual records.
6. Create A and B views by attaching their respective expected values and correct statuses.
   PASS iff the actual structured value equals expected; exception => FAIL unless an
   explicit exception specification exists. Do not label every execution FAIL.
7. Differences between views may be expected values and derived statuses only. Input,
   buggy runtime events, actual outputs, source associations and test ordering must match.
8. Hash normalized buggy source + ordered input arguments to identify bundle overlap.
   Assign both sides to the same split. Reproduce from saved records, not fresh sampling
   during evaluation.

Require `expected_a != expected_b` with a type-aware comparator. Python `True == 1` must not
collapse distinct value types. Use JSON-compatible tagged values (`int`, `bool`, `str`,
`list`, `none`) and deterministic serialization; avoid `eval(repr)`.

Do not assume 'has duplicate' implies unique-first/unique-last differ, or any sampled
rotation inputs distinguish left/right. Check the actual expected outputs.

The old choose sampler has only 60 possible ordered triples when using three distinct
lengths from 1..5 and repeated `x` strings. Add neutral randomized character contents to
avoid exhausting short in-range split support, while selecting shortest/longest by length.
Never add different character distributions to A and B.

### 4.3 Separate distribution axes

Refactor `sample_args()` into an explicit `InputDistribution` with independent
`bundle_test_count`, `input_item_count` and `value_domain` parameters. Do not reuse the old
`heldout_values=True` flag: it changes item count as well as values, and for some families
does not change value range at all.

Primary splits:

| Split | Tests per bundle | Input item-count distribution | Value domain |
|---|---:|---|---|
| train / dev / test-short | 2–4 | fixed per-family short distribution | train domain |
| test-bundle-long | 5–8 | same as train | train domain |
| test-value-new | 2–4 | same as train | predeclared new-value domain |
| test-bundle-long-value-new | 5–8 | same as train | new-value domain |
| later test-input-long | 2–4 | independently longer inputs | train domain |

Use a per-family manifest for admissible domains. For numbers, keep negative/zero/positive
and even/odd representation where the task requires it; record unseen magnitudes instead
of changing signs only. For strings use declared length ranges and character distributions.
Call the split 'new values' when appropriate rather than falsely claiming every field is
disjoint. If an invariance changes behavior (e.g. arbitrary scaling changes parity), do
not use it as a positive augmentation.

Concrete initial distributions (uniform choice of lengths and values unless noted):

| Family | Short input items | Later input-long items | Train values | New values |
|---|---|---|---|---|
| locate | 3–5 | 9–12 | integers 0..5; target 0..5 | integers in FAR; target 8..20 |
| aggregate | 2–4 | 9–12 | integers -4..4 | integers in FAR |
| extreme | 3–5 | 9–12 | integers -5..5 | integers in FAR |
| measure | 3–6 | 9–12 | integers -4..4 | integers in FAR |
| arrange | 3–5 | 9–12 | integers -5..5 | integers in FAR |
| choose | 3 | 5 | distinct string lengths 1..5; characters a/b/c | distinct lengths 8..12; same characters |
| truth | 3–5 | 9–12 | integers -4..4 | integers in FAR |
| unique | 4–6 | 9–12 | integers 0..3 | integers 20..23 |
| median | 4 or 6 | 10 or 12 | distinct integers -8..8 | distinct integers in FAR |
| rotate | 3–5 | 9–12 | integers -4..4 | integers in FAR |
| transform | 3–5 | 9–12 | integers -4..4 | integers in FAR |
| filter | 3–5 | 9–12 | integers -4..4 | integers in FAR |

`FAR = {-20, ..., -8} union {8, ..., 20}`. Locate must force at least two occurrences of
the target at distinct sampled positions. Median samples without replacement. Other
families resample until A/B reference outputs differ; truth therefore includes mixed
positive/nonpositive values. Unique-first/last must have a genuine ordering difference.
Save the full domain table and rejected-sample counts in the manifest. These distributions
test input/evidence transfer within known programs, not arbitrary algorithmic extrapolation.

Initial counts, **pairs** rather than individual views:

- Train: 64 pairs per family = 768 pairs = 1,536 views.
- Development: 10 pairs per family = 120 pairs = 240 views.
- Each primary test split: 10 pairs per family = 120 pairs = 240 views.
- Smoke: 2 pairs per family; overfit set: 2 fixed pairs per family (48 views).

Allow smaller feasible counts if explicit uniqueness checks exhaust a finite family;
write actual counts and reasons. Never silently duplicate dev/test input bundles from train.
Data seed `401`; model seeds `401, 409, 419`. Training seed must not regenerate data.

### 4.4 Trace collection and budgets

Keep `execute_with_trace()` unchanged for historical runs. V2 must retain before-line
semantics. A locals change observed at the next line belongs to the previously executed
statement; it is not evidence that the current line defined the value. Label inferred
definitions with their attribution method, and use UNKNOWN when control flow makes an
inference uncertain. Merely starting with `if` or `for` is not an observed Boolean outcome.

The old collector returns only bounded repr strings, so it cannot directly provide exact
typed output comparison. Implement `structured_runtime.py: execute_structured_trace()` for
trusted curated sources, returning legacy-compatible event summaries plus tagged actual
return value, error, and typed snapshots where supported. Reuse/factor tracing logic without
changing old API results. Serialize snapshots inside the tracing callback, before mutable
lists can change again; do not retain references and serialize them after execution.
Bound recursion/items/characters and mark truncation UNKNOWN explicitly. Record before-line
timing. This collector must trace and capture actual output in the **same single call**.
Do not reexecute solely to obtain a typed output or parse truncated repr to recover it.
Use trusted sources only; generated patches continue through the subprocess evaluators.
Test typed returns and in-place list changes against the original trusted function.

Keep all events before budgeting; select boundary/state-change events with a deterministic
policy, recording omitted counts. Preserve input, actual, expected and status records.
Use the same selected events for both representation methods and every text baseline.
Set v2 `max_events=512`, `max_tests=8`, `max_content_tokens=128` initially. Record every
truncation; do not silently drop expected evidence. Legacy checkpoints still use 192.
Increase caps for all compared conditions together if an audited distinguishing field is lost.

### 4.5 Outputs and acceptance

Store records under `.local/datasets/paired_runtime_v2/` as JSONL. Commit only the manifest,
generation config, hashes and aggregate audit at
`artifacts/paired_runtime_v2/data_audit.json`.

Tests must verify pair identity, correct A/B expectations, truthful PASS/FAIL, no label or
target leakage, paired split placement, train/dev/test input-hash separation, determinism,
distinguishing tests and explicit truncation. Audit all generated records.

**Gate A:** no unresolved invariant failures. If a family cannot produce valid distinguishable
samples, report it and exclude it consistently from every condition. Do not proceed with
corrupted pair data to meet a requested sample count.

## 5. Phase B: two content encoders, otherwise identical architecture

Implement `FrozenNativeFeatureExtractor` separately from the trainable event encoder.
Do not register the full receiver as a trainable encoder submodule: calling `.train()` on
the adapter must not change the receiver's mode.

API:

```python
extract_contents(contents: list[str], method: str) -> Tensor  # [N, 1536], detached
collate_views(views, cache, device) -> dict                   # [B,E,*], masks
encode_evidence(batch) -> Tensor                            # [B,8,1536], differentiable
```

### B0: `mean0`

Exactly masked mean of the receiver's frozen input token embeddings per neutral payload.
Preserve token count and truncation metadata. It is the v2 baseline, not an exact reproduction
of seed 317: dataset and lexical role cleanup have changed.

### B1: `context2`

Tokenize the same payload; preserve token order. Run the receiver's frozen input embeddings
through its first **two Qwen2 Transformer layers**, then use the last valid token's hidden
state as the event content feature. No output LM head or final model norm in the feature
extractor. This contextualizes within an event; the trainable event Transformer handles
relations across events. Last-valid pooling is the predefined choice for this experiment.

Implementation for the pinned Transformers version:

- Reuse `model.model.layers[:2]` as frozen components on CUDA; no second full model copy.
- Use `torch.no_grad()` only for this frozen feature computation/cache.
- Supply causal attention plus padding masks, `position_ids`, `cache_position`,
  `past_key_value=None`, `use_cache=False`, `output_attentions=False`.
- In Transformers 4.44.2, `Qwen2DecoderLayer.forward` accepts `position_ids` and
  `cache_position`; do not paste newer examples requiring `position_embeddings`.
- Make the mask `[B,1,L,L]`, with future keys and padded keys blocked. Never pool padding.
  Detect non-finite features and entirely empty tokenizations.
- Start with receiver `attn_implementation='eager'` for simple parity verification; use the
  same implementation across compared runs. SDPA is a measured later optimization.
- Compare the partial path with `model.model(..., output_hidden_states=True,
  use_cache=False).hidden_states[2]` on two unpadded short strings and padded batches.
  The full forward is just a verification check; feature preparation uses the partial path.
  Use BF16-appropriate relative/absolute tolerances and document them.

Batch content extraction by token lengths, initial batch size 16, token cap 128. Reduce
microbatch on OOM without changing payload selection or token cap. Extract unique payloads
once; cache detached BF16 features under `.local/cache/paired_runtime_v2/features/`.
Cache keys include model snapshot/revision, tokenizer hash, neutral payload, method,
pooling, layer count, maximum token length and schema version. An ID from a sorted vocabulary
alone is insufficient. Evaluation may use frozen feature caching, but its labels/targets
must not affect training or cache construction.

### Shared adapter

Reuse `RoleAwareEventEncoder` with:

```text
model_width = 1536 (assert against receiver config)
hidden_width = 256
num_layers = 2
num_heads = 4
latent_states = 8
roles = 10
max_events = 512
max_tests = 8
dropout = 0
output_anchor = mean of the fixed 24 oracle codes, detached
output = anchor + learned residual
slot_roles = unrestricted for the initial 2x2
```

A 'slot' here means one of the eight learned query vectors and resulting continuous message
states. It has no guaranteed semantic meaning unless separately supervised or audited.

Do not assume the existing `slot_role_mask` provides strict role separation: it masks
resampling after an all-event Transformer has already mixed roles. Restricting final
attention cannot erase information introduced earlier. A later factorized-role experiment
must restrict mixing before resampling or be described only as a pooling bias.

**Gate B:** shape, padding invariance, token-order sensitivity of `context2`, feature-cache
round-trip, frozen-component checks, and the partial/full-layer parity check pass. Input
embedding means should be identical for an exact permutation of the same token IDs;
contextual features need not be.

## 6. Phase C: decoder scores, correct gradients, and loss choices

Implement scores centrally in `receiver_training.py`:

```python
patch_token_logprobs(model, prompt_embeds, target_ids) -> Tensor  # [target_len]
patch_score(logprobs, token_mask=None) -> Tensor                # masked mean
patch_nll(logprobs) -> Tensor                                  # negative mean
paired_patch_loss(view, positive_patch, negative_patch, ...) -> dict
```

Use the existing codebook prompt/rendering and `MARKER` splicing. Do not append diagnosis
text or behavior names to the prompt. For opposite candidates the visible context is the
same; only latent evidence changes. Targets are fenced complete functions plus EOS,
using existing `target_ids()`.

Causal alignment, copied from the verified existing code:

```python
teacher = embedding(target_ids[:, :-1])
inputs = torch.cat([prompt_embeds, teacher], dim=1)
logits = model(inputs_embeds=inputs, attention_mask=mask, use_cache=False).logits
start = prompt_embeds.shape[1] - 1
predicted = logits[:, start:start + target_ids.shape[1]]
```

Calculate target-only token log probabilities with FP32 logits or stable log-softmax.
Keep prompt positions out of the loss. Wrong shifts can make a model seem to learn while
only rewarding patch copying. Verify with a tiny hand-checkable fixture.

### C0: `vector`

Original distillation objective:
`L_vector = MSE(Z, oracle_code) + 0.1 * (1 - cosine(flat(Z), flat(oracle_code)))`.
Oracle codes are supervision only; input evidence has no label lookup at inference.

### C1: `decoder`

For each orientation, let `P+` be the patch required by that evidence, and `P-` the opposite
pair's patch. Both use the same visible buggy code/public test.

```text
s+ = mean token log-probability of P+ conditioned on Z
s- = mean token log-probability of P- conditioned on Z
L_rank = softplus(0.1 - s+ + s-)
L_decoder = NLL(P+ | Z) + 1.0 * L_rank + 0.1 * L_vector
```

Apply to A and B symmetrically. These weights are initial preregistered settings, not tuned
results. Log each component, `s+ - s-`, gradient norms, and evidence-controlled score changes.

Initially rank whole sequences with length-normalized scores. Also report a distinguishing
token score using `difflib.SequenceMatcher(..., autojunk=False)` over target token IDs.
For each side mark tokens outside common matching blocks; masks and denominators may have
different lengths. Shared boilerplate is excluded only from this diagnostic initially.
If whole-sequence ranking is diluted, a separately named run may use differing-token
ranking; predeclare the change and preserve the original run.

The receiver has `requires_grad=False`, but its forward during loss computation **must not**
be inside `no_grad()` or `inference_mode()`. Gradients must flow through frozen operations
to `Z` and the encoder. Do not detach `Z` or copy it through a tensor constructor.
Casting it to BF16 for `inputs_embeds` keeps the gradient path.

Before a long run verify:

- A real decoder loss produces finite, nonzero gradients on trainable encoder parameters.
  The zero-initialized last output layer can initially block earlier-layer gradients;
  require nonzero projection gradients first and upstream gradients after an update.
- Frozen model parameters have no gradients and unchanged checksums on sampled tensors.
- Distinct fixed oracle latents change candidate scores in a toy check of the scoring path.
  The zero-output-initialized encoder may emit identical latents before training; do not
  require untrained evidence views to have different scores as a plumbing test.
- EOS and target-only masking are aligned correctly.
- Checkpoint save/load preserves latent outputs and greedy generation.

To limit graph memory, compute orientation A's two candidate forwards and backward, then
recompute encoder output for orientation B and backward. Normalize each orientation by
`2 * gradient_accumulation`; step once per accumulation group. Do not retain a graph across
optimizer steps. A/B symmetry must remain even when microbatching.

## 7. Phase D: overfit sanity check and controlled 2x2 matrix

### D0: overfit gate

Use the 48-view fixed overfit set across all 12 families. It is explicitly training-set
evaluation, never a generalization result. Train `context2/vector` first for up to 400
optimizer updates; evaluate oracle-code identification. Then try decoder supervision
for up to 400 more updates if useful repair control is absent.

Use the same instruction as codebook training first; evaluate held-out wording separately.
If train-set code identification cannot reach 90%, inspect data correctness, oracle
loading, gradients, scaling and optimization before any OOD claims. If it identifies codes
well but continuous latents repair poorly, evaluate nearest-code projection as a diagnostic.
If no train-set repair benefit appears, do not spend all three seeds on the full matrix.

### D1: main four runs

| Run name | Content features | Post-warmup objective |
|---|---|---|
| `mean_vector` | `mean0` | vector distillation |
| `mean_decoder` | `mean0` | decoder NLL + paired ranking + auxiliary vector |
| `context2_vector` | `context2` | vector distillation |
| `context2_decoder` | `context2` | decoder NLL + paired ranking + auxiliary vector |

For each representation, first train a common 400-update vector warmup and save it. Fork
that same warmup into the two objectives. Reset optimizer state in **both** branches, then
train each for 800 additional updates. Paired batch exposure per update must match across
branches; compute/FLOPs and wall time will differ and must be reported.

Initial optimizer: AdamW, weight decay 0.01, gradient clipping 1.0. Warmup-stage LR 1e-3;
post-warmup LR 3e-4 for both branches. Use a linear schedule with 5% LR warmup, independent
of the semantic vector warmup stage. Save config and actual schedule in checkpoint metadata.

Use effective 8 pairs/update. Vector branch may process the 16 views in one CUDA batch;
decoder branch uses one pair/microbatch and gradient accumulation 8. Each optimizer update
consumes the same sampled pair IDs for both loss branches (save/replay a seeded schedule).
The encoder trains FP32; receiver is BF16. BF16 normally does not require GradScaler.

Start with model seed 401 only. Assess development evidence every 100 optimizer updates;
select checkpoint by **development pair success**, then score margin as a tie-breaker.
Do not select by test-split results or training cosine. Generation on full dev sets is
expensive: use a predefined balanced 24-view dev panel every 200 updates and full dev
generation at final/checkpoint-selection points. Selection policy must be consistent.

After fixing the pipeline/config, replicate all compared main runs with seeds 409 and 419
if resources permit. At minimum replicate the proposed winner and its matched baseline;
label other one-seed runs as exploratory. Do not claim a robust interaction from one seed.

### D2: limits and persistence

Before any job use `torch.cuda.mem_get_info()` and the observed smoke peak + 2 GiB headroom.
Do not assume the old 3B MBPP peak applies to the new candidate-scoring loop. Set a configurable
resource ceiling (initially 24 GiB allocated) and record reserved as well as allocated memory.
On OOM reduce feature/decoder microbatch or recompute candidate graphs; keep data, token cap,
effective pair batch and objectives fixed. If those cannot fit, report the specific stage.

Checkpoint every 100 updates: trainable encoder state, anchors, config, oracle code hash,
ordered behavior IDs, data manifest hash, model/tokenizer revision, feature cache metadata,
optimizer/scheduler states, next update and Python/Torch/CUDA RNG states. Resume without
regenerating data or retraining frozen features. Save with temporary path then atomic rename.

Never truncate a patch to fit silently. Initial maximum prompt-plus-target length 1024
tokens and greedy output cap 192; log cap hits. Any cap change requires the same setting
for all conditions of that comparison.

## 8. Phase E: evaluation conditions and semantic metrics

Use saved test records, held-out instruction wording, greedy decoding and fixed budgets.
Evaluate both sides of every pair for every saved bundle, not one arbitrary example per label.

Required conditions:

1. `no_evidence`: same visible prompt, evidence unavailable.
2. `oracle_latent`: fixed known-label capacity reference; label access is explicitly oracle.
3. `true_latent`: encoder's continuous output for the correct view.
4. `paired_swap`: other view from the **same matched pair**, keeping visible code fixed.
5. `expected_removed`: expected payload replaced by one neutral unknown payload; retain
   event count and role so removal tests value content. Record status as a possible proxy.
6. `expected_and_status_removed`: remove expected values and their derived status cues.
7. `role_swap_expected_actual`: swap role IDs while preserving payloads and all other fields.
   These are deliberately corrupted views, not alternate valid specifications.
8. `no_roles`: replace roles with one shared ID; payloads must not contain lexical role names.
9. `io_only`: inputs + expected outputs, with shared TEST_START/required metadata; both text
   and latent forms. Train a separate matched-budget IO-only model before claiming runtime
   events add value; masking a full-trace model only measures inference ablation.
10. `structured_text`: same selected records rendered with explicit roles for the decoder.
    Record actual tokenizer length. Add full text and deterministic compact text baselines
    when estimating the effect of evidence length.

For expected removal inspect whether all-fail/all-pass status leaks a behavior. Do not
describe a residual effect as expected-value reasoning without the status-removed control.

Optional diagnostic: `nearest_oracle_latent` snaps encoder predictions to the closest
oracle code by flattened cosine and then decodes. This isolates continuous-code fidelity
from code classification; it is restricted to the known behavior vocabulary.

Per generated row save: run/checkpoint/data hashes, pair UID, behavior side, split, condition,
event/token counts, candidate margins, response, extracted source, cap-hit flag, validation
category, A-tests/B-tests results, and encoder/prefill/decode/test timings.

Metrics:

- `Repair@1`: passes the intended public and hidden tests.
- `PairSuccess`: BOTH sides pass their respective tests for the same input bundle.
- `TrueMinusSwap`: paired repair difference, with raw denominators.
- `OppositeSelection`: swapped evidence produces a patch satisfying the opposite behavior
  tests, preferably not the intended side. Build the opposite RepairCase with the same
  public context and the opposite hidden tests; validate in the existing subprocess runner.
- Correct-sign score margins for A/B, whole sequence and distinguishing tokens.
- Nearest oracle-code accuracy overall/per family; cosine is an auxiliary measure.
- Text/AST equality rate, explicitly syntactic. Semantic switch requires executed tests.
- Invalid syntax, validator rejection, timeout, public-fail, hidden-fail and token-cap rates.
- Peak GPU memory, trainable parameters, feature/write and receiver/read latency separately.

Use a cluster bootstrap over family/program groups and report per-family results. Multiple
bundles from 12 programs are not 240 independent programs. At later MBPP stages bootstrap
task IDs; preserve mutation/test bundles within each task. Save bootstrap seed and 2,000
resamples. A result on twelve toy families is a mechanism finding with limited external scope.

Benchmark limits: these toy pairs use different intended specifications for the same visible
code. They are a controlled communication experiment. Their behavior labels are not real
bug classes, and both sides need not be alternate fixes for one fixed complete specification.

**Mechanism go/no-go, predeclared working thresholds (not achieved results):**

- On new short/in-range evidence, true repair >=75%, pair success >=60%, true-minus-swap
  >=30 percentage points, and intended opposite behavior selected in >=60% swapped views.
- Replicated improvement over the matched v2 baseline, not merely over the historical 6/24.
- Report new-values and long-bundle degradation even if the in-range gate passes.
- If IO-only performs comparably, narrow the conclusion to behavioral-specification
  communication and investigate runtime utility before claiming a trace contribution.

These are pilot decisions, not significance thresholds or a guarantee of publication.

## 9. Phase F: typed binding branch if the controlled matrix fails or plateaus

Implement this only after the matrix/audits identify a need. Change one representation at
a time using the winning or best-understood loss and saved data.

### F1: generic exact values and item positions

`typed_values.py` should encode:

- Tags for none, bool, int, float, string, list/tuple and unknown.
- Signed integers as sign plus declared-width bit features (initial signed 64-bit support),
  and a bounded scalar magnitude feature. Preserve raw integer JSON exactly; explicitly
  reject/mark overflow rather than silently rounding via float32.
- List items as separate records with item indices and parent references, preserving order;
  use sinusoidal or normalized positions with a declared length policy.
- Strings as ordered token features with an explicit length; do not reduce the string to
  length only, since future repair can depend on character content.
- Booleans separately from integers. Avoid untrusted `eval` and avoid `ast.literal_eval`
  on truncated repr strings; collector schema must mark truncation/unknown.

Bit features are a tested inductive bias candidate, not a guarantee of arithmetic
extrapolation. Add parity/sign/equality features as a separately named generic ablation
if needed. Do not add family-specific sums/products/minima, patch IDs or diagnosis flags.

Train a small value projection into hidden width 256 and concatenate/project it with
contextual content, role, test and position features. Keep eight output states and the
receiver frozen. Add probes for typed-value reconstruction, equality, order and last
definition using training/runtime-generated labels; evaluate probes on new values.

### F2: explicit source and dependency binding

Use `ast` to map trusted source spans to stable local source-node IDs. Track variable version
on observed writes; connect each read to an attributable last definition. Represent relations
as `(src_event_id, dst_event_id, relation_type)` with types NEXT, SAME_TEST, SAME_OBJECT,
DEF_USE and CONTROL when faithfully collected. Do not call a guessed edge exact dynamic
dependence, and do not attach changes to the wrong before-line event.

Initially restrict to a documented Python subset (assignments, arithmetic, if and for),
with unsupported constructs marked UNKNOWN. Instrument branch tests only if needed, using
an AST helper that evaluates the original expression once and preserves short-circuit,
truthiness and side effects. Test instrumented-vs-original outputs before trusting edges.

A concrete lightweight relational implementation needs no PyG: add relation embeddings
to attention biases/masks or use indexed messages with `index_add_`. Use two propagation
layers followed by the existing resampler. Report actual parameter/time increase.

Factor source location, object/version IDs and relations in ablations; canonicalize variable
renaming consistently without leaking the canonical target. Only introduce role-specific
slots after addressing earlier cross-role mixing; describe unrestricted slots as unrestricted.

**Gate F:** typed-field/edge correctness tests, new-value probe gains and improved repair
pair selection. If only the classifier improves, continue diagnosing receiver robustness.

## 10. Phase G: task-disjoint Python repair generalization

Once the mechanism gate passes, transfer the same evidence schema and loss to mutated MBPP.
Use the existing `mbpp.py` machinery and local `.local/datasets/mbpp/mbpp.jsonl`.

- Official train task IDs: 601–974; validation: 511–600; test: 11–510. Verify the loader,
  don't assume these ranges alone prevent overlap. All mutants, fuzz inputs and augmentations
  of a task remain in that task's split.
- Freeze the same 1.5B receiver initially to isolate transfer. Use existing 3B-Coder runs as
  historical context, not matching checkpoints. A 3B receiver needs a new width-specific
  encoder/output projection and training; report it as a separate replication.
- Remove oracle behavior-code auxiliary supervision: unseen MBPP tasks have no member of
  the 24-code dictionary. Initialize transferable encoder layers/anchor from v2 and train
  with patch NLL plus applicable verified negatives. Do not query label codes at inference.
- Generate train mutants using existing AST mutation, derive visible failing evidence and
  expected values from canonical **training** programs; no GPT API required.
- For patch negatives, retain only candidate mutants/patches that demonstrably contradict
  the supplied evidence. Do not force ranking against a semantically equivalent alternate
  correct patch. If no validated negative exists, use NLL plus evidence interventions.
- Use passing and failing executions when available. Canonical full execution traces are
  an oracle signal unavailable at normal inference; expected test outputs are allowed,
  but do not pass canonical trace states/source into the encoder silently.
- Keep final held-out assertions out of prompts/latents/training. Use available tests or
  fuzz calls as visible evidence and disjoint saved calls/tests for patch evaluation.
  Log how evidence/evaluation test sets are constructed and any overlap limitations.
- Audit normalized AST/source similarity, task descriptions, mutation parents and near
  duplicates. Prior use of MBPP splits must be documented; do not claim untouched test data
  simply because an ID range was reserved in an old note.
- Start with train <=1,536 validated bundles, dev 48 distinct tasks if feasible, and a
  fixed larger final test after validation selection. Log actual retained task counts.
  Improve sample diversity rather than creating unlimited near-duplicates.
- Adapt extraction/validation to the existing MBPP worker; curated `sandbox.py` rejects
  attributes, imports and lambdas and cannot evaluate all legitimate MBPP fixes. Preserve
  worker time/resource limits and record unsupported examples consistently.
- Compare no evidence, selected structured text, compact text, true latent, different-task
  shuffle, same-task wrong-evidence swap when available, role/value removal, and IO-only.

HumanEval-based repair or a second local Python benchmark follows MBPP validation. If new
dataset/model downloads are needed, store inside project caches, record source/license and
revision, and consult primary documentation for current APIs. The immediate v2 pipeline
does not require any download or API generation.

**Generalization claim requires:** unseen task/source evidence, hidden-test repair gains
over matched baselines, useful true-versus-corrupt evidence differences, seed replication,
and honest overlap/uncertainty reporting. Prompt wording transfer alone is insufficient.

## 11. Phase H: direct KV writing, only after useful runtime control

Initial experiments use `[B,8,d]` vectors inserted via `inputs_embeds`. Normal receiver
prefill creates a KV cache from these vectors; this is not an external per-layer KV writer.

Only after the encoder demonstrates useful evidence control, add an interface ablation:

- Soft-state prefill baseline: generate caches through the frozen receiver normally.
- Direct writer: map encoder latents to each selected layer's key/value tensors,
  `[B, num_key_value_heads, K, head_dim]`. Qwen uses grouped-query attention; derive these
  dimensions from config, not `num_attention_heads` alone.
- Define cache insertion order, attention lengths, cache_position, and RoPE treatment once.
  Keys must match the chosen positional convention; values are not rotary-encoded.
- Initially use prefix-before-code KV so code prefill can attend to evidence. If comparing
  against the existing mid-prompt marker, separately test prefix soft-state placement:
  otherwise location and interface are confounded.
- Train only the KV writer with the same decoder-facing supervision; initialize/distill
  from successful soft-state caches if helpful. Keep old interface checkpoints distinct.
- Verify cached-vs-uncached logits, sequence lengths, absence of NaNs, gradients, intervention
  effects and hidden-test repairs. Use pinned Transformers cache API; don't upgrade it.

Report prefix vectors and layerwise cache bytes separately: eight BF16 input vectors use
`8*d*2` bytes, whereas stored per-layer KV uses
`2*layers*kv_heads*8*head_dim*2` bytes. Measure actual compute/memory/write cost; direct KV
writing is not automatically smaller or semantically cleaner.

## 12. Config, CLI and output contract

Implement a JSON config loader with explicit validation. Example core keys:

```json
{
  "schema_version": 2,
  "run_name": "context2_decoder",
  "receiver": "Qwen/Qwen2.5-1.5B-Instruct",
  "data_manifest": ".local/datasets/paired_runtime_v2/manifest.json",
  "oracle_checkpoint": "checkpoints/repair_latent/expanded_codebook_seed313.pt",
  "feature_method": "context2",
  "feature_pooling": "last_valid",
  "feature_layers": 2,
  "objective": "decoder",
  "hidden_width": 256,
  "latent_states": 8,
  "max_events": 512,
  "max_tests": 8,
  "max_content_tokens": 128,
  "max_sequence_tokens": 1024,
  "max_new_tokens": 192,
  "warmup_updates": 400,
  "postwarmup_updates": 800,
  "pairs_per_update": 8,
  "decoder_microbatch_pairs": 1,
  "gradient_accumulation": 8,
  "warmup_lr": 0.001,
  "postwarmup_lr": 0.0003,
  "weight_decay": 0.01,
  "cosine_weight": 0.1,
  "rank_weight": 1.0,
  "rank_margin": 0.1,
  "vector_aux_weight": 0.1,
  "lr_warmup_fraction": 0.05,
  "gradient_clip": 1.0,
  "checkpoint_every": 100,
  "seed": 401,
  "data_seed": 401
}
```

Validate effective batching and objective-specific flags. Record resolved config, not
only the input config. `--seed` overrides model seed only. Required script interfaces:

```text
prepare_paired_evidence.py --config CONFIG --output-dir DIR
audit_paired_evidence.py --manifest PATH --output JSON
train_paired_trace_encoder.py --config CONFIG --seed INT --output-dir DIR
                             [--checkpoint-dir DIR] [--init PATH] [--resume PATH]
                             [--stage warmup|postwarmup]
evaluate_paired_trace_encoder.py --checkpoint PATH --manifest PATH
                                --splits ... --conditions ... --output-dir DIR
run_paired_matrix.py --config-dir DIR --seeds ... --output-root DIR [--resume]
summarize_paired_matrix.py --artifact-root DIR --output JSON --report MD
```

Trainer `--stage` must allow shared representation warmup forks. Orchestrator must run
preparation/audit/smoke explicitly, reuse a common warmup per representation and seed,
launch sequential jobs, capture failures and resume completed stages by config hashes.
It must not silently run all later optional phases or modify configurations to get a win.

Planned commands, executable **after implementing these interfaces**:

```bash
.venv/bin/python scripts/prepare_paired_evidence.py \
  --config configs/paired_runtime_v2/smoke.json \
  --output-dir .local/datasets/paired_runtime_v2_smoke

.venv/bin/python scripts/audit_paired_evidence.py \
  --manifest .local/datasets/paired_runtime_v2_smoke/manifest.json \
  --output artifacts/paired_runtime_v2/smoke_data_audit.json

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python \
  scripts/train_paired_trace_encoder.py --config configs/paired_runtime_v2/smoke.json \
  --seed 401 --output-dir artifacts/paired_runtime_v2/smoke_seed401

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python \
  scripts/run_paired_matrix.py --config-dir configs/paired_runtime_v2 \
  --seeds 401 --output-root artifacts/paired_runtime_v2/matrix --resume
```

Use `.local/logs/paired_runtime_v2/` for logs, `.local/cache/` for frozen feature caches,
`checkpoints/paired_runtime_v2/` for resumable `.pt` files, and
`artifacts/paired_runtime_v2/<run>/seed<seed>/` for JSON/JSONL summaries/rows and resolved config.
Do not accidentally save large optimizer states under committed `artifacts/`.
If `--checkpoint-dir` is omitted, derive
`checkpoints/paired_runtime_v2/<run_name>/seed<seed>/`; the artifact output directory is
never the default optimizer-checkpoint directory.

Smoke config: 24 pairs total, feature batch <=8, vector warmup 2 updates and postwarmup
2 updates, one generation per family, same schema as main. Its score has no research value.

## 13. Checklist and implementation order for the next coding model

1. Read this plan, the review, the existing helper code and any applicable `AGENTS.md`.
   Recheck environment/GPU/git state. Verify the two old checkpoints exist.
2. Commit the scoped existing result/review correction if still uncommitted. Preserve
   unrelated user changes. Record the commit hash before new experimental work.
3. Implement Phase A dataclasses, generator, manifest, audit and data correctness tests.
4. Implement Phase B extractor/cache and shared v2 adapter collator. Verify contextual
   parity and frozen weights on GPU.
5. Implement Phase C token scoring, paired loss and gradient/alignment checks.
6. Add smoke config/CLI, checkpoint/resume metadata and a smoke report. Fix plumbing errors.
7. Commit tested implementation and config; run/record the overfit gate.
8. If gate passes, prepare audited main data and run the seed-401 2x2 with shared warmups.
9. Evaluate saved evidence controls/four primary transfer splits; publish a local report
   with raw rows, pair metrics and exact failure categories.
10. Diagnose the result according to the decision tree below. Replicate promising fixed
    configurations; optional typed branch, MBPP transfer and direct KV are dependent phases.
11. Write an updated handover with completed/pending steps, reproducible commands, hashes,
    checkpoint paths, latest results, GPU status and the next concrete action.

Decision tree:

| Observation | Next action |
|---|---|
| Paired-data audit fails | Fix data/status/leakage; no architecture claims |
| Context2 differs from full-layer reference | Fix masks/positions/pooling; no long runs |
| Decoder loss gives no encoder gradients | Fix gradient path/target shift before training |
| Cannot overfit code identity | Debug data/optimization; compare representations on the small set |
| Code identity good, continuous repair bad, snapped repair good | Optimize decoder-facing fidelity; MSE/cosine alone is insufficient |
| Mean decoder succeeds, mean vector fails | Supervision is a key factor; contextual architecture novelty is not required |
| Context2 vector succeeds, mean vector fails | Order/context representation matters; test role/value controls |
| Context2 decoder succeeds only in-range | Use typed value/relational branch and factor OOD shifts |
| IO-only matches full traces | Scope claim to behavioral specification; test intermediate runtime utility |
| Pair gate succeeds but unseen MBPP does not | Report mechanism-only result; increase program diversity and audit overlap |
| No configuration has useful pair control | Record negative result and failures; don't infer a latent-medium impossibility |

## 14. Deliverables and reporting discipline

Required immediate deliverables are code/config/tests for Phases A–E, audited paired data
manifest, overfit and seed-401 matrix artifacts, and a readable local results/handover note.
Dependent phases should run only when their gates justify them. This is not a request to
launch every possible model or dataset job without evaluating intermediate results.

The results note must distinguish measured facts from hypotheses, program generalization
from prompt/evidence transfer, and lexical patch differences from executed behavior.
Report IO-only and trace budgets, hidden-test separation, trainable counts, GPU costs and
seed uncertainty. Fixed-size latent messages cannot promise exact preservation of arbitrary
length traces; task-relevant retention is the empirical claim to test.

Code/results belong in Git; trainable-only release weights belong in the existing Hugging
Face repository if publication is requested/authorized. Never re-upload base Qwen weights.
Use existing export scripts as a template, include matching schema/config/model-card and
checksums, and record the behavior ordering. Pushing is a separate operation from local
implementation; do not print, embed or persist auth keys in reports/commits.

## 15. Copy-paste kickoff prompt for the implementation agent

> Work in `/workspace/hf_cache/test/my-project/codelm_exp2`. Read
> `docs/implementation_plan_runtime_latent_2026-09-17.md` fully and the linked literature
> review. Implement Phases A–E in dependency order, starting with matched paired evidence
> and truthful test statuses. Reuse the local `.venv`, pinned Torch/Transformers and frozen
> Qwen2.5-1.5B-Instruct. Neural work must use the A100; make no systemwide changes.
> Preserve historical artifacts and checkpoints. New CLI/config names in the plan are
> implementation targets. Verify data invariants, contextual-layer parity, receiver loss
> alignment and gradients before a long job. Commit tested scoped changes before training.
> Run smoke and the fixed training-set overfit gate, then the seed-401 2x2 only if gates
> pass. Save configs, hashes, checkpoints, raw evaluation rows, controls and a handover note.
> Evaluate evidence values and bundle length separately. Do not claim unseen-code
> generalization or useful role control from training cosine, equal aggregate counts or
> different patch text. If blocked by a failed gate, diagnose the specified failure rather
> than starting MBPP/KV phases or adding unrelated technologies. Keep progress updates
> concise and report negative results. No LLM data-generation API is needed for this pilot.
