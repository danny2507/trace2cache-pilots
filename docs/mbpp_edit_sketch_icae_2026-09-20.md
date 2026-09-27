# Edit-sketch ICAE for MBPP latent communication — 2026-09-20

Communicate the *edit*, not the observation. Stage 0 showed the frozen 3B
repairs from a one-line `REPLACE` (114/128) and does not repair from a
debugger gist (54/128). Do not retune Stage C. Do not claim KV injection.
Canonical source is a train-only decoder target; it never enters a test-split
prompt.

## Why this architecture

The skiphide/const 72/128 win is a generic 8-slot prefix (`true = shuffled =
corrupted`). The gist autoencoder bound instance-specific states (ranking left
the margin, 51/128 patches changed under shuffle) and *hurt* Repair@1. Binding
the wrong payload is worse than a constant prefix.

Keep the const prefix as a frozen floor. Add an ICAE-style residual that can
only help if the eight gist tokens carry a sketch the decoder can apply.

```
Z = const + residual_zero(h_gist)
```

`residual` is Linear(2048, 2048) with zero weight and bias, so step 0 equals
the 72/128 const run. `h_gist` is the frozen LM's last-layer hidden state at
eight learned gist tokens appended to a chat of buggy source, public test, and
`gist_text(events)`.

## Method

Frozen `Qwen/Qwen2.5-Coder-3B-Instruct` BF16. Eight continuous states spliced
as `inputs_embeds` at `<TRACE2CACHE_REPAIR_CODE>`. Not per-layer KV.

1. **Frozen const prefix.** Load `checkpoints/mbpp_generalization/latent_seed1001_const.pt`
   slots `[8, 2048]` as a buffer. Do not train them.
2. **ICAE encode.** Frozen backbone (`model.model`) over encode-chat + 8 gist
   token embeddings. Read last-layer hidden at those 8 positions. Project with
   the zero-init residual onto const.
3. **Sketch readout.** Teacher-force `edit_sketch(buggy, canonical)` — a
   one-line `REPLACE` from `ast.unparse` line diff — through the decoder splice.
   The sketch prompt has the marker and omits the public test. Canonical text
   is the target sequence only.
4. **Edit-span repair NLL.** Mask patch loss to tokens outside the shared
   prefix/suffix with the buggy program. Hide the public test at train.
   Evaluation still includes it.
5. **Shuffle is another program.** ICAE encode of `shuffled_latent` uses the
   other example's buggy source, public test, and events. Encoding this program
   with another program's events is not a shuffle for this encoder.
6. **No cosine separation. No vector reconstruction. No gist-of-LINE/STATE/RETURN
   as the reconstruction target.** Rank on sketch NLL.

Loss:

```
L = L_edit
  + sketch_weight * L_sketch_true
  + contrastive_weight * relu(margin + L_sketch_true - L_sketch_shuffled)
```

Restore the checkpoint with the best sketch-ranking plus true/shuffled cosine
after step 200. Do not restore a constant-prefix-style early collapse if ranking
never moves; the same rule as gist.

Train on every in-budget train example. Do not filter to wrong-value failures:
the sketch target is defined for every first-order mutant.

## Conditions and decision

Same matched protocol conditions as `docs/mbpp_generalization_protocol_2026-09-19.md`.
`oracle_sketch` is a Stage 0 diagnostic and is not in the default `full` table.

A latent communication claim needs higher task-level Repair@1 than
`compact_text` **and** `no_evidence`, plus a shuffle or corruption drop. If
true ≈ shuffled, report a generic residual on the const prefix. If ranking
moves and Repair@1 falls, report bound-but-harmful sketch, as with gist.

Canonical `REPLACE` is never in an evaluation prompt. The eval panel is the
frozen 128 tasks (ids 11–177).

## Run

`artifacts/mbpp_generalization/sketchicae_seed1001/` with seed 1001, 800 steps,
learning rate 1e-3, sketch-weight 1.0, contrastive-weight 3.0, reconstruction
and separation weights 0, `--edit-sketch-icae --edit-span-loss`.
Checkpoint: `checkpoints/mbpp_generalization/latent_seed1001_sketchicae.pt`.

Gradient checkpointing is on for the encode backbone. The other GPU job is
left alone (`--min-free-gib 6`).

## Result (seed 1001, 128-task panel)

Sketch ranking left the margin (24 / 41 logged steps below 0.099; last step
ranking 0.0, cosine 0.995). Best checkpoint restored from step 700. 66 / 128
true vs shuffled patches differ, so the decoder is reading the residual.
That is not a communication win.

| Condition | Repair@1 |
| --- | ---: |
| `true_latent` | 59 |
| `shuffled_latent` | 60 |
| `corrupted_latent` | 57 |
| `compact_text` | 64 |
| `io_text` | 67 |
| `no_evidence` | 68 |
| const prefix (frozen floor) | 72 |
| Stage 0 `oracle_sketch` (text, not protocol) | 114 |

McNemar: true vs no_evidence 11–20 (p = 0.15); true vs compact_text 13–18
(p = 0.47); true vs const 5–18 (p = 0.0106); true vs shuffled 6–7 (p = 1.0);
true vs corrupted 7–5 (p = 0.77). Help/hurt vs no_evidence 11–20; vs const
5–18.

The residual bound instance-specific states and moved off the const floor.
Binding an inferred sketch is net-negative for hidden-test repair, while
putting the same `REPLACE` in the prompt (Stage 0) is 114/128. The frozen
decoder can apply a text sketch; it cannot usefully apply this eight-slot
encoding of one. Report bound-but-harmful sketch residual, not evidence-use
and not a generic skiphide-style prefix. Do not claim latent communication.
Full table: `artifacts/mbpp_generalization/RESULTS_2026-09-19.md`.
