# Gist autoencoder for MBPP latent communication — 2026-09-20

Force the frozen receiver to *read* eight continuous states by reconstructing a
short debugger gist through its next-token head, then train repair only on the
edit span. Do not retune Stage C. Do not claim KV injection.

## Why the previous runs failed

Full-patch NLL is insensitive to the eight slots: the public test and buggy
source already determine most tokens. Ranking on that NLL stuck at the margin.
Vector reconstruction toward mean-pooled native embeddings left the LM manifold
and hurt Repair@1. Cosine separation made encodings differ; the decoder ignored
them. A 16 384-parameter constant prefix matched the 4.7M encoder at 72/128, so
that win was a generic soft-prompt.

Gold `compact_text` (~869 tokens) is also net-negative versus `no_evidence` on
this first-order-mutant cohort. Long traces are the wrong payload.

## Method

Keep frozen `Qwen/Qwen2.5-Coder-3B-Instruct`, eight spliced `inputs_embeds`
slots, and `RoleAwareEventEncoder`. Change the payload and the tokens in the
loss.

1. **Narrow events.** Encode only TEST / PASS / FAIL plus the last LINE, STATE,
   and RETURN. Drop loop-unrolled intermediates.
2. **Gist readout through the frozen LM.** Training prompt: buggy source, no
   public test, no text trace, eight slots. Teacher-force a short gist (clipped
   assertion, outcome, last line/return). Shuffled-task slots must produce
   higher gist NLL.
3. **Edit-span repair NLL.** Mask patch loss to target tokens outside the
   shared prefix/suffix with the buggy program. Hide the public test at train.
   Evaluation still includes it.
4. **Train filter.** Encoder updates use wrong-value public failures
   (`test outcome fail`), not immediate exceptions. The frozen 128-task eval
   panel does not change. Also score the predeclared wrong-value subset of that
   panel, chosen from events before seeing Repair@1.

Loss:

```
L = L_edit + gist_weight * L_gist_true
  + contrastive_weight * relu(margin + L_gist_true - L_gist_shuffled)
```

No mean-pool reconstruction. No cosine separation on the eight vectors.
Restore the checkpoint with the best gist-ranking plus separation after step
200.

## Conditions and decision

Same matched conditions as `docs/mbpp_generalization_protocol_2026-09-19.md`.
A latent communication claim needs higher task-level Repair@1 than
`compact_text` and `no_evidence`, plus a shuffle or corruption drop. If gist
ranking never leaves the margin, or true and shuffled patches match, report a
generic prefix again.

Diagnostics that must move before interpreting the 128-task panel: gist ranking
below margin, true/shuffled cosine not collapsed to 1, generated patches that
differ under shuffle.

## Run

`artifacts/mbpp_generalization/gist_seed1001/` with seed 1001, 800 steps,
learning rate 1e-3, gist-weight 1.0, contrastive-weight 3.0, reconstruction
and separation weights 0, `--gist-latent --edit-span-loss --train-value-failures`.

The frozen 128-task panel is unchanged (task_ids 11–177). 112 of those tasks
are predeclared wrong-value public failures in
`artifacts/mbpp_generalization/gist_seed1001/value_failure_panel.json`.
Score both the full panel and that subset. Do not retune after seeing Repair@1.

## Result (seed 1001, 128-task panel)

Gist ranking left the margin (18 / 41 logged steps below 0.099; last step
ranking 0.0, cosine 0.97). 51 / 128 true vs shuffled patches differ, so the
decoder is reading the slots. That is not a communication win.

| Condition | Repair@1 |
| --- | ---: |
| `true_latent` | 54 |
| `shuffled_latent` | 53 |
| `corrupted_latent` | 56 |
| `compact_text` | 64 |
| `io_text` | 67 |
| `no_evidence` | 68 |

McNemar: true vs no_evidence 6–20 (p = 0.009); true vs compact_text 6–16
(p = 0.053); true vs shuffled 6–5 (p = 1.0); true vs corrupted 0–2 (p = 0.50).
Predeclared 112-task wrong-value subset: true 49 / no_evidence 59 / compact 56
/ shuffled 48.

The gist objective bound instance-specific states. Binding this I/O-plus-last-
line gist is net-negative for hidden-test repair, matching gold `compact_text`
already being worse than `no_evidence`. Report bound-but-harmful gist, not
evidence-use and not a generic skiphide-style prefix. Do not claim latent
communication. Full table: `artifacts/mbpp_generalization/RESULTS_2026-09-19.md`.
