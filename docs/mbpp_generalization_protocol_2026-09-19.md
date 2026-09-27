# MBPP generalization protocol — 2026-09-19

Compare eight model-native latent states with matched text-only prompting on
MBPP mutants whose hidden tests never appear in the prompt.

## Frozen choices

- Receiver: frozen `Qwen/Qwen2.5-Coder-3B-Instruct`, BF16, greedy decoding.
- Official MBPP ID split. Every mutant of a task stays in that task's split.
- Public evidence is exactly one official failing assertion plus its execution
  trace. Remaining official assertions, plus canonical-passing MBPP / MBPP+
  extras with disjoint call hashes, are hidden tests.
- A task is eligible only if canonical source passes every official test, the
  mutant fails the public test, the mutant fails at least one hidden test, and
  public/hidden call hashes are disjoint.
- Canonical source is an offline oracle and the train-split supervision target.
  It is never placed in a prompt.

## Conditions

Same buggy program, public test, receiver, and decoder budget. The English
specification is omitted from every condition: it restates the intended
behavior and makes `no_evidence` a rewrite-from-spec baseline rather than a
repair baseline. Training may also withhold the public test so the encoder
must put I/O into the eight states; evaluation always keeps the public test.

| Condition | Evidence |
| --- | --- |
| `no_evidence` | `unavailable` |
| `io_text` | assertion + pass/fail only |
| `compact_text` | named, test-grouped full public trace |
| `true_latent` | eight continuous states from that trace |
| `shuffled_latent` | states from a different-task trace |
| `corrupted_latent` | I/O kept, intermediate payloads replaced |

Repair@1 requires the extracted patch to pass every hidden test. Extraction,
policy, runtime, and timeout failures stay in the denominator. The uncertainty
unit is the task.

## Decision

A latent win needs higher task-level Repair@1 than `compact_text` and
`no_evidence`, plus a drop under shuffle or runtime corruption. If true and
shuffled match, report a generic soft-prompt effect rather than evidence use.

## Result (seed 1001, 128-task panel)

`true_latent` 72 / `compact_text` 64 / `no_evidence` 68. The same 72 holds for
shuffled, corrupted, and an event-independent 8-embedding prompt-tune
(`const_seed1001`, 16 384 params). McNemar vs compact_text is 13–5
(p = 0.096). Report a generic soft-prompt, not evidence use.

Gist autoencoder (`gist_seed1001`): ranking left the margin and 51/128 patches
differ under shuffle, but Repair@1 falls to 54 vs 68 no_evidence (p = 0.009).
Bound-but-harmful gist, not a communication win.

Edit-sketch ICAE (`sketchicae_seed1001`): frozen const prefix plus a zero-init
residual from frozen-LM gist tokens, trained to reconstruct the oracle
`REPLACE`. Ranking left the margin (24/41 logs below 0.099; last ranking 0.0).
66/128 patches change under shuffle, so the decoder reads the residual.
Repair@1 is 59 vs 68 no_evidence, 64 compact, 72 const (McNemar vs const 5–18,
p = 0.0106). Shuffle 60, corruption 57. Bound-but-harmful sketch residual;
it did not keep the const floor. Full table:
`artifacts/mbpp_generalization/RESULTS_2026-09-19.md`.
Method: `docs/mbpp_edit_sketch_icae_2026-09-20.md`.

Haque Gate 0 (`haque_text_seed1001`): `gist_text` 63 and `collated_text` 60
vs `no_evidence` 68 and `compact_text` 64. A short text trace (rule-based
OPT analogue, not GPT-4) does not beat the error prompt on this panel.
Do not train another trace encoder here. Method:
`docs/latent_trace_channel_2026-09-20.md`.

RunBugRun Gate 0 (`runbugrun_text_seed1001`, 128 Python test-split bugs):
`gist_text` 28, `compact_text` 26, `io_text` 24, `no_evidence` 23,
`collated_text` 21. Gist vs no_evidence 6–1 p = 0.125; vs io_text 4–0
p = 0.125. First non-negative discrepancy result; not McNemar. Do not
train a K-slot trace encoder.

