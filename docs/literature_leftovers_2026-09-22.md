# What this literature still has untried — 2026-09-22

Standing constraints: frozen `Qwen/Qwen2.5-Coder-3B-Instruct` BF16 receiver
except LoRA when the method *is* training the LM; official Google MBPP ID
split; do not train on the frozen 128; do not overwrite frozen repair
`runbugrun_cohort_v1` test sha `ddae5cf9…` or sim
`execution_sim_runbugrun_v1` test sha `074dd6ce…`; do not hide stdin; do
not claim a channel if shuffle does not drop; not GainFill; not a negative
paper.

Event-chain SFT called **`sft_fail`** (3 vs 14). Coconut called
**`coconut_fail`** (latent 5 vs stdout 14). LDP Gate 1 called
**`ldp_g1_fail`** (true 64 vs no_evidence 68 vs shuffle 66). Sketch
distillation called **`sketch_distill_fail`** (91 vs direct LoRA 97).
Residualize, splices, tracer distill, gist/sketch ICAE, and repair Gate
0 traces are closed. Do not train LDP Gate 2. Do not train sketch
distill again.

## Already done — do not reopen

Haque-style traces in the prompt (raw, gist, collated, discrepancy, failing
line) on MBPP, Refactory, and RunBugRun, 3B and 7B. Compact intermediate
*text* for stdout prediction (McNemar win: 25 vs 12). Every non-text
channel of those same events: K-slot encoder, mean-pooled native splice,
vocab-snap, tracer distill, event-chain SFT, Coconut. Residual-Δ, gist
autoencoder, sketch ICAE, LDP Gate 1 (oracle REPLACE as K=8
`inputs_embeds`). ReflectionCoder-style sketch distillation vs direct
repair LoRA. Residualize. Shuffle never dropped on a splice or on
LDP Gate 1. Direct repair LoRA is 97/128 vs frozen `no_evidence` 68.

## In the papers, untried here, but a bad next step

CodeExecutor/TRACED continued pretraining is an occupied paper.
TraceFixer’s desired intermediate state is a different setting.
Cache-to-Cache / LatentPress / LaMAR per-layer KV is the same splice claim
one layer down, with title overlap. NALU/CLRS/IPA-GNN/DPE train an
executor, not a frozen CodeLM. Outcome RL on event-chain SFT would train a
teacher that already lost 3–14. 14B was deferred on purpose.

## Actually untried, and still on-constraint

1. **Coconut curriculum — closed `coconut_fail`.** Mix8/latent8 5/128 vs
   stdout control 14/128. Do not train Coconut again.

2. **xRAG’s actual second stage.** Distill used cosine to native event
   vectors. xRAG trains the projector so \(p(\text{stdout}\mid\text{code}, Z)\)
   matches \(p(\text{stdout}\mid\text{code}, \text{trace text})\) by KL from
   the 25-win teacher. Gold embeddings already hurt (teacher 5 vs
   `code_sft` 11). Different objective, same risk.

3. **LDP Gate 1 — closed `ldp_g1_fail`.** Oracle REPLACE as K=8 encode
   input: true 64, shuffle 66, no_evidence 68. Do not train Gate 2.
   Protocol fallback is Gate 3 (same-sequence continuation of a sketch
   segment), not another 8-slot splice. Do not write a negative paper.

4. **NExT as text, for repair.** Self-train short execution rationales
   filtered by a hidden-test-correct patch, not raw PySnooper dumps.
   Overlap with NExT is high if it works.

5. **Measurement, not a method.** Execution-sim Gate 0 was only RunBugRun
   gold programs. The same `code_input` vs `trace_text` panel was never
   run on official MBPP.

If the goal is still a 3B A/A* method paper, do not reopen splices,
event-chain SFT, Residualize, Coconut, LDP Gate 1/2, or sketch
distillation. Direct repair LoRA (97) is a control, not a method paper.
Skip LDP Gate 3 / xRAG / NExT-as-text unless asked. The 2026-09-27 menu
(not 8-slot, motivated from published 2025 SOTA) and the literature
review are in `docs/status_and_next_2026-09-27.md` §3 and §5:

1. **S\*** test-time scaling for repair (Findings EMNLP 2025) — first run.
2. **AuPair** ICL pairs (ICML 2025) — if S\* is a wash.
3. **Test-time training** on the public test (Akyürek ICML 2025).
4. **Inverse-Instruct** data (InverseCoder AAAI 2025) — only if 1–3 close,
   and only if it beats direct LoRA 97 (sketch distill already lost).

---

## Experiment 1 — Coconut curriculum for stdout prediction (closed `coconut_fail`)

Working title: **Think in States: Coconut Execution for a Small CodeLM**.

### Why this is not a closed method

| Closed method | What it did | Coconut |
| --- | --- | --- |
| Event-chain SFT | Teacher-force compact event *tokens* then stdout | Those tokens are stage 0 only |
| Tracer distill | Align projected hidden states to *debugger* embeddings | No tracer vectors |
| Native / snap splice | Frozen 3B reads mean-pooled event embeddings | Receiver LoRA; thoughts are its own hidden states |

Stage 0 is already done: `sft_events` LoRA
`checkpoints/mbpp_generalization/event_sft_events_seed1001.pt`, 3/128.
Coconut *starts from that checkpoint* and replaces event lines with
continuous thoughts. Control is the existing `sft_stdout` 14/128. Do not
retrain stage 0. Do not retrain the control.

### Protocol

Prompt stays `ask_events` (code + stdin, no gold trace, stdin not hidden).
A thought is the last hidden state of the LoRA 3B, concatenated as the
next input embedding. Language NLL is only on tokens after the thoughts.

| Stage | Steps | Thoughts \(k\) | Language target |
| --- | ---: | ---: | --- |
| `mix8` | 640 | 8 | remaining gold event lines (after the first 8) then stdout fence |
| `latent8` | 640 | 8 | stdout fence only |

Init: event-chain LoRA. LoRA r=16, alpha=32, dropout 0.05, q/k/v/o, lr
1e-4, accum 4, seed 1001, max_sequence_tokens 4096. Train on disjoint 320
sha `02ddf7fe…`. Eval greedy last-fence stdout on frozen 128 sha
`074dd6ce…`, max_new_tokens 512. Inference: 8 thoughts, then generate.
Do not train on the frozen 128.

### Kill rule

`coconut_pass` iff `coconut_latent` beats `sft_stdout` by McNemar
(left_only > right_only and \(p < 0.05\)). Else `coconut_fail`. Do not
shuffle unless pass. Do not start MBPP or 14B for this family unless
asked after a pass.

### Artifacts

`artifacts/mbpp_generalization/execution_sim_coconut_seed1001/`
`checkpoints/mbpp_generalization/coconut_{mix8,latent8}_seed1001.pt`
`scripts/run_coconut_execution.py`
`src/trace2cache/coconut_execution.py`
