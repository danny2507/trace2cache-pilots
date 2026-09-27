# Closed experiments and next SOTA leads — 2026-09-27

Canonical snapshot of the A/A* small-CodeLM method search. The project is **not**
a negative paper. Do not reopen a family whose call is `*_fail` /
`gate0_insufficient` / `aborted_name_leak`. Do not train on the frozen 128.

Receiver: `Qwen/Qwen2.5-Coder-3B-Instruct` BF16. Official Google MBPP ID split.
LoRA is allowed only when the method *is* training the LM. Shared A100: never
kill other users’ GPU PIDs. Python: `.venv/bin/python`.

---

## 1. What the numbers actually say

Three facts survived every later experiment.

1. **On repair, extra runtime text does not help this 3B.** Buggy code plus the
   failing public test (`no_evidence` 68/128) beats gist, collated, compact
   traces, discrepancy one-liners, and failing-line locators on MBPP, Refactory,
   and RunBugRun (3B and 7B). Encoding those traces as K-slot `inputs_embeds`
   is an unused prefix: shuffle never drops.

2. **On stdout prediction of a gold program, compact event *text* is a real
   reading aid** (`trace_text` 25 vs `code_input` 12, McNemar 15–2, p=0.0023).
   The 3B cannot *invent* those events (ask-events 5; event-chain SFT 3 vs
   stdout LoRA 14). Every non-text channel of the same events died.

3. **Training the 3B moves repair; encoding a gold intermediate does not.**
   Direct repair LoRA is **97/128** vs frozen `no_evidence` 68 (McNemar 39–10,
   p=3.8e-5). Oracle `REPLACE` in the *eval* prompt is 114/128. The same
   `REPLACE` as a training prefix (sketch distill 91) or as K=8 latents (LDP
   64) loses. Direct LoRA is a **control**, not a method paper.

Remaining headroom on the frozen 128 that is not “put the oracle in the
prompt”: **97 → 114, seventeen tasks**. That is the size of a method result
on this panel.

---

## 2. Closed families (do not rerun)

All greedy, seed 1001, unless noted.

### 2.1 Early pilots (already on `main` before this dump)

Toy paired-runtime on frozen 1.5B, 8 soft states, known families: decoder NLL
helps in-family repair and does **not** establish task-disjoint generalization
or a benefit over text. Causal-path synthetic probes: the one-state latent
beats shuffle on unseen output pairs; that never transferred to real MBPP
repair. Docs: `docs/handover_2026-09-18.md`, `docs/pilot_results_2026-09-13.md`.

### 2.2 MBPP repair, frozen 128 (`cohort_v1`)

Hidden-test Repair@1. Spec off. Public failing test always in the prompt.
Cohort sha `acc350b87778d21a6ef441ec9cab4e4ef0d6cbaca1061703bb069a476f658baa`.
Test sha `6d6600ff1898d6881c9ac0f4cca30d868c6bb549d2dccd415e68761c3d96f9cb`.

| Run | Repair@1 | shuffled | vs `no_evidence` 68 | call |
| --- | ---: | ---: | --- | --- |
| `no_evidence` (buggy + public test) | 68 | — | — | frozen floor |
| `io_text` | 67 | — | lose | traces lose |
| `compact_text` | 64 | — | lose | traces lose |
| const 8-slot (no events) | 72 | 72 | 12–8, p=0.50 | unused prefix |
| gist autoencoder | 54 | 53 | 6–20, p=0.009 | bound, harmful |
| sketch ICAE | 59 | 60 | 11–20, p=0.15 | bound, harmful |
| haque `gist_text` | 63 | — | 5–10, p=0.30 | do not encode |
| discrepancy gist | 62 | — | 5–11, p=0.21 | do not encode |
| LDP Gate 1 (oracle REPLACE as K=8) | 64 | 66 | 14–18, p=0.60 | **`ldp_g1_fail`** |
| sketch distill LoRA (REPLACE then patch) | 91 | — | 37–14, p=0.0018 | **`sketch_distill_fail`** vs direct |
| **direct repair LoRA** | **97** | — | 39–10, p=3.8e-5 | **control, not a method** |
| oracle REPLACE **in the eval prompt** | **114** | — | 46–0 | Stage 0 ceiling, not a method |

Sketch vs direct: 13–19, p=0.377. Do not train LDP Gate 2. Do not train sketch
distill again.

### 2.3 Refactory repair Gate 0 (119 student programs)

`no_evidence` 53 > gist 44 > compact 42 > collated 39. Gist vs no_evidence
2–11, p=0.0225 (gist **loses**). Do not train.

### 2.4 RunBugRun repair Gate 0 (frozen test sha `ddae5cf9…`)

| Condition | 3B | 7B |
| --- | ---: | ---: |
| gist_text | 28 | 41 |
| io_text | 24 | 42 |
| no_evidence | 23 | 40 |
| compact_text | 26 | — |
| line_text | 20 | — |
| discrepancy one-liner | 23 | — |

3B gist vs no_evidence 6–1, p=0.125. 7B 5–4, p=1.0. Residual-Δ v1/v2 and
discrepancy-embed: shuffle does not drop. Do not train a K-slot repair
encoder. Do not start 14B unless asked.

### 2.5 Residualize (MBPP lookup-table residual)

Hypothesis: 3B writes a PE-style table under I/O-only prompts. Kill required
≥10 public-pass table and ≥10 generic, table hidden-rate < generic, Fisher
p<0.05.

| Panel | call |
| --- | --- |
| spec + names | `gate0_insufficient` (0 public-pass tables) |
| nospec + names | `aborted_name_leak` |
| anon `def f` nospec | `gate0_insufficient` (5 tables < 10) |

Do not train. No fourth Residualize gate unless asked.

### 2.6 Execution-sim (stdout of a **gold** program)

Same 128 RunBugRun test programs. Metric: exact match on normalized stdout.
Gold stdout redacted from events. Test sha `074dd6ce…`.

| Condition | exact / 128 | call |
| --- | ---: | --- |
| gold `trace_text` | **25** | Gate 0 **pass** vs code |
| `code_input` | 12 | baseline |
| `trace_gist` | 14 | not a win |
| `true_latent` K=8 | 14 | `latent_fail` (shuffle 14) |
| `native_events` mean-pool | 11 | `native_fail` (shuffle 11) |
| `vocab_snap` | 10 | `snap_fail` (14 unique ids) |
| `code_sft` LoRA | 11 | distill control |
| `latent_teacher` | 5 | embeddings hurt |
| `latent_rollout` | 4 | `distill_fail` |
| frozen `ask_events` | 5 | `gate0_sft_justified` |
| `sft_stdout` LoRA | **14** | control |
| `sft_events` LoRA | **3** | **`sft_fail`** (0–11, p=0.00098) |
| Coconut mix8 / latent8 | **5 / 5** | **`coconut_fail`** (latent vs stdout 2–11, p=0.022) |

Gold events in the prompt help. Asking the 3B to emit them, splice them, snap
them, distill them, or think them as Coconut states all fail. Kill those
families.

---

## 3. Occupied published methods — do not clone

| Paper | Venue | Why occupied |
| --- | --- | --- |
| NextCoder + SeleKT | ICML 2025 | Qwen2.5-Coder-3B edit + sparse adaptation |
| SLM-as-a-judge | ICSE 2026 | small model as judge |
| RLEF | ICML 2025 | RL from execution feedback |
| μCODE | ICML 2025 | code RL |
| ACECODER | ACL 2025 | process rewards for code |
| CoCoS | Findings EMNLP 2025 | online RL self-correction, 1B on MBPP |
| GainFill | — | senior overlap |
| Cascaded / NTR | — | template-then-patch |
| Self-Debug | ICLR 2024 | decode a diagnosis at eval |
| CodeExecutor / TRACED | 2023 | continued pretraining on traces |

Citing them as related work is required. Reimplementing them as *our* method
is not a paper.

---

## 4. Next directions (motivated from 2025–2026 published SOTA)

**Not limited to 8-slot / latent / splices.** Those families are scientifically
closed on this hardware. Published A* small-CodeLM papers in 2025 are
**inference-time compute, ICL pairs, test-time training, and self-instruct
data** — not another encoder of REPLACE.

Each item is a lead. Write a kill rule before any GPU job. Control is the
strongest honest baseline (direct LoRA 97 on repair; `sft_stdout` 14 on sim).
Do not hide the public test. Do not train on the frozen 128.

### N1. S*-style test-time scaling for repair — first recommended run

- **Paper:** Li et al., [S*: Test Time Scaling for Code Generation](https://aclanthology.org/2025.findings-emnlp.865/), Findings of EMNLP 2025.
- **SOTA fact:** hybrid parallel + sequential scaling with execution-grounded
  selection; a **3B** model outperforms GPT-4o-mini; GPT-4o-mini + S* beats
  o1-preview by 3.7% on LiveCodeBench.
- **Why it fits us:** this panel has only ever been greedy. Direct LoRA 97 is
  one sample. The public failing test is a legal verifier (hidden tests stay
  hidden). Headroom to the oracle-in-prompt ceiling is 17 tasks.
- **Gate 0 (cheap, no new weights):** frozen 3B and LoRA-97 checkpoint, budget
  N ∈ {4, 8, 16}. Conditions: (a) parallel best-of-N filtered by the public
  test; (b) sequential self-repair: if the public test fails, feed the failure
  back for one revision; (c) hybrid (a) then (b). Metric: hidden-test Repair@1.
- **Kill:** `sstar_pass` iff hybrid beats greedy LoRA 97 by McNemar (left_only >
  right_only, p<0.05) **and** beats naive public-test BoN of the same N. Else
  `sstar_fail`. Do not then invent a pairwise “distinguishing input” selector
  just to clone S*’s second stage.
- **Overlap:** S* is code *generation*. Applying it to hidden-test *repair* on
  official MBPP mutants is a transfer, not a clone, only if Gate 0 moves 97.
  If BoN already saturates toward 114, the paper is sample-efficiency of the
  sequential mix, not “we sampled N times.”

### N2. AuPair-style ICL pairs — second, if N1 is a wash

- **Paper:** Mavalankar et al., [AuPair: Golden Example Pairs for Code Repair](https://proceedings.mlr.press/v267/mavalankar25a.html), ICML 2025, PMLR 267:43276–43301.
- **SOTA fact:** inference-time (wrong, fixed) pairs as 1-shot ICL beat
  best-of-N and standard self-repair on competitive programming, 5 LLMs, 7
  datasets. No SFT.
- **Why it fits us:** 780 disjoint train `(buggy, canonical)` pairs already
  exist. This panel has never used ICL. Frozen 3B 0-shot is 68; LoRA 97 is the
  trained floor.
- **Gate 0:** 1-shot retrieved train pair (BM25 / embedding / random) vs 0-shot
  frozen 68 vs greedy LoRA 97. Same public test in the prompt. No LoRA in the
  ICL conditions unless the method *is* “LoRA + ICL.”
- **Kill:** ICL beats 0-shot frozen by McNemar. To be a method versus training,
  it must beat LoRA 97 or beat LoRA at matched inference budget (1 vs 1+shot
  tokens). Else `aupair_fail`. Do not copy AuPair’s complementarity ranking as
  the contribution until random/retrieved pairs already help.

### N3. Test-time training on the public test

- **Paper:** Akyürek et al., [The Surprising Effectiveness of Test-Time Training for Few-Shot Learning](https://proceedings.mlr.press/v267/akyurek25a.html), ICML 2025, PMLR 267:942–963. Related: Hu et al., [Test-Time Learning for Large Language Models](https://proceedings.mlr.press/v267/hu25z.html), ICML 2025.
- **SOTA fact:** temporarily updating weights at inference from the input (or
  from ICL examples) is a 6× few-shot gain on ARC for 8B; TLM reports ≥20%
  domain-adaptation gains via perplexity minimization + LoRA.
- **Why it fits us:** per-problem LoRA step on the public failing I/O, discarded
  after the patch, is not “we fine-tuned 3B on 780 mutants.” Code-TTT for APR
  is not an occupied ICML/ACL title.
- **Gate 0:** one (or few) LoRA steps on a public-test objective (NLL of a
  public-pass candidate sampled from the current model, or TTT on a retrieved
  train analog). Then greedy decode. Compare to greedy LoRA 97 *without* TTT.
- **Kill:** TTT beats greedy LoRA 97 by McNemar p<0.05. Else `ttt_fail`. Do not
  turn this into GRPO (occupied RL family).

### N4. Inverse-Instruct data, not REPLACE-then-patch

- **Paper:** Wu et al., [InverseCoder: Self-improving Instruction-Tuned Code LLMs with Inverse-Instruct](https://ojs.aaai.org/index.php/AAAI/article/view/34742), AAAI 2025.
- **SOTA fact:** code→instruction is easier than instruction→code; generating
  extra instructions from existing code and fine-tuning lifts HumanEval+ /
  MBPP+ for open 6–7B models. Related data recipe: SCoder progressive
  self-distillation, Findings EMNLP 2025.
- **Why it fits us:** sketch distill already showed that teacher-forcing the
  oracle `REPLACE` line then the patch **loses** to direct patch SFT (91 vs
  97). Inverse-Instruct is the opposite direction: from `(buggy, fixed)` emit
  a natural-language “what was wrong / how to fix it” and train on
  instruction diversity, not on a gold sketch the 3B must hide at eval.
- **Kill:** Inverse-Instruct LoRA beats direct LoRA 97 by McNemar. Else
  `inverse_fail`. High overlap risk if it is “we synthesized more MBPP-like
  instructions.” Only run if N1–N3 are closed and the claim is a data recipe
  with a 3B-specific twist.

### Explicitly not next

- Any new splice of events or REPLACE into a frozen 3B (K-slot, native,
  snap, residual-Δ, discrepancy-embed, xRAG KL, LDP Gate 2/3).
- Event-chain SFT v2, Coconut v2, sketch distill v2, Residualize n≥10.
- NextCoder SeleKT, SLM-as-a-judge, RLEF / μCODE / ACECODER / CoCoS / GainFill.
- RepairLLaMA representation replication unless N1’s BoN saturates and the
  remaining question is infill vs full rewrite (oracle REPLACE 114 already
  answers “the 3B can apply an infill sketch”).
- Starting 14B “just to see.” Hiding stdin. Training on the frozen 128.
- Writing a negative paper out of the closed families.

---

## 5. How to eval a new idea

1. Write the kill rule **before** training. McNemar on the frozen 128, p<0.05,
   left_only > right_only.
2. Control is the strongest honest baseline (direct LoRA 97, not frozen 68,
   once LoRA exists). Do not drop the public test.
3. If the claim is a channel: shuffle must drop. Identical pass/fail under
   shuffle = unused prefix.
4. Text / inference Gate 0 first. If gold text or cheap BoN does not move the
   metric, do not encode it and do not add a second-stage selector.
5. Never train on the frozen 128. Assert disjoint `example_id`, `task_id`,
   `source_hash`. Verify cohort shas before writing artifacts.
6. Hidden tests are Repair@1 for repair; last fenced block is stdout for sim.
7. `.venv/bin/python`. Do not kill other GPU PIDs.

---

## 6. Frozen hashes (do not overwrite)

| Object | sha256 prefix |
| --- | --- |
| MBPP repair `cohort_v1` | `acc350b87778d21a6ef441ec9cab4e4ef0d6cbaca1061703bb069a476f658baa` |
| MBPP repair test.jsonl | `6d6600ff1898d6881c9ac0f4cca30d868c6bb549d2dccd415e68761c3d96f9cb` |
| MBPP repair train 780 | `dfb9a8aab098d82598d7f6308fd640016c3061d6c7b6511b5de9d8e83a6a6792` |
| RunBugRun repair test | `ddae5cf90f72357efaa1465a432abfbbc860e94506d35fe488f2312a27a7c29c` |
| Execution-sim test n=128 | `074dd6cecdd960970356907168b6f3ea5c086ae9b487ca56aecec6d0b943276c` |
| Execution-sim train n=320 | `02ddf7fe8cb110397250d30142413b0c3234e1a476fcf3d0db4fb6c15fad3390` |

Large `rows.jsonl` / `train.jsonl` stay local (checkpoints are gitignored).
Rebuild via `scripts/build_*_cohort.py`. Per-run `summary.json` is in
`artifacts/mbpp_generalization/*/summary.json`.

---

## 7. File map for the closed 2026-09 work

| Path | Role |
| --- | --- |
| `src/trace2cache/mbpp_generalization.py` | RepairExample, prompts, hidden-test eval |
| `src/trace2cache/sketch_distill.py` | sketch vs direct targets + kill |
| `scripts/run_sketch_distill.py` | two repair LoRAs |
| `src/trace2cache/ldp.py` / `scripts/run_ldp_gate1.py` | LDP Gate 1 |
| `src/trace2cache/execution_sim.py` | sim examples, events, McNemar |
| `scripts/run_event_chain_sft.py` | event vs stdout LoRA |
| `src/trace2cache/coconut_execution.py` | Coconut schedule + kill |
| `src/trace2cache/residualize.py` | Residualize (closed) |
| `docs/handover_2026-09-22.md` | successor protocol, eval checklist |
| `docs/literature_leftovers_2026-09-22.md` | leftover log (latent-era) |
| `artifacts/mbpp_generalization/RESULTS_2026-09-19.md` | early repair scoreboard |

Venue target is unchanged: ICLR 2027 is too tight. Realistic A* is ICML 2027
or ARR → ACL/EMNLP 2027.
