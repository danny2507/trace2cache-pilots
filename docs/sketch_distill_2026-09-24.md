# Sketch distillation vs direct repair LoRA

Working title:

> **Don't Hide the Sketch: Structured Edit-Sketch Distillation for a Small CodeLM**

One-shot hidden-test repair. Same prompt as frozen `no_evidence`. The method
is the *training target*, not a latent channel.

## Claim

Teacher-forcing an oracle `REPLACE` line before the canonical patch teaches
the 3B to repair better than SFT on the patch alone.

## Why this is not a closed method

| Closed method | What it did | This |
| --- | --- | --- |
| Oracle REPLACE in the **eval** prompt | Stage 0, 114/128 | Oracle is train-only |
| Sketch ICAE / LDP Gate 1 | Encode REPLACE as K=8 `Z` | No splice. LoRA on the 3B |
| Event-chain SFT | Compact events then stdout | Repair, not stdout; target is REPLACE then patch |
| Self-Debug | Decode a diagnosis at **eval** | Eval prompt is one-shot, same as `no_evidence` |

Shape: ReflectionCoder (sketch/reflection sequence as SFT target, optional
dynamic mask on the sketch tokens). Ablation is direct patch SFT. Baseline
is the frozen 68 `no_evidence` rows — copied, not retrained.

## Protocol

Receiver: `Qwen/Qwen2.5-Coder-3B-Instruct` BF16 + LoRA r=16, alpha=32,
dropout 0.05, q/k/v/o. lr 1e-4, accum 4, 1280 steps, seed 1001. Official
MBPP ID split. Train 780 disjoint. Eval greedy hidden-test Repair@1 on
frozen 128 (`cohort_v1` sha `acc350b8…`, test sha `6d6600ff…`). Public
failing test ON. Spec OFF. Do not train on the frozen 128.

Prompt (train and eval, both LoRAs): `render_chat(..., "unavailable")`.

| Condition | Train target | Eval prompt |
| --- | --- | --- |
| `direct_sft` | fenced canonical patch | no_evidence prompt |
| `sketch_distill` | oracle `edit_sketch` then fenced patch | same prompt |
| `no_evidence` | — | copied from sketchicae, 68/128 |

`sketch_distill` zeros loss on sketch tokens with p=0.5 (ReflectionCoder
mask). Eval never includes the oracle REPLACE or the canonical program.

## Kill

`sketch_distill_pass` iff `sketch_distill` beats `direct_sft` **and**
`no_evidence` by McNemar (left_only > right_only and p<0.05). Else
`sketch_distill_fail`. Do not train it again on fail.

## Result — 2026-09-24, seed 1001. Call `sketch_distill_fail`.

Frozen 128, hidden-test Repair@1, greedy, spec off, public test on.

| Condition | Repair@1 |
| --- | ---: |
| `direct_sft` | **97** |
| `sketch_distill` | 91 |
| `no_evidence` (copied) | 68 |
| `io_text` | 67 |
| `compact_text` | 64 |

McNemar: sketch vs direct 13 vs 19, p=0.377 (sketch lost). Sketch vs
no_evidence 37 vs 14, p=0.0018 (beats the frozen prompt, fails the
ablation). Direct vs no_evidence 39 vs 10, p=3.8e-5. One sketch
extraction failure; zero for direct. Mask fired on 628/1280 distill
steps. Peak 12.9 GiB, 4483 s.

Oracle REPLACE in the **eval** prompt is still 114. Training on that
line then hiding it at eval is worse than SFT on the patch alone.
Do not train sketch distillation again. Direct repair LoRA is the
honest 3B repair floor on this panel; it is not this method.

## Artifacts

`artifacts/mbpp_generalization/sketch_distill_seed1001/`
`checkpoints/mbpp_generalization/sketch_distill_{direct,sketch}_seed1001.pt`
`scripts/run_sketch_distill.py`
`src/trace2cache/sketch_distill.py`
