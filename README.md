# Trace2Cache

Trace2Cache is an empirical pilot for a narrow question:

> Can a frozen small CodeLM use compact, model-native runtime evidence more reliably than a
> textual execution trace for program repair?

The project starts with falsification gates. Pilot 1 measures whether textual trace utility degrades
as the trace becomes longer/noisier. The latent pilots then test progressively harder interfaces:
native anchors, one-state multi-fact codes, held-out value combinations, full trace distillation,
and typed causal paths with heavy distractors. Direct KV injection is intentionally deferred until
the soft-state representation and real-code integration are established.

## Current status (2026-09-27)

The latent / K-slot / splice / Residualize / Coconut / sketch-distill families are
**closed**. Full scoreboard, kill calls, and the next SOTA-motivated menu (test-time
scaling, ICL pairs, TTT — not 8-slot) are in
[`docs/status_and_next_2026-09-27.md`](docs/status_and_next_2026-09-27.md).

Headline on the frozen 128 MBPP repair panel, greedy hidden-test Repair@1,
`Qwen/Qwen2.5-Coder-3B-Instruct` BF16:

- frozen `no_evidence` (buggy + public test): **68**
- direct repair LoRA (control, not a method): **97**
- oracle `REPLACE` in the eval prompt (ceiling, not a method): **114**
- sketch distill **91** (`sketch_distill_fail`); LDP Gate 1 **64** (`ldp_g1_fail`)

On stdout prediction, gold compact event *text* is 25 vs code-only 12; every
non-text channel of those events failed (`sft_fail` 3 vs 14, `coconut_fail` 5 vs 14).

Do not reopen a `*_fail` family. Do not train on the frozen 128. Shared A100: never
kill other users’ GPU PIDs.

Earlier toy paired-runtime and synthetic causal-path pilots remain in
[`docs/handover_2026-09-18.md`](docs/handover_2026-09-18.md) and
[`docs/pilot_results_2026-09-13.md`](docs/pilot_results_2026-09-13.md).
Trainable-only adapters: [`danny2507/trace2cache-pilots`](https://huggingface.co/danny2507/trace2cache-pilots).

## Reproduce

Create the project-local, pinned CUDA 11.8 environment and run:

```bash
uv venv --python /workspace/envs/AIC2026/bin/python --system-site-packages
uv pip install --python .venv/bin/python -e . --no-deps
export PYTHONPATH="$PWD/src"
.venv/bin/python scripts/check_env.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/run_pilot1.py --limit 2 --conditions test_only compact_text
```

`--system-site-packages` reuses the server's existing CUDA wheels and avoids a multi-GB duplicate
download. The project package itself is installed into the local `.venv`.

Results are under `artifacts/`; large local checkpoints remain ignored. To create safe,
trainable-only `safetensors` releases without duplicating the frozen base model:

```bash
.venv/bin/python scripts/export_hf_adapters.py
```

## Research safeguards

- Report repair success only when every hidden test passes, not merely the shown failing test.
- Match prompt instructions and decoding across conditions.
- Record actual tokenizer lengths and generation latency.
- Compare latent messages against token-budget-matched text and dynamic-slice text.
- Freeze the CodeLM for the main claim; report trainable parameter count.
- Treat soft-prefix and KV injection as distinct interventions.

See `docs/research_plan.md` for hypotheses, controls, and stop/go criteria.

Current directional results, including Refactory, three-seed latent replications, causal
interventions, and real Python trace examples, are recorded in
`docs/pilot_results_2026-09-13.md`.
