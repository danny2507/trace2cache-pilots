# Trace2Cache

Trace2Cache is an empirical pilot for a narrow question:

> Can a frozen small CodeLM use compact, model-native runtime evidence more reliably than a
> textual execution trace for program repair?

The project intentionally starts with a falsification gate. Pilot 1 measures whether textual trace
utility degrades as the trace becomes longer/noisier on the exact target model. Pilot 2 will train a
small soft-prefix trace encoder on execution-state probes. Direct KV injection is only attempted if
both gates pass.

## Current target

- Model: `Qwen/Qwen2.5-Coder-3B-Instruct` (frozen, BF16)
- Hardware: one A100 40 GB
- Task: deterministic Python function repair, checked on held-out tests
- Conditions: failing test only, compact text trace, full text trace, structured JSON trace

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

Results are append-only JSONL under `artifacts/pilot1/`, so an interrupted run can be resumed.

## Research safeguards

- Report repair success only when every hidden test passes, not merely the shown failing test.
- Match prompt instructions and decoding across conditions.
- Record actual tokenizer lengths and generation latency.
- Compare latent messages against token-budget-matched text and dynamic-slice text.
- Freeze the CodeLM for the main claim; report trainable parameter count.
- Treat soft-prefix and KV injection as distinct interventions.

See `docs/research_plan.md` for hypotheses, controls, and stop/go criteria.

Current directional results, including Refactory and latent true-vs-shuffled controls, are recorded
in `docs/pilot_results_2026-09-13.md`.
