# Trace2Cache

Trace2Cache is an empirical pilot for a narrow question:

> Can a frozen small CodeLM use compact, model-native runtime evidence more reliably than a
> textual execution trace for program repair?

The project starts with falsification gates. Pilot 1 measures whether textual trace utility degrades
as the trace becomes longer/noisier. The latent pilots then test progressively harder interfaces:
native anchors, one-state multi-fact codes, held-out value combinations, full trace distillation,
and typed causal paths with heavy distractors. Direct KV injection is intentionally deferred until
the soft-state representation and real-code integration are established.

## Current status

- Text repair receiver: `Qwen/Qwen2.5-Coder-3B-Instruct` (frozen, BF16)
- Latent probe receiver: `Qwen/Qwen2.5-1.5B-Instruct` (frozen, BF16)
- Hardware: one A100 40 GB
- Text task: Python function repair, checked on held-out tests
- Latent task: compress paired, typed runtime events into one receiver-readable state

Across three causal-path seeds, the one-state latent reaches `58.0 ± 10.5%` macro accuracy on
unseen output pairs and much longer traces, versus `19.6 ± 1.4%` after message shuffling. Branch
flips and candidate-role swaps reduce accuracy to chance, supporting use of the intended causal
fields. This is controlled synthetic evidence; the generated real Python traces are not yet wired
into the latent encoder.

Trainable-only model artifacts are published separately at
[`danny2507/trace2cache-pilots`](https://huggingface.co/danny2507/trace2cache-pilots).

The latest paired-runtime update is a careful negative-control result: direct decoder
supervision gives useful paired repair behavior on known toy program families, but generic typed
value nodes, cross-latent counterfactual loss, and audited binding relations do not improve it.
It does not support a task-disjoint repair or direct-KV claim.  See
[`docs/handover_2026-09-18.md`](docs/handover_2026-09-18.md) for exact denominators, controls,
checkpoints, and release contents.

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
