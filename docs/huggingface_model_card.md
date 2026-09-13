---
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: pytorch
tags:
  - code
  - program-repair
  - execution-trace
  - latent-representation
  - adapter
---

# Trace2Cache research adapters

Trainable-only adapters from controlled Trace2Cache pilots. They test whether a frozen small LM can
consume structured runtime evidence through one model-native continuous state. These files are
research artifacts, not standalone language models or a production program-repair system.

The receiver was frozen `Qwen/Qwen2.5-1.5B-Instruct`. Each `.safetensors` file excludes the base
model and its token-embedding table. Architecture and evaluation code live in the
[Trace2Cache source repository](https://github.com/danny2507/trace2cache-pilots).

## Main findings

The compositional pair encoder generalized to unseen value combinations at 70.3 ± 11.1% macro
accuracy across three seeds, versus 19.5 ± 1.7% after shuffling latent messages. A full event
resampler then compressed 24–48 typed causal events with heavy distractors into one state. Across
three seeds it achieved 58.0 ± 10.5%, versus 19.6 ± 1.4% shuffled.

Causal interventions erased the gain: branch-selector flips scored 19.2 ± 4.9% and candidate-role
swaps scored 19.3 ± 5.1%. Removing false/true candidate polarity during training reduced seed-83
accuracy from 65.8% to 32.9%. These results support dependence on typed causal roles rather than
event position or a masked-sink shortcut.

## Contents

- `pair_encoder_seed{29,41,53}`: receiver-readable one-state teachers.
- `sinkmarked_trace_resampler_seed71`: successful semantic sink-marker pilot.
- `fixedpos_trace_resampler_seed71_negative_control`: raw sequential-event length-transfer failure.
- `causal_path_resampler_seed{83,97,109}`: main typed causal-path models.
- `causal_path_coarse_roles_seed83_ablation`: role-removal ablation.
- `results/`: evaluation payload stored with each original checkpoint.
- `manifest.json` and `SHA256SUMS`: provenance, parameter counts, and integrity hashes.

Load tensors without pickle execution:

```python
from safetensors.torch import load_file

state_dict = load_file("adapters/causal_path_resampler_seed83.safetensors")
```

Instantiate the matching adapter class from the source repository and load this state dict with the
frozen receiver. See the experiment scripts for the exact architecture and prompt insertion point.

## Limitations

The positive evidence is controlled and synthetic. The executed six-program Python corpus is not
yet connected to these adapters, so these weights do not establish real-world repair gains. Exact
modulo-delta composition remains weak and seed variance is substantial. Only one frozen receiver
family and one-state soft insertion have been evaluated; direct KV-cache injection has not yet been
tested. Do not interpret these artifacts as evidence that generated patches are safe or correct.
