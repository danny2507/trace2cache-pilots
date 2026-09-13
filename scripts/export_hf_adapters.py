#!/usr/bin/env python3
"""Export trainable-only Trace2Cache checkpoints as safe Hugging Face artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from safetensors.torch import save_file
import torch


ROOT = Path(__file__).resolve().parents[1]

# Deliberately excludes early ~480 MB checkpoints that duplicated the frozen
# Qwen embedding table. These checkpoints contain only the learned interface.
EXPORTS = {
    "pair_encoder_seed29": (
        "compositional_pair_encoder_balanced_1200.pt",
        "compositional pair encoder; held-out value-pair gate",
    ),
    "pair_encoder_seed41": (
        "compositional_pair_encoder_balanced_seed41.pt",
        "compositional pair encoder; held-out value-pair gate",
    ),
    "pair_encoder_seed53": (
        "compositional_pair_encoder_balanced_seed53.pt",
        "compositional pair encoder; held-out value-pair gate",
    ),
    "sinkmarked_trace_resampler_seed71": (
        "trace_distillation_sinkmarked_2000.pt",
        "event resampler with semantic sink roles",
    ),
    "fixedpos_trace_resampler_seed71_negative_control": (
        "trace_distillation_fixedpos_3000.pt",
        "raw sequential event negative control",
    ),
    "causal_path_resampler_seed83": (
        "causal_path_distillation_seed83.pt",
        "typed causal-path event resampler",
    ),
    "causal_path_resampler_seed97": (
        "causal_path_distillation_seed97.pt",
        "typed causal-path event resampler",
    ),
    "causal_path_resampler_seed109": (
        "causal_path_distillation_seed109.pt",
        "typed causal-path event resampler",
    ),
    "causal_path_coarse_roles_seed83_ablation": (
        "causal_path_remove_roles_seed83.pt",
        "false/true candidate-role removal ablation",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / ".local" / "hf-export")
    args = parser.parse_args()

    source_dir = ROOT / "checkpoints" / "latent_probe"
    output = args.output.resolve()
    adapter_dir = output / "adapters"
    result_dir = output / "results"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "format": "safetensors",
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "frozen_receiver": True,
        "exports": {},
    }
    checksums: list[str] = []

    for name, (filename, description) in EXPORTS.items():
        source = source_dir / filename
        payload = torch.load(source, map_location="cpu", weights_only=True)
        state = payload["adapter_trainable"]
        forbidden = [key for key in state if "token_embedding.weight" in key]
        if forbidden:
            raise RuntimeError(f"{source} contains frozen receiver weights: {forbidden}")
        tensors = {
            key: value.detach().cpu().contiguous()
            for key, value in sorted(state.items())
        }
        destination = adapter_dir / f"{name}.safetensors"
        save_file(
            tensors,
            destination,
            metadata={
                "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
                "description": description,
                "source_checkpoint": filename,
            },
        )
        result = payload.get("result")
        if result is not None:
            (result_dir / f"{name}.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        relative = destination.relative_to(output).as_posix()
        digest = sha256(destination)
        checksums.append(f"{digest}  {relative}")
        manifest["exports"][name] = {
            "description": description,
            "source_checkpoint": filename,
            "parameters": sum(tensor.numel() for tensor in tensors.values()),
            "tensor_count": len(tensors),
            "sha256": digest,
        }

    shutil.copy2(ROOT / "docs" / "huggingface_model_card.md", output / "README.md")
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    print(f"Exported {len(EXPORTS)} adapters to {output}")
    print(f"Total bytes: {sum(path.stat().st_size for path in output.rglob('*') if path.is_file())}")


if __name__ == "__main__":
    main()
