#!/usr/bin/env python3
"""Export trainable-only MBPP trace encoders as a safe Hugging Face release bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from safetensors.torch import save_file
import torch


ROOT = Path(__file__).resolve().parents[1]
EXPORTS = {
    "mbpp_initial_seed173": (
        "latent_repair_seed173.pt",
        "latent_repair_seed173.json",
        "initial full-patch objective; generic-prefix diagnostic",
    ),
    "mbpp_contrastive_augmented_seed173": (
        "contrastive_augmented_seed173.pt",
        "contrastive_augmented_seed173.json",
        "expanded mutations plus different-task shuffled-trace hinge objective",
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
    parser.add_argument("--output", type=Path, default=ROOT / ".local" / "hf-mbpp-release")
    args = parser.parse_args()

    output = args.output.resolve()
    adapter_dir = output / "adapters"
    result_dir = output / "results"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "format": "safetensors",
        "base_model": "Qwen/Qwen2.5-Coder-3B-Instruct",
        "frozen_receiver": True,
        "exports": {},
    }
    checksums: list[str] = []

    for name, (checkpoint_name, result_name, description) in EXPORTS.items():
        source = ROOT / "checkpoints" / "mbpp_repair" / checkpoint_name
        payload = torch.load(source, map_location="cpu", weights_only=True)
        # This fixed native anchor is reconstructed from the frozen receiver.
        state = {
            key: value.detach().cpu().contiguous()
            for key, value in sorted(payload["encoder"].items())
            if key != "output_anchor"
        }
        if any("token_embedding" in key for key in state):
            raise RuntimeError(f"{source} unexpectedly contains receiver embedding weights")
        destination = adapter_dir / f"{name}.safetensors"
        save_file(
            state,
            destination,
            metadata={
                "base_model": "Qwen/Qwen2.5-Coder-3B-Instruct",
                "description": description,
                "source_checkpoint": checkpoint_name,
                "omitted_fixed_tensor": "output_anchor",
            },
        )
        shutil.copy2(ROOT / "artifacts" / "mbpp_repair" / result_name, result_dir / f"{name}.json")
        relative = destination.relative_to(output).as_posix()
        digest = sha256(destination)
        checksums.append(f"{digest}  {relative}")
        manifest["exports"][name] = {
            "description": description,
            "source_checkpoint": checkpoint_name,
            "result": f"results/{name}.json",
            "parameters": sum(tensor.numel() for tensor in state.values()),
            "tensor_count": len(state),
            "omitted_fixed_tensor": "output_anchor",
            "sha256": digest,
        }

    shutil.copy2(ROOT / "docs" / "huggingface_mbpp_model_card.md", output / "README.md")
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    total = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    print(f"Exported {len(EXPORTS)} MBPP adapters to {output}")
    print(f"Total bytes: {total}")


if __name__ == "__main__":
    main()
