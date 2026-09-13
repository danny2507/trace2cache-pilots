#!/usr/bin/env python3
"""Print a compact, non-mutating CUDA/PyTorch compatibility report."""

from __future__ import annotations

import json
import platform
import subprocess
import sys


def command_output(*command: str) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc}"


def main() -> None:
    import torch
    import transformers

    report: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "nvcc": command_output("nvcc", "--version").splitlines()[-1],
        "nvidia_smi": command_output(
            "nvidia-smi",
            "--query-gpu=driver_version,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        report.update(
            {
                "gpu": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
