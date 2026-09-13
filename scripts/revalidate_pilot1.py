#!/usr/bin/env python3
"""Re-run held-out tests for saved generations without invoking the model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace2cache.benchmark import CASES
from trace2cache.sandbox import evaluate_patch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    cases = {case.case_id: case for case in CASES}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with Path(args.input).open(encoding="utf-8") as source, output.open("w", encoding="utf-8") as target:
        for line in source:
            row = json.loads(line)
            patch = row.get("patch")
            if patch:
                try:
                    row["validation"] = evaluate_patch(patch, cases[row["case_id"]])
                except Exception as exc:
                    row["validation"] = {
                        "passed": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "tests": [],
                    }
            target.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
