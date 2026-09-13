#!/usr/bin/env python3
"""Summarize Pilot 1 JSONL without third-party dependencies."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/pilot1/results.jsonl")
    groups: dict[str, list[dict]] = defaultdict(list)
    for line in path.read_text().splitlines():
        row = json.loads(line)
        groups[row["condition"]].append(row)
    print("condition\tn\trepair@1\tavg_input_tokens\tavg_seconds")
    for condition, rows in sorted(groups.items()):
        n = len(rows)
        accuracy = sum(bool(row["validation"]["passed"]) for row in rows) / n
        tokens = sum(row["input_tokens"] for row in rows) / n
        seconds = sum(row["generation_seconds"] for row in rows) / n
        print(f"{condition}\t{n}\t{accuracy:.3f}\t{tokens:.1f}\t{seconds:.2f}")


if __name__ == "__main__":
    main()

