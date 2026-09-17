#!/usr/bin/env python3
"""Audit saved paired-runtime evidence, including cross-split input disjointness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace2cache.paired_evidence import audit_records, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("dataset_dir")
    args = parser.parse_args(); root = Path(args.dataset_dir)
    by_split = {path.stem: read_jsonl(path) for path in sorted(root.glob("*.jsonl"))}
    all_records = [record for records in by_split.values() for record in records]
    report = audit_records(all_records)
    split_hashes = {name: {record.input_hash for record in records} for name, records in by_split.items()}
    overlaps = {f"{left}:{right}": len(split_hashes[left] & split_hashes[right]) for left in split_hashes for right in split_hashes if left < right}
    report["splits"] = {name: len(rows) for name, rows in by_split.items()}; report["cross_split_overlaps"] = overlaps
    report["passed"] = report["passed"] and not any(overlaps.values())
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
