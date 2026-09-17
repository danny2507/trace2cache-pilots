#!/usr/bin/env python3
"""Write paired repair analysis and train/development input-overlap diagnostics."""
import argparse
import json
from pathlib import Path

from trace2cache.paired_evidence import read_jsonl
from trace2cache.transfer_analysis import analyze_interventions, audit_input_overlap

parser = argparse.ArgumentParser()
parser.add_argument("--evaluation-dir", required=True)
parser.add_argument("--train-dataset", required=True)
parser.add_argument("--development-dataset", required=True)
parser.add_argument("--seed", type=int, default=401)
args = parser.parse_args()
directory = Path(args.evaluation_dir)
rows = [json.loads(line) for line in (directory / "rows.jsonl").read_text().splitlines() if line]
summary = json.loads((directory / "summary.json").read_text())
if any(row["checkpoint_sha256"] != summary["checkpoint_sha256"] or row["dataset_sha256"] != summary["dataset_sha256"] for row in rows):
    raise ValueError("evaluation hashes disagree")
result = analyze_interventions(rows, seed=args.seed)
result["checkpoint_sha256"] = summary["checkpoint_sha256"]
result["dataset_sha256"] = summary["dataset_sha256"]
result["input_overlap"] = audit_input_overlap(read_jsonl(args.train_dataset), read_jsonl(args.development_dataset))
result["summary"] = summary["summary"]
(directory / "analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, indent=2, sort_keys=True))
