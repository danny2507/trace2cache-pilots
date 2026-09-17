#!/usr/bin/env python3
"""Write paired repair analysis and train/development input-overlap diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path

from trace2cache.paired_evidence import read_jsonl
from trace2cache.transfer_analysis import analyze_interventions, audit_input_overlap, fully_novel_pair_uids

parser = argparse.ArgumentParser()
parser.add_argument("--evaluation-dir", required=True)
parser.add_argument("--train-dataset", required=True)
parser.add_argument("--development-dataset", required=True)
parser.add_argument("--reference-dir", help="optional earlier encoder evaluated on a subset of the same dev panel")
parser.add_argument("--seed", type=int, default=401)
args = parser.parse_args()
directory = Path(args.evaluation_dir)
rows = [json.loads(line) for line in (directory / "rows.jsonl").read_text().splitlines() if line]
summary = json.loads((directory / "summary.json").read_text())
if any(row["checkpoint_sha256"] != summary["checkpoint_sha256"] or row["dataset_sha256"] != summary["dataset_sha256"] for row in rows):
    raise ValueError("evaluation hashes disagree")
if hashlib.sha256(Path(args.development_dataset).read_bytes()).hexdigest() != summary["dataset_sha256"]:
    raise ValueError("overlap audit development dataset does not match evaluation")
result = analyze_interventions(rows, seed=args.seed)
result["checkpoint_sha256"] = summary["checkpoint_sha256"]
result["dataset_sha256"] = summary["dataset_sha256"]
train = read_jsonl(args.train_dataset)
development = read_jsonl(args.development_dataset)
result["train_dataset_sha256"] = hashlib.sha256(Path(args.train_dataset).read_bytes()).hexdigest()
result["input_overlap"] = audit_input_overlap(train, development)
novel = fully_novel_pair_uids(train, development)
novel_rows = [row for row in rows if row["pair_uid"] in novel]
result["fully_novel_inputs_subset"] = analyze_interventions(novel_rows, seed=args.seed) if novel_rows else None
result["summary"] = summary["summary"]
if args.reference_dir:
    reference_directory = Path(args.reference_dir)
    reference_summary = json.loads((reference_directory / "summary.json").read_text())
    if reference_summary["dataset_sha256"] != summary["dataset_sha256"]:
        raise ValueError("reference uses a different development dataset")
    for field in ("heldout_wording", "max_new_tokens"):
        if reference_summary["args"][field] != summary["args"][field]:
            raise ValueError("reference generation config differs")
    reference_rows = [json.loads(line) for line in (reference_directory / "rows.jsonl").read_text().splitlines() if line]
    panel = {row["pair_uid"] for row in reference_rows}
    current_rows = [row for row in rows if row["pair_uid"] in panel]
    if {row["pair_uid"] for row in current_rows} != panel:
        raise ValueError("current run is missing reference panel pairs")
    old = analyze_interventions(reference_rows, seed=args.seed)
    new = analyze_interventions(current_rows, seed=args.seed)
    result["matched_reference_panel"] = {"reference_checkpoint_sha256": reference_summary["checkpoint_sha256"],
        "reference": old, "current": new,
        "metric_differences": {key: new["metrics"][key] - value for key, value in old["metrics"].items()},
        "note": "Different training-set sizes at equal updates; one seed, not matched-baseline replication."}
(directory / "analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, indent=2, sort_keys=True))
