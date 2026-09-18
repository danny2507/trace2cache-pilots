#!/usr/bin/env python3
"""Re-score saved paired-runtime generations without decoding or modifying history.

The source artifacts remain untouched.  This script writes a separately versioned evaluator view
with response, source-row, test-suite, and evaluator-policy hashes so timeout and infrastructure
outcomes can be audited rather than merged into a generic failed repair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from trace2cache.ambiguous_repair import get_ambiguous_cases
from trace2cache.sandbox import EVALUATOR_REVISION, evaluate_patch, extract_function


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_hash(value: object) -> str:
    return _sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _case_hash(case: object) -> str:
    return _canonical_hash(asdict(case))


def _validate(response: str, case: object, policy: dict[str, float]) -> tuple[str | None, dict]:
    try:
        source = extract_function(response, case.function_name)
    except Exception as error:  # noqa: BLE001 - extraction is a measured outcome
        return None, {
            "passed": False,
            "outcome": "extraction_failure",
            "error": f"{type(error).__name__}: {error}",
            "tests": [],
            "evaluator_revision": EVALUATOR_REVISION,
        }
    return source, evaluate_patch(source, case, **policy)


def _source_paths(inputs: list[str]) -> list[Path]:
    if inputs:
        paths = [Path(item) for item in inputs]
    else:
        paths = sorted(Path("artifacts/paired_runtime_v2").glob("*/rows.jsonl"))
    if not paths:
        raise ValueError("no rows.jsonl artifacts found")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    return paths


def _summary(output_root: Path, paths: list[Path]) -> dict:
    runs = {}
    for source_path in paths:
        name = source_path.parent.name
        old_rows = [json.loads(line) for line in source_path.read_text().splitlines() if line]
        old_by_hash = {_canonical_hash(row): row for row in old_rows}
        output_path = output_root / name / "rows.jsonl"
        new_rows = [json.loads(line) for line in output_path.read_text().splitlines() if line]
        conditions = {}
        for condition in sorted({row.get("condition") for row in new_rows}):
            rows = [row for row in new_rows if row.get("condition") == condition]
            changed = [
                row
                for row in rows
                if bool(old_by_hash[row["original_row_sha256"]]["intended"]["passed"])
                != bool(row["intended"]["passed"])
                or bool(old_by_hash[row["original_row_sha256"]]["opposite"]["passed"])
                != bool(row["opposite"]["passed"])
            ]
            def outcomes(side: str) -> dict[str, int]:
                result: dict[str, int] = {}
                for row in rows:
                    outcome = row[side]["outcome"]
                    result[outcome] = result.get(outcome, 0) + 1
                return result
            conditions[condition] = {
                "views": len(rows),
                "intended_pass": sum(row["intended"]["passed"] for row in rows),
                "opposite_pass": sum(row["opposite"]["passed"] for row in rows),
                "intended_outcomes": outcomes("intended"),
                "opposite_outcomes": outcomes("opposite"),
                "changed_pass_fail_outcomes_vs_history": len(changed),
            }
        runs[name] = {"source_rows": len(old_rows), "revalidated_rows": len(new_rows), "conditions": conditions}
    return {"schema_version": 1, "evaluator_revision": EVALUATOR_REVISION, "runs": runs}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="*", help="explicit source rows.jsonl files; defaults to all paired-runtime artifacts")
    parser.add_argument("--output-dir", default="artifacts/paired_runtime_v2_revalidation_20260918")
    parser.add_argument("--startup-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--candidate-timeout-seconds", type=float, default=3.0)
    parser.add_argument("--resume", action="store_true", help="continue only when immutable manifest matches exactly")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.startup_timeout_seconds <= 0 or args.candidate_timeout_seconds <= 0:
        raise SystemExit("timeouts must be positive")
    paths = _source_paths(args.inputs or [])
    output_root = Path(args.output_dir)
    policy = {
        "startup_timeout_seconds": args.startup_timeout_seconds,
        "candidate_timeout_seconds": args.candidate_timeout_seconds,
    }
    manifest = {
        "schema_version": 1,
        "evaluator_revision": EVALUATOR_REVISION,
        "policy": policy,
        "sources": [
            {"path": str(path), "sha256": _sha256_bytes(path.read_bytes())}
            for path in paths
        ],
    }
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous != manifest:
            raise RuntimeError("existing revalidation manifest differs; choose a new output directory")
        if not args.resume:
            raise RuntimeError("revalidation already exists; use --resume or a new output directory")
    else:
        output_root.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    items = get_ambiguous_cases()
    total = 0
    for source_path in paths:
        run_name = source_path.parent.name
        output_path = output_root / run_name / "rows.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        existing: dict[str, dict] = {}
        if output_path.exists():
            existing = {
                row["original_row_sha256"]: row
                for row in (json.loads(line) for line in output_path.read_text().splitlines() if line)
            }
        rows = [json.loads(line) for line in source_path.read_text().splitlines() if line]
        with output_path.open("a") as handle:
            for row in rows:
                original_hash = _canonical_hash(row)
                if original_hash in existing:
                    continue
                if not isinstance(row.get("response"), str) or not isinstance(row.get("label"), int):
                    raise ValueError(f"{source_path}: missing response or integer label")
                label = row["label"]
                if label < 0 or label >= len(items):
                    raise ValueError(f"{source_path}: label out of range: {label}")
                opposite_label = label ^ 1
                source, intended = _validate(row["response"], items[label].case, policy)
                _, opposite = _validate(row["response"], items[opposite_label].case, policy)
                revalidated = {
                    "original_run": run_name,
                    "original_row_sha256": original_hash,
                    "response_sha256": _sha256_bytes(row["response"].encode()),
                    "pair_uid": row.get("pair_uid"),
                    "side": row.get("side"),
                    "condition": row.get("condition"),
                    "label": label,
                    "patch": source,
                    "patch_sha256": _sha256_bytes(source.encode()) if source is not None else None,
                    "intended_test_suite_sha256": _case_hash(items[label].case),
                    "opposite_test_suite_sha256": _case_hash(items[opposite_label].case),
                    "intended": intended,
                    "opposite": opposite,
                    "evaluator_revision": EVALUATOR_REVISION,
                    "policy": policy,
                }
                handle.write(json.dumps(revalidated, sort_keys=True) + "\n")
                handle.flush()
                existing[original_hash] = revalidated
                total += 1
                if total % 25 == 0:
                    print(json.dumps({"revalidated_rows": total, "run": run_name}), flush=True)
        expected_hashes = [_canonical_hash(row) for row in rows]
        missing = [key for key in expected_hashes if key not in existing]
        if missing:
            raise RuntimeError(f"{source_path}: missing {len(missing)} revalidated rows")
        # A tool interruption can leave append-only duplicates.  Canonicalize only our newly
        # created artifact, in immutable source-row order; raw historical artifacts are untouched.
        canonical_rows = [existing[key] for key in expected_hashes]
        temporary = output_path.with_suffix(".tmp")
        temporary.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in canonical_rows)
        )
        temporary.replace(output_path)
    summary = _summary(output_root, paths)
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"revalidated_rows": total, "output_dir": str(output_root), "manifest": manifest}, sort_keys=True))


if __name__ == "__main__":
    main()
