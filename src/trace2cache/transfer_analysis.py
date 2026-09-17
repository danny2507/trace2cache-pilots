"""Paired, program-clustered analysis of executed latent evidence interventions."""
from __future__ import annotations

import json
import random


def _metrics(groups):
    pairs = [pair for group in groups for pair in group]
    denominator = 2 * len(pairs)
    true = sum(row["intended"]["passed"] for pair in pairs for row in pair["true_latent"])
    swap = sum(row["intended"]["passed"] for pair in pairs for row in pair["paired_swap"])
    opposite = sum(row["opposite"]["passed"] for pair in pairs for row in pair["paired_swap"])
    both = sum(all(row["intended"]["passed"] for row in pair["true_latent"]) for pair in pairs)
    result = {"true_repair": true / denominator, "swap_intended_repair": swap / denominator,
            "true_minus_swap": (true - swap) / denominator,
            "swap_opposite_repair": opposite / denominator, "true_pair_success": both / len(pairs)}
    true_rows = [row for pair in pairs for row in pair["true_latent"]]
    if all("nearest_label" in row and "label" in row for row in true_rows):
        result["true_nearest_code_accuracy"] = sum(row["nearest_label"] == row["label"] for row in true_rows) / denominator
        result["correct_nearest_but_failed_repair"] = sum(row["nearest_label"] == row["label"] and not row["intended"]["passed"] for row in true_rows) / denominator
    return result


def analyze_interventions(rows, *, seed=401, resamples=2000):
    if resamples < 1:
        raise ValueError("resamples must be positive")
    families = {}
    for row in rows:
        if row["condition"] not in ("true_latent", "paired_swap"):
            continue
        pair = families.setdefault(row["family"], {}).setdefault(row["pair_uid"], {})
        sides = pair.setdefault(row["condition"], {})
        if row["side"] in sides:
            raise ValueError("duplicate intervention row")
        sides[row["side"]] = row
    if not families:
        raise ValueError("no paired intervention rows")
    groups = []
    per_family = {}
    for family, pairs in sorted(families.items()):
        group = []
        for conditions in pairs.values():
            if set(conditions) != {"true_latent", "paired_swap"} or any(set(sides) != {"a", "b"} for sides in conditions.values()):
                raise ValueError("incomplete matched pair")
            group.append({name: [sides[side] for side in ("a", "b")] for name, sides in conditions.items()})
        groups.append(group)
        per_family[family] = {"pairs": len(group), "views": 2 * len(group), **_metrics([group])}
    point = _metrics(groups)
    rng = random.Random(seed)
    samples = {key: [] for key in point}
    for _ in range(resamples):
        values = _metrics([rng.choice(groups) for _ in groups])
        for key, value in values.items():
            samples[key].append(value)
    intervals = {}
    for key, values in samples.items():
        values.sort()
        intervals[key] = [values[int(0.025 * (resamples - 1))], values[int(0.975 * (resamples - 1))]]
    gates = {"true_repair_at_least_75pct": point["true_repair"] >= 0.75,
             "pair_success_at_least_60pct": point["true_pair_success"] >= 0.60,
             "true_minus_swap_at_least_30pp": point["true_minus_swap"] >= 0.30,
             "swap_opposite_at_least_60pct": point["swap_opposite_repair"] >= 0.60}
    return {"program_clusters": len(groups), "pairs": sum(len(group) for group in groups),
            "metrics": point, "per_family": per_family, "cluster_bootstrap_95pct": intervals,
            "bootstrap_seed": seed, "bootstrap_resamples": resamples, "numerical_gates": gates,
            "all_numerical_gates_pass": all(gates.values()),
            "scope": "New input bundles for known toy programs; numerical gates alone do not establish baseline replication or unseen-program generalization."}


def audit_input_overlap(train, development):
    def tests(records):
        return {(record.buggy_source, json.dumps(meta["args"], sort_keys=True))
                for record in records for meta in record.tests_metadata}
    train_tests, dev_tests = tests(train), tests(development)
    repeated = train_tests & dev_tests
    dev_occurrences = [(record.buggy_source, json.dumps(meta["args"], sort_keys=True))
                       for record in development for meta in record.tests_metadata]
    fully_novel = fully_novel_pair_uids(train, development)
    return {"train_pairs": len(train), "development_pairs": len(development),
            "exact_bundle_hash_overlap": len({row.input_hash for row in train} & {row.input_hash for row in development}),
            "train_unique_individual_tests": len(train_tests), "development_unique_individual_tests": len(dev_tests),
            "individual_test_overlap": len(repeated), "development_test_occurrences": len(dev_occurrences),
            "development_test_occurrences_seen_in_train": sum(test in train_tests for test in dev_occurrences),
            "development_pairs_with_no_individual_test_overlap": len(fully_novel)}


def fully_novel_pair_uids(train, development):
    seen = {(record.buggy_source, json.dumps(meta["args"], sort_keys=True))
            for record in train for meta in record.tests_metadata}
    return {record.pair_uid for record in development
            if all((record.buggy_source, json.dumps(meta["args"], sort_keys=True)) not in seen
                   for meta in record.tests_metadata)}
