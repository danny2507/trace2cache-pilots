#!/usr/bin/env python3
"""Paired task-level comparison for the MBPP generalization panel."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default="artifacts/mbpp_generalization/full_seed1001/rows.jsonl")
    parser.add_argument("--summary", default="artifacts/mbpp_generalization/full_seed1001/summary.json")
    return parser.parse_args()


def mcnemar(wins: int, losses: int) -> float:
    """Two-sided exact binomial McNemar p-value for discordant pairs."""
    n = wins + losses
    if n == 0:
        return 1.0
    # P(X <= min) * 2, with X ~ Binomial(n, 0.5), capped at 1.
    k = min(wins, losses)
    p = 0.0
    coeff = 1.0
    for i in range(k + 1):
        if i:
            coeff *= (n - i + 1) / i
        p += coeff
    p *= 0.5 ** n
    return min(1.0, 2.0 * p)


def main() -> None:
    args = parse_args()
    rows = [json.loads(line) for line in Path(args.rows).read_text().splitlines() if line.strip()]
    by_task: dict[int, dict[str, bool]] = defaultdict(dict)
    for row in rows:
        by_task[row["task_id"]][row["condition"]] = bool(row["passed"])
    conditions = sorted({row["condition"] for row in rows})
    print(json.dumps({"tasks": len(by_task), "conditions": conditions}, sort_keys=True))
    if Path(args.summary).exists():
        payload = json.loads(Path(args.summary).read_text())
        print(json.dumps({"stored_summary": payload.get("summary", {})}, sort_keys=True))
    pairs = [
        ("true_latent", "no_evidence"),
        ("true_latent", "io_text"),
        ("true_latent", "compact_text"),
        ("true_latent", "shuffled_latent"),
        ("true_latent", "corrupted_latent"),
        ("compact_text", "no_evidence"),
        ("compact_text", "io_text"),
        ("gist_text", "no_evidence"),
        ("gist_text", "io_text"),
        ("gist_text", "compact_text"),
        ("collated_text", "no_evidence"),
        ("collated_text", "io_text"),
        ("collated_text", "compact_text"),
        ("gist_text", "collated_text"),
        ("line_text", "no_evidence"),
        ("line_text", "io_text"),
        ("line_text", "gist_text"),
        ("line_text", "collated_text"),
        ("discrepancy_text", "no_evidence"),
        ("discrepancy_text", "io_text"),
        ("discrepancy_text", "gist_text"),
        ("discrepancy_embed", "no_evidence"),
        ("discrepancy_embed", "discrepancy_text"),
        ("discrepancy_embed", "gist_text"),
        ("discrepancy_embed", "shuffled_embed"),
        ("discrepancy_embed", "corrupted_embed"),
    ]
    for left, right in pairs:
        if left not in conditions or right not in conditions:
            continue
        both = win = lose = 0
        for outcomes in by_task.values():
            if left not in outcomes or right not in outcomes:
                continue
            a, b = outcomes[left], outcomes[right]
            both += 1
            win += int(a and not b)
            lose += int(b and not a)
        print(
            json.dumps(
                {
                    "comparison": f"{left} vs {right}",
                    "tasks": both,
                    "left_only": win,
                    "right_only": lose,
                    "mcnemar_p": round(mcnemar(win, lose), 4),
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
