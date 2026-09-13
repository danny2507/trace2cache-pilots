"""MBPP loading, deterministic mutation, and execution helpers.

The executor is intended only for the published MBPP benchmark and mutations of
its canonical solutions.  It is process-isolated, but it is not a security
sandbox for arbitrary untrusted Python.
"""

from __future__ import annotations

import ast
import copy
import json
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Split = Literal["prompt", "test", "validation", "train"]


@dataclass(frozen=True)
class MBPPTask:
    task_id: int
    description: str
    canonical_source: str
    tests: tuple[str, ...]
    setup_source: str = ""
    challenge_tests: tuple[str, ...] = ()

    @property
    def split(self) -> Split:
        return split_for_task_id(self.task_id)


@dataclass(frozen=True)
class MBPPMutation:
    kind: str
    ordinal: int
    source: str


def split_for_task_id(task_id: int) -> Split:
    """Return the split specified by the original Google MBPP release."""
    if 1 <= task_id <= 10:
        return "prompt"
    if 11 <= task_id <= 510:
        return "test"
    if 511 <= task_id <= 600:
        return "validation"
    if 601 <= task_id <= 974:
        return "train"
    raise ValueError(f"task id is outside the official MBPP range: {task_id}")


def load_mbpp(path: str | Path, *, split: Split | None = None) -> tuple[MBPPTask, ...]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    tasks = tuple(
        MBPPTask(
            task_id=row["task_id"],
            description=row["text"],
            canonical_source=row["code"],
            tests=tuple(row["test_list"]),
            setup_source=row.get("test_setup_code", ""),
            challenge_tests=tuple(row.get("challenge_test_list", ())),
        )
        for row in rows
    )
    return tasks if split is None else tuple(task for task in tasks if task.split == split)


_COMPARE_SWAPS = {
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.Is: ast.IsNot,
    ast.IsNot: ast.Is,
    ast.In: ast.NotIn,
    ast.NotIn: ast.In,
}
_BINOP_SWAPS = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.Add,
    ast.FloorDiv: ast.Mult,
    ast.Mod: ast.FloorDiv,
}


def _eligible_kind(node: ast.AST) -> str | None:
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _COMPARE_SWAPS:
        return "compare"
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOP_SWAPS:
        return "binop"
    if isinstance(node, ast.BoolOp):
        return "boolop"
    if isinstance(node, ast.Constant) and type(node.value) is int and -100 <= node.value <= 100:
        return "constant"
    return None


def _mutation_sites(tree: ast.AST) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    sites = []
    for node in ast.walk(tree):
        kind = _eligible_kind(node)
        if kind is None:
            continue
        ordinal = counts.get(kind, 0)
        counts[kind] = ordinal + 1
        if kind == "constant":
            sites.extend((("constant+1", ordinal), ("constant-1", ordinal)))
        else:
            sites.append((kind, ordinal))
    return sites


class _SingleMutation(ast.NodeTransformer):
    def __init__(self, kind: str, ordinal: int) -> None:
        self.kind = kind
        self.ordinal = ordinal
        self.counts: dict[str, int] = {}
        self.done = False

    def generic_visit(self, node: ast.AST):
        base_kind = "constant" if self.kind.startswith("constant") else self.kind
        node_kind = _eligible_kind(node)
        if not self.done and node_kind == base_kind:
            ordinal = self.counts.get(base_kind, 0)
            self.counts[base_kind] = ordinal + 1
            if ordinal == self.ordinal:
                if base_kind == "compare":
                    node.ops[0] = _COMPARE_SWAPS[type(node.ops[0])]()
                elif base_kind == "binop":
                    node.op = _BINOP_SWAPS[type(node.op)]()
                elif base_kind == "boolop":
                    node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
                else:
                    node.value += 1 if self.kind.endswith("+1") else -1
                self.done = True
        return super().generic_visit(node)


def generate_mutants(
    task: MBPPTask, *, limit: int = 12, seed: int | None = None
) -> tuple[MBPPMutation, ...]:
    """Generate reproducible first-order mutants without looking at test outcomes."""
    tree = ast.parse(task.canonical_source)
    sites = _mutation_sites(tree)
    random.Random(task.task_id if seed is None else seed).shuffle(sites)
    mutations = []
    seen = set()
    for kind, ordinal in sites:
        transformer = _SingleMutation(kind, ordinal)
        mutated = transformer.visit(copy.deepcopy(tree))
        ast.fix_missing_locations(mutated)
        source = ast.unparse(mutated) + "\n"
        if transformer.done and source not in seen and source.strip() != task.canonical_source.strip():
            seen.add(source)
            mutations.append(MBPPMutation(kind, ordinal, source))
        if len(mutations) >= limit:
            break
    return tuple(mutations)


def run_mbpp_tests(
    task: MBPPTask, source: str | None = None, *, timeout_seconds: float = 2.0
) -> dict:
    """Execute each MBPP assertion and report per-test outcomes in a fresh process."""
    payload = {
        "source": task.canonical_source if source is None else source,
        "setup_source": task.setup_source,
        "tests": task.tests,
    }
    try:
        result = subprocess.run(
            [sys.executable, "-I", str(Path(__file__).with_name("_mbpp_worker.py"))],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "tests": []}
    if result.returncode != 0:
        return {
            "status": "worker_error",
            "tests": [],
            "error": result.stderr.strip() or result.stdout.strip(),
        }
    return json.loads(result.stdout)
