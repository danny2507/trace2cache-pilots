"""MBPP loading, deterministic mutation, and execution helpers.

The executor is intended only for the published MBPP benchmark and mutations of
its canonical solutions.  It is process-isolated, but it is not a security
sandbox for arbitrary untrusted Python.
"""

from __future__ import annotations

import ast
import copy
import json
import os
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


@dataclass(frozen=True)
class _ExpandedSite:
    category: str
    ordinal: int
    variant: str


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


def _expanded_category(node: ast.AST) -> str | None:
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        return "compare"
    if isinstance(node, ast.BinOp):
        return "binop"
    if isinstance(node, ast.AugAssign):
        return "augassign"
    if isinstance(node, ast.BoolOp):
        return "boolop"
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return "remove_not"
    if isinstance(node, (ast.If, ast.While)):
        return "negate_condition"
    if isinstance(node, (ast.Assign, ast.AugAssign)):
        return "delete_assignment"
    if isinstance(node, ast.Constant) and type(node.value) is bool:
        return "boolean"
    if isinstance(node, ast.Constant) and type(node.value) is int and -100 <= node.value <= 100:
        return "constant"
    return None


def _operator_name(operator: ast.AST) -> str:
    return type(operator).__name__


def _expanded_sites(tree: ast.AST) -> list[_ExpandedSite]:
    compare_ops = (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)
    arithmetic_ops = (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)
    counts: dict[str, int] = {}
    sites = []
    for node in ast.walk(tree):
        category = _expanded_category(node)
        if category is None:
            continue
        ordinal = counts.get(category, 0)
        counts[category] = ordinal + 1
        if category == "compare":
            variants = [
                _operator_name(operator())
                for operator in compare_ops
                if not isinstance(node.ops[0], operator)
            ]
        elif category in {"binop", "augassign"}:
            variants = [
                _operator_name(operator())
                for operator in arithmetic_ops
                if not isinstance(node.op, operator)
            ]
        elif category == "constant":
            variants = ("plus_one", "minus_one", "zero", "one")
        else:
            variants = ("toggle",)
        sites.extend(_ExpandedSite(category, ordinal, variant) for variant in variants)
    return sites


_OPERATOR_CLASSES = {
    operator.__name__: operator
    for operator in (
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.FloorDiv,
        ast.Mod,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.Eq,
        ast.NotEq,
    )
}


class _ExpandedMutation(ast.NodeTransformer):
    def __init__(self, site: _ExpandedSite) -> None:
        self.site = site
        self.counts: dict[str, int] = {}
        self.done = False

    def generic_visit(self, node: ast.AST):
        category = _expanded_category(node)
        if not self.done and category == self.site.category:
            ordinal = self.counts.get(category, 0)
            self.counts[category] = ordinal + 1
            if ordinal == self.site.ordinal:
                if category == "compare":
                    node.ops[0] = _OPERATOR_CLASSES[self.site.variant]()
                elif category in {"binop", "augassign"}:
                    node.op = _OPERATOR_CLASSES[self.site.variant]()
                elif category == "boolop":
                    node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
                elif category == "remove_not":
                    self.done = True
                    return self.visit(node.operand)
                elif category == "negate_condition":
                    node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
                elif category == "delete_assignment":
                    self.done = True
                    return ast.copy_location(ast.Pass(), node)
                elif category == "boolean":
                    node.value = not node.value
                elif self.site.variant == "plus_one":
                    node.value += 1
                elif self.site.variant == "minus_one":
                    node.value -= 1
                elif self.site.variant == "zero":
                    node.value = 0
                else:
                    node.value = 1
                self.done = True
        return super().generic_visit(node)


def generate_expanded_mutants(
    task: MBPPTask, *, limit: int = 32, seed: int | None = None
) -> tuple[MBPPMutation, ...]:
    """Generate a broader deterministic family of first-order semantic mutants."""
    tree = ast.parse(task.canonical_source)
    sites = _expanded_sites(tree)
    random.Random(task.task_id if seed is None else seed).shuffle(sites)
    mutations = []
    seen = set()
    for site in sites:
        transformer = _ExpandedMutation(site)
        mutated = transformer.visit(copy.deepcopy(tree))
        ast.fix_missing_locations(mutated)
        source = ast.unparse(mutated) + "\n"
        if transformer.done and source not in seen and source.strip() != task.canonical_source.strip():
            seen.add(source)
            mutations.append(
                MBPPMutation(f"{site.category}:{site.variant}", site.ordinal, source)
            )
        if len(mutations) >= limit:
            break
    return tuple(mutations)


def _run_mbpp_worker(
    task: MBPPTask,
    source: str | None,
    *,
    timeout_seconds: float,
    collect_trace: bool,
    max_events: int,
) -> dict:
    payload = {
        "source": task.canonical_source if source is None else source,
        "setup_source": task.setup_source,
        "tests": task.tests,
        "collect_trace": collect_trace,
        "max_events": max_events,
    }
    try:
        result = subprocess.run(
            [sys.executable, "-I", str(Path(__file__).with_name("_mbpp_worker.py"))],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
            env={**os.environ, "PYTHONHASHSEED": "0"},
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "tests": []}
    if result.returncode != 0:
        return {
            "status": "worker_error",
            "tests": [],
            "error": result.stderr.strip() or result.stdout.strip(),
        }
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {
            "status": "worker_error",
            "tests": [],
            "error": "invalid_worker_json",
        }


def run_mbpp_tests(
    task: MBPPTask, source: str | None = None, *, timeout_seconds: float = 2.0
) -> dict:
    """Execute each MBPP assertion and report per-test outcomes in a fresh process."""
    return _run_mbpp_worker(
        task,
        source,
        timeout_seconds=timeout_seconds,
        collect_trace=False,
        max_events=512,
    )


def trace_mbpp_tests(
    task: MBPPTask,
    source: str | None = None,
    *,
    timeout_seconds: float = 3.0,
    max_events: int = 512,
) -> dict:
    """Execute MBPP assertions and retain line-level program events for each test."""
    return _run_mbpp_worker(
        task,
        source,
        timeout_seconds=timeout_seconds,
        collect_trace=True,
        max_events=max_events,
    )


def run_mbpp_calls(
    task: MBPPTask,
    calls: list[str],
    source: str | None = None,
    *,
    timeout_seconds: float = 3.0,
    collect_trace: bool = True,
    max_events: int = 512,
) -> dict:
    """Evaluate literal function-call expressions and optionally trace their execution."""
    payload = {
        "source": task.canonical_source if source is None else source,
        "setup_source": task.setup_source,
        "calls": calls,
        "collect_trace": collect_trace,
        "max_events": max_events,
    }
    try:
        result = subprocess.run(
            [sys.executable, "-I", str(Path(__file__).with_name("_mbpp_worker.py"))],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
            env={**os.environ, "PYTHONHASHSEED": "0"},
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "calls": []}
    if result.returncode != 0:
        return {
            "status": "worker_error",
            "calls": [],
            "error": result.stderr.strip() or result.stdout.strip(),
        }
    return json.loads(result.stdout)
