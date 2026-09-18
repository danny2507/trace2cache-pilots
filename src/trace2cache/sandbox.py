"""Conservative validation and subprocess execution for generated Python functions."""

from __future__ import annotations

import ast
import json
import re
import select
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .benchmark import RepairCase


ALLOWED_CALLS = {"ValueError", "abs", "all", "any", "enumerate", "len", "list", "max", "min", "range", "reversed", "round", "sorted", "sum", "zip"}
FORBIDDEN_NODES = (
    ast.AsyncFunctionDef,
    ast.Await,
    ast.ClassDef,
    ast.Delete,
    ast.Global,
    ast.Import,
    ast.ImportFrom,
    ast.Lambda,
    ast.Nonlocal,
    ast.Try,
    ast.With,
)
EVALUATOR_REVISION = "sandbox-v2-startup-and-candidate-budget"


def extract_function(text: str, function_name: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates = blocks or [text]
    for candidate in candidates:
        candidate = candidate.strip()
        try:
            tree = ast.parse(candidate)
        except SyntaxError:
            continue
        if any(isinstance(node, ast.FunctionDef) and node.name == function_name for node in tree.body):
            return candidate + "\n"
    raise ValueError(f"no parseable function named {function_name}")


def validate_function(source: str, function_name: str) -> None:
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(tree.body) != 1 or len(functions) != 1 or functions[0].name != function_name:
        raise ValueError("patch must contain exactly the requested top-level function")
    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN_NODES):
            raise ValueError(f"forbidden syntax: {type(node).__name__}")
        if isinstance(node, ast.Attribute):
            raise ValueError("attribute access is forbidden in pilot patches")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("only direct calls to allowlisted functions are permitted")
            if node.func.id not in ALLOWED_CALLS | {function_name}:
                raise ValueError(f"call is not allowlisted: {node.func.id}")


def _failure(category: str, error: str, *, tests: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "passed": False,
        "outcome": category,
        "error": error,
        "tests": [] if tests is None else tests,
        "evaluator_revision": EVALUATOR_REVISION,
    }


def _read_ready_line(worker: subprocess.Popen[str], timeout_seconds: float) -> str | None:
    """Read the worker handshake without charging interpreter startup to candidate execution."""
    if worker.stdout is None:
        raise RuntimeError("worker stdout pipe is unavailable")
    ready, _, _ = select.select([worker.stdout], [], [], timeout_seconds)
    return worker.stdout.readline() if ready else None


def evaluate_patch(
    source: str,
    case: RepairCase,
    *,
    startup_timeout_seconds: float = 5.0,
    candidate_timeout_seconds: float = 3.0,
) -> dict[str, Any]:
    """Validate and execute a patch under separate startup and candidate budgets.

    The worker acknowledges startup before it receives candidate code.  A startup fault is therefore
    never silently reported as a bad patch, while a wall/CPU limit after that acknowledgement is
    retained as a candidate timeout.  The limits remain in the worker; this is not a sandbox for
    arbitrary untrusted Python.
    """
    try:
        validate_function(source, case.function_name)
    except Exception as error:  # policy is an evaluation result, not an evaluator crash
        return _failure("policy_rejection", f"{type(error).__name__}: {error}")
    if startup_timeout_seconds <= 0 or candidate_timeout_seconds <= 0:
        raise ValueError("startup and candidate timeouts must be positive")
    payload = {
        "source": source,
        "function_name": case.function_name,
        "tests": [
            {"args": test.args, "expected": test.expected}
            for test in (case.public_test, *case.hidden_tests)
        ],
        "candidate_wall_seconds": candidate_timeout_seconds,
        "candidate_cpu_seconds": max(1, int(candidate_timeout_seconds)),
    }
    worker = subprocess.Popen(
        [sys.executable, "-I", str(Path(__file__).with_name("_sandbox_worker.py"))],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={"PYTHONPATH": ""},
    )
    started = time.perf_counter()
    try:
        ready_line = _read_ready_line(worker, startup_timeout_seconds)
    except Exception as error:  # noqa: BLE001 - report evaluator/worker failures explicitly
        worker.kill()
        worker.communicate()
        return _failure("infrastructure_failure", f"startup handshake error: {type(error).__name__}: {error}")
    if ready_line is None:
        worker.kill()
        _, stderr = worker.communicate()
        return _failure("infrastructure_failure", f"worker startup timeout after {startup_timeout_seconds:.3f}s: {stderr.strip()}")
    try:
        ready = json.loads(ready_line)
    except json.JSONDecodeError:
        worker.kill()
        _, stderr = worker.communicate()
        return _failure("infrastructure_failure", f"invalid worker handshake: {ready_line.strip()} {stderr.strip()}")
    if ready != {"event": "ready"}:
        worker.kill()
        _, stderr = worker.communicate()
        return _failure("infrastructure_failure", f"unexpected worker handshake: {ready!r} {stderr.strip()}")
    try:
        if worker.stdin is None:
            raise RuntimeError("worker stdin pipe is unavailable")
        worker.stdin.write(json.dumps(payload) + "\n")
        worker.stdin.close()
        worker.stdin = None
        stdout, stderr = worker.communicate(timeout=candidate_timeout_seconds + 1.0)
    except subprocess.TimeoutExpired:
        worker.kill()
        _, stderr = worker.communicate()
        return _failure("candidate_wall_timeout", f"candidate exceeded {candidate_timeout_seconds:.3f}s wall budget after startup: {stderr.strip()}")
    except Exception as error:  # noqa: BLE001
        worker.kill()
        worker.communicate()
        return _failure("infrastructure_failure", f"worker I/O error: {type(error).__name__}: {error}")
    elapsed = time.perf_counter() - started
    if worker.returncode != 0:
        return _failure("infrastructure_failure", f"worker exit {worker.returncode}: {stderr.strip()}")
    try:
        result = json.loads(stdout)
    except json.JSONDecodeError:
        return _failure("infrastructure_failure", f"invalid worker result: {stdout.strip()} {stderr.strip()}")
    if not isinstance(result, dict) or "passed" not in result or "outcome" not in result:
        return _failure("infrastructure_failure", f"malformed worker result: {result!r}")
    result["evaluator_revision"] = EVALUATOR_REVISION
    result["worker_elapsed_seconds"] = elapsed
    return result
