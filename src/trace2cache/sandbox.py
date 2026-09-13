"""Conservative validation and subprocess execution for generated Python functions."""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
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


def evaluate_patch(source: str, case: RepairCase, timeout_seconds: float = 3.0) -> dict[str, Any]:
    validate_function(source, case.function_name)
    payload = {
        "source": source,
        "function_name": case.function_name,
        "tests": [
            {"args": test.args, "expected": test.expected}
            for test in (case.public_test, *case.hidden_tests)
        ],
    }
    result = subprocess.run(
        [sys.executable, "-I", str(Path(__file__).with_name("_sandbox_worker.py"))],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        env={"PYTHONPATH": ""},
    )
    if result.returncode != 0:
        return {"passed": False, "error": result.stderr.strip() or "worker failed", "tests": []}
    return json.loads(result.stdout)
