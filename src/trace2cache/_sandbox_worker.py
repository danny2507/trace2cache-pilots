"""Isolated patch worker; intentionally has no imports from the trace2cache package."""

from __future__ import annotations

import builtins
import json
import math
import resource
import sys


ALLOWED_CALLS = {
    "ValueError",
    "abs",
    "all",
    "any",
    "enumerate",
    "len",
    "list",
    "max",
    "min",
    "range",
    "reversed",
    "round",
    "sorted",
    "sum",
    "zip",
}


def limit_resources() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    memory = 1 * 2**30
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (16, 16))


def main() -> None:
    limit_resources()
    payload = json.loads(sys.stdin.read())
    namespace = {"__builtins__": {name: getattr(builtins, name) for name in ALLOWED_CALLS}}
    exec(compile(payload["source"], "<candidate-patch>", "exec"), namespace)
    function = namespace[payload["function_name"]]
    outcomes = []
    for test in payload["tests"]:
        try:
            actual = function(*test["args"])
            expected = test["expected"]
            equal = (
                math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)
                if isinstance(actual, float) and isinstance(expected, (float, int))
                else actual == expected
            )
            outcomes.append({"passed": equal, "actual": repr(actual)[:200]})
        except Exception as exc:
            outcomes.append({"passed": False, "error": f"{type(exc).__name__}: {exc}"[:200]})
    print(json.dumps({"passed": all(item["passed"] for item in outcomes), "tests": outcomes}))


if __name__ == "__main__":
    main()
