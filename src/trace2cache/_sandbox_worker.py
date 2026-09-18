"""Isolated patch worker; intentionally has no imports from the trace2cache package."""

from __future__ import annotations

import builtins
import json
import math
import resource
import signal
import sys
import time


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


class CandidateTimeout(Exception):
    pass


def _timeout_handler(_signum, _frame) -> None:
    raise CandidateTimeout("candidate exceeded resource budget")


def limit_resources(cpu_seconds: int) -> None:
    # A soft CPU limit generates SIGXCPU, which is converted to a classified result below.
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
    memory = 1 * 2**30
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (16, 16))


def main() -> None:
    print(json.dumps({"event": "ready"}), flush=True)
    payload = json.loads(sys.stdin.readline())
    limit_resources(int(payload["candidate_cpu_seconds"]))
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.signal(signal.SIGXCPU, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, float(payload["candidate_wall_seconds"]))
    started = time.perf_counter()
    namespace = {"__builtins__": {name: getattr(builtins, name) for name in ALLOWED_CALLS}}
    outcomes = []
    try:
        exec(compile(payload["source"], "<candidate-patch>", "exec"), namespace)
        function = namespace[payload["function_name"]]
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
            except CandidateTimeout:
                raise
            except Exception as exc:
                outcomes.append({"passed": False, "error": f"{type(exc).__name__}: {exc}"[:200]})
        print(json.dumps({"passed": all(item["passed"] for item in outcomes), "outcome": "semantic_pass" if all(item["passed"] for item in outcomes) else "semantic_failure", "tests": outcomes, "candidate_elapsed_seconds": time.perf_counter() - started}))
    except CandidateTimeout as error:
        print(json.dumps({"passed": False, "outcome": "candidate_timeout", "error": str(error), "tests": outcomes, "candidate_elapsed_seconds": time.perf_counter() - started}))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    main()
