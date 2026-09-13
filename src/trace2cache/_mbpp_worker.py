"""Private subprocess worker for MBPP assertions."""

from __future__ import annotations

import json
import sys


def main() -> None:
    payload = json.loads(sys.stdin.read())
    namespace = {"__name__": "__mbpp__"}
    try:
        # Some MBPP setup snippets instantiate classes defined by the solution.
        exec(payload["source"], namespace)  # noqa: S102 -- purpose of isolated worker
        exec(payload.get("setup_source", ""), namespace)  # noqa: S102
    except Exception as exc:  # noqa: BLE001 -- report benchmark runtime errors
        print(json.dumps({"status": "load_error", "tests": [], "error": type(exc).__name__}))
        return

    outcomes = []
    for test in payload["tests"]:
        try:
            exec(test, namespace)  # noqa: S102 -- published benchmark assertions
            outcomes.append("pass")
        except AssertionError:
            outcomes.append("fail")
        except Exception as exc:  # noqa: BLE001 -- each runtime error is an outcome
            outcomes.append(f"error:{type(exc).__name__}")
    if outcomes and all(outcome == "pass" for outcome in outcomes):
        status = "all_pass"
    elif "pass" in outcomes:
        status = "mixed"
    else:
        status = "all_fail"
    print(json.dumps({"status": status, "tests": outcomes}))


if __name__ == "__main__":
    main()
