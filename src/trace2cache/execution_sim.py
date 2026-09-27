"""Output-prediction Gate 0: code+input versus intermediate execution text.

The decoder sees gold source and stdin and must emit stdout. Intermediate
LINE/STATE/CALL/BRANCH may appear as a chain-of-execution. TEST/PASS/FAIL,
RETURN, and the gold stdout itself never belong in the prompt. This is not
repair: the program in the prompt is the program to run.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .mbpp import MBPPMutation, MBPPTask
from .mbpp_generalization import (
    BRANCH,
    CALL,
    LINE,
    MARKER,
    PASS,
    ROLE_NAMES,
    STATE,
    EvidenceEvent,
    RepairExample,
    _clip_gist_line,
    build_events,
    compact_trace_text,
    encode_stdio_test,
    parse_stdio_test,
)

RUNTIME_KEEP = frozenset((CALL, BRANCH, STATE, LINE))
SIM_SLOT_ROLES = tuple(tuple(sorted(RUNTIME_KEEP)) for _ in range(8))
TEXT_CONDITIONS = ("code_input", "trace_gist", "trace_text")
LATENT_CONDITIONS = ("true_latent", "shuffled_latent")
NATIVE_CONDITIONS = ("native_events", "shuffled_native")
SNAP_CONDITIONS = ("vocab_snap", "shuffled_snap")
SIM_CONDITIONS = TEXT_CONDITIONS + LATENT_CONDITIONS + NATIVE_CONDITIONS + SNAP_CONDITIONS
NATIVE_CAP = 24
ASK_EVENTS = "ask_events"
ASK_EVENT_ROLES = ("CALL", "BRANCH", "LINE", "STATE")
SFT_EVENTS = "sft_events"
SFT_STDOUT = "sft_stdout"
REDACT = "[REDACTED]"
_EVENT_ROLE_LINE = re.compile(r"^(CALL|BRANCH|LINE|STATE):", re.MULTILINE)
_FENCE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\s*\n(.*?)```", re.DOTALL)
_EVENT_CONTENT = re.compile(
    r"^(?P<kind>call|line|return|exception) (?P<function>\S+) line (?P<line>\d+) "
    r"source (?P<source>.*?) state (?P<state>.*?)(?: return=(?P<ret>.*))?$"
)
_COMPACT_WIDTH = 180


@dataclass(frozen=True)
class SimExample:
    example_id: str
    task_id: int
    source: str
    stdin: str
    gold_stdout: str
    events: tuple[EvidenceEvent, ...]
    event_count: int
    source_hash: str


def normalize_output(text: str) -> str:
    return str(text).replace("\r\n", "\n").rstrip("\n")


def outputs_match(predicted: str, gold: str) -> bool:
    return normalize_output(predicted) == normalize_output(gold)


def redact_answer(text: str, gold: str) -> str:
    """Remove the gold stdout (and its long lines) from a trace string."""
    gold_n = normalize_output(gold)
    if not gold_n:
        return text
    needles = [gold_n]
    needles.extend(line for line in gold_n.split("\n") if len(line) >= 2)
    result = text
    for needle in sorted(set(needles), key=len, reverse=True):
        if needle in result:
            result = result.replace(needle, REDACT)
    return result


def filter_intermediate(
    events: tuple[EvidenceEvent, ...], gold: str
) -> tuple[EvidenceEvent, ...]:
    """Keep runtime events only, with gold stdout redacted from their text."""
    kept: list[EvidenceEvent] = []
    for event in events:
        if event.role_id not in RUNTIME_KEEP:
            continue
        kept.append(
            EvidenceEvent(event.test_id, event.role_id, redact_answer(event.content, gold))
        )
    return tuple(kept)


def gist_intermediate(events: tuple[EvidenceEvent, ...]) -> tuple[EvidenceEvent, ...]:
    last: dict[int, EvidenceEvent] = {}
    for event in events:
        if event.role_id in RUNTIME_KEEP:
            last[event.role_id] = event
    return tuple(last[role] for role in (CALL, BRANCH, LINE, STATE) if role in last)


def native_event_texts(
    events: tuple[EvidenceEvent, ...], *, cap: int = NATIVE_CAP
) -> tuple[str, ...]:
    """One compact-trace line per runtime event, in order. No TEST/END_TEST wrappers."""
    if cap <= 0:
        raise ValueError("cap must be positive")
    return tuple(
        f"{ROLE_NAMES[event.role_id]}: {event.content}" for event in events[:cap]
    )


def snap_table_to_vocab(table, embed_weight, batch_size: int = 256):
    """Replace each row with the cosine-nearest frozen embedding-table row.

    The spliced vector is the raw vocab row, not a unit vector. `table` and
    `embed_weight` are torch tensors [N, D] and [V, D].
    """
    import torch
    import torch.nn.functional as F

    if table.ndim != 2 or embed_weight.ndim != 2:
        raise ValueError("table and embed_weight must be rank-2")
    if table.shape[-1] != embed_weight.shape[-1]:
        raise ValueError("table and embed_weight width must match")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    vectors = table.float()
    vocab = embed_weight.float()
    vectors_n = F.normalize(vectors, dim=-1)
    vocab_n = F.normalize(vocab, dim=-1)
    indices = []
    cosines = []
    for start in range(0, vectors_n.shape[0], batch_size):
        similarity = vectors_n[start : start + batch_size] @ vocab_n.T
        values, index = similarity.max(dim=-1)
        indices.append(index)
        cosines.append(values)
    index = torch.cat(indices)
    cosine = torch.cat(cosines)
    snapped = vocab[index].to(dtype=table.dtype)
    return snapped, index, cosine


def evidence_for_condition(example: SimExample, condition: str) -> str:
    if condition == "code_input":
        return ""
    if (
        condition in LATENT_CONDITIONS
        or condition in NATIVE_CONDITIONS
        or condition in SNAP_CONDITIONS
    ):
        return MARKER
    if condition == "trace_gist":
        chosen = gist_intermediate(example.events)
        if not chosen:
            return "unavailable"
        return "\n".join(
            f"{ROLE_NAMES[event.role_id]}: {_clip_gist_line(event.content)}" for event in chosen
        )
    if condition == "trace_text":
        if not example.events:
            return "unavailable"
        return compact_trace_text(example.events)
    raise ValueError(f"unknown condition: {condition}")


def render_user_prompt(example: SimExample, evidence: str) -> str:
    evidence_block = ""
    if evidence == MARKER:
        evidence_block = f"\nPartial execution (stdout redacted):\n{MARKER}\n"
    elif evidence.strip() and evidence.strip() != "unavailable":
        evidence_block = f"\nPartial execution (stdout redacted):\n```\n{evidence.rstrip()}\n```\n"
    stdin = example.stdin.rstrip("\n")
    return f"""Predict the exact stdout of this Python program on the given input. Do not explain. Reply with only the stdout inside a markdown code block.

Program:
```python
{example.source.rstrip()}
```

Input:
```
{stdin}
```
{evidence_block}
What is the exact stdout?
"""


def render_chat(tokenizer, example: SimExample, evidence: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": "You are a precise Python interpreter. Emit only the program's stdout.",
            },
            {"role": "user", "content": render_user_prompt(example, evidence)},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def render_ask_events_prompt(example: SimExample, *, cap: int = NATIVE_CAP) -> str:
    """Code+stdin; the model must invent compact events, then stdout. No gold trace."""
    if cap <= 0:
        raise ValueError("cap must be positive")
    stdin = example.stdin.rstrip("\n")
    roles = "\n".join(f"{role}: ..." for role in ASK_EVENT_ROLES)
    return f"""Predict the exact stdout of this Python program on the given input.

First write a compact execution trace, one event per line, using only these roles:
{roles}
Write at most {cap} events. Do not write TEST, PASS, FAIL, RETURN, or END_TEST. Do not put the final stdout in the trace.

Then write the exact stdout inside one markdown code block. No other explanation.

Program:
```python
{example.source.rstrip()}
```

Input:
```
{stdin}
```
"""


def render_ask_events_chat(tokenizer, example: SimExample, *, cap: int = NATIVE_CAP) -> str:
    return tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": (
                    "You are a precise Python interpreter. First emit a compact "
                    "execution trace, then the program's stdout."
                ),
            },
            {"role": "user", "content": render_ask_events_prompt(example, cap=cap)},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


_IDENT_EQ = re.compile(r"(?<![A-Za-z0-9_])([A-Za-z_][A-Za-z0-9_]*)=")


def user_state_fields(state: str) -> str:
    """Keep user locals. Dunder dumps may be truncated, so do not brace-parse them."""
    matches = list(_IDENT_EQ.finditer(state or ""))
    kept: list[str] = []
    for index, match in enumerate(matches):
        name = match.group(1)
        if name.startswith("__"):
            continue
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(state)
        value = state[start:end].strip().rstrip(",")
        if value:
            kept.append(f"{name}={value}")
    return ", ".join(kept)


def compact_ask_event_line(event: EvidenceEvent) -> str | None:
    """One CALL/BRANCH/LINE/STATE teacher line, without tracer dumps."""
    if event.role_id not in RUNTIME_KEEP:
        return None
    role = ROLE_NAMES[event.role_id]
    parsed = _EVENT_CONTENT.match((event.content or "").strip())
    if parsed is None:
        body = user_state_fields(event.content or "") or (event.content or "").strip()
        body = _clip_gist_line(body, _COMPACT_WIDTH)
        return f"{role}: {body}" if body else None
    source = parsed.group("source").strip()
    state = user_state_fields(parsed.group("state"))
    if event.role_id == CALL:
        body = parsed.group("function")
    elif event.role_id == STATE:
        body = state or source
    else:
        body = source or state
    body = _clip_gist_line(body, _COMPACT_WIDTH)
    if not body:
        return None
    return f"{role}: {body}"


def gold_ask_events_text(example: SimExample, *, cap: int = NATIVE_CAP) -> str:
    """Capped compact teacher trace. Gold stdout is redacted from event lines."""
    if cap <= 0:
        raise ValueError("cap must be positive")
    lines: list[str] = []
    gold_n = normalize_output(example.gold_stdout)
    for event in example.events:
        line = compact_ask_event_line(event)
        if not line:
            continue
        if len(gold_n) >= 2:
            line = redact_answer(line, example.gold_stdout)
        lines.append(line)
        if len(lines) >= cap:
            break
    return "\n".join(lines)


def sft_stdout_target(example: SimExample) -> str:
    return f"```\n{normalize_output(example.gold_stdout)}\n```"


def sft_events_target(example: SimExample, *, cap: int = NATIVE_CAP) -> str:
    events = gold_ask_events_text(example, cap=cap)
    if not events.strip():
        raise ValueError(f"no compact events for {example.example_id}")
    return f"```\n{events}\n```\n\n{sft_stdout_target(example)}"


def sft_target(example: SimExample, condition: str, *, cap: int = NATIVE_CAP) -> str:
    if condition == SFT_STDOUT:
        return sft_stdout_target(example)
    if condition == SFT_EVENTS:
        return sft_events_target(example, cap=cap)
    raise ValueError(f"unknown SFT condition: {condition}")


def assert_train_disjoint(train: list[SimExample], test: list[SimExample]) -> None:
    for field in ("example_id", "task_id", "source_hash"):
        overlap = {getattr(row, field) for row in train} & {getattr(row, field) for row in test}
        if overlap:
            preview = ", ".join(str(item) for item in sorted(overlap, key=str)[:3])
            raise ValueError(f"train/test overlap on {field}: {preview}")


def count_event_role_lines(text: str) -> int:
    return len(_EVENT_ROLE_LINE.findall(text or ""))


def response_event_prefix(response: str) -> str:
    """Text before the last fenced block, which is scored as stdout."""
    blocks = list(_FENCE.finditer(response or ""))
    if not blocks:
        return response or ""
    return (response or "")[: blocks[-1].start()]


def mcnemar_exact(wins: int, losses: int) -> float:
    """Two-sided exact binomial McNemar p-value for discordant pairs."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = 0.0
    coeff = 1.0
    for i in range(k + 1):
        if i:
            coeff *= (n - i + 1) / i
        p += coeff
    p *= 0.5 ** n
    return min(1.0, 2.0 * p)


def _mcnemar_pair(rows: list[dict], left: str, right: str) -> dict:
    by_task: dict[int, dict[str, bool]] = {}
    for row in rows:
        by_task.setdefault(int(row["task_id"]), {})[row["condition"]] = bool(row["passed"])
    left_ids: list[int] = []
    right_ids: list[int] = []
    for task_id, outcomes in by_task.items():
        if left not in outcomes or right not in outcomes:
            continue
        if outcomes[left] and not outcomes[right]:
            left_ids.append(task_id)
        if outcomes[right] and not outcomes[left]:
            right_ids.append(task_id)
    return {
        "left_only": len(left_ids),
        "right_only": len(right_ids),
        "p": mcnemar_exact(len(left_ids), len(right_ids)),
        "left_ids": left_ids,
        "right_ids": right_ids,
    }


def summarize_ask_events_gate0(rows: list[dict]) -> dict:
    """Kill rule for interpreter-event chains. Gold traces stay in the prompt as trace_text."""
    present = {row["condition"] for row in rows}
    required = {"code_input", "trace_text", ASK_EVENTS}
    if not required <= present:
        missing = sorted(required - present)
        raise ValueError(f"ask-events Gate 0 needs {sorted(required)}; missing {missing}")
    counts = {
        condition: sum(row["passed"] for row in rows if row["condition"] == condition)
        for condition in ("code_input", "trace_text", ASK_EVENTS)
    }
    n = {
        condition: sum(1 for row in rows if row["condition"] == condition)
        for condition in counts
    }
    ask_vs_code = _mcnemar_pair(rows, ASK_EVENTS, "code_input")
    gold_vs_code = _mcnemar_pair(rows, "trace_text", "code_input")
    ask_vs_gold = _mcnemar_pair(rows, ASK_EVENTS, "trace_text")

    def beats(pair: dict) -> bool:
        return pair["left_only"] > pair["right_only"] and pair["p"] < 0.05

    if beats(ask_vs_code):
        call = "gate0_pass_inference"
        note = "ask_events beats code_input by McNemar. Inference-only method; skip SFT."
    elif beats(gold_vs_code):
        call = "gate0_sft_justified"
        note = (
            "ask_events does not beat code_input, but gold compact events still do. "
            "The model can read events it cannot invent. SFT on disjoint train is justified."
        )
    else:
        call = "gate0_fail"
        note = "Gold compact events no longer beat code_input. Kill the method. Do not SFT."
    return {
        "call": call,
        "note": note,
        "n": n,
        "correct": counts,
        "ask_vs_code": ask_vs_code,
        "gold_vs_code": gold_vs_code,
        "ask_vs_gold": ask_vs_gold,
        "mean_event_lines": (
            sum(int(row.get("event_lines") or 0) for row in rows if row["condition"] == ASK_EVENTS)
            / max(1, n[ASK_EVENTS])
        ),
    }


def summarize_event_sft(rows: list[dict]) -> dict:
    """Kill rule: event-chain SFT must beat stdout-only SFT on the frozen 128."""
    present = {row["condition"] for row in rows}
    required = {SFT_EVENTS, SFT_STDOUT}
    if not required <= present:
        missing = sorted(required - present)
        raise ValueError(f"event-chain SFT needs {sorted(required)}; missing {missing}")
    counts = {
        condition: sum(row["passed"] for row in rows if row["condition"] == condition)
        for condition in (SFT_EVENTS, SFT_STDOUT)
    }
    n = {
        condition: sum(1 for row in rows if row["condition"] == condition)
        for condition in counts
    }
    events_vs_stdout = _mcnemar_pair(rows, SFT_EVENTS, SFT_STDOUT)
    beats = events_vs_stdout["left_only"] > events_vs_stdout["right_only"] and events_vs_stdout["p"] < 0.05
    if beats:
        call = "sft_pass"
        note = (
            "Event-chain SFT beats stdout-only SFT by McNemar. "
            "Process supervision in interpreter-event language is live."
        )
    else:
        call = "sft_fail"
        note = (
            "Event-chain SFT does not beat stdout-only SFT. "
            "Gold-prompt win was a reading aid. Kill the method."
        )
    return {
        "call": call,
        "note": note,
        "n": n,
        "correct": counts,
        "events_vs_stdout": events_vs_stdout,
        "mean_event_lines": (
            sum(int(row.get("event_lines") or 0) for row in rows if row["condition"] == SFT_EVENTS)
            / max(1, n[SFT_EVENTS])
        ),
    }


def extract_output(response: str) -> str:
    """Take the last fenced block, else the stripped response."""
    blocks = _FENCE.findall(response or "")
    if blocks:
        return blocks[-1]
    return (response or "").strip()


def prompt_contains_gold(prompt: str, gold: str) -> bool:
    gold_n = normalize_output(gold)
    if len(gold_n) < 2:
        return False
    return gold_n in prompt


def prompt_introduces_gold(prompt: str, example: SimExample) -> bool:
    """True when gold stdout is in the prompt and not already in source or stdin."""
    gold_n = normalize_output(example.gold_stdout)
    if len(gold_n) < 2 or gold_n not in prompt:
        return False
    return gold_n not in example.source and gold_n not in example.stdin


def example_from_stdio(
    source: str,
    stdin: str,
    gold: str,
    *,
    task_id: int,
    example_id: str,
    source_hash: str = "",
    max_events_per_test: int = 24,
    timeout_seconds: float = 4.0,
) -> SimExample | None:
    test = encode_stdio_test(stdin, gold)
    task = MBPPTask(
        task_id=task_id,
        description="",
        canonical_source=source,
        tests=(test,),
    )
    mutation = MBPPMutation("canonical", 0, source)
    events = build_events(
        task,
        mutation,
        (test,),
        max_events_per_test=max_events_per_test,
        timeout_seconds=timeout_seconds,
    )
    if not events or not any(event.role_id == PASS for event in events):
        return None
    intermediate = filter_intermediate(events, gold)
    if not intermediate:
        return None
    return SimExample(
        example_id=example_id,
        task_id=task_id,
        source=source,
        stdin=stdin,
        gold_stdout=gold,
        events=intermediate,
        event_count=len(intermediate),
        source_hash=source_hash or source,
    )


def different_task_example(examples: list[SimExample], index: int) -> SimExample:
    task_id = examples[index].task_id
    for offset in range(1, len(examples)):
        candidate = examples[(index + offset) % len(examples)]
        if candidate.task_id != task_id:
            return candidate
    raise ValueError("shuffled control requires at least two task IDs")


def example_from_repair(
    example: RepairExample,
    *,
    max_events_per_test: int = 24,
    timeout_seconds: float = 4.0,
) -> SimExample | None:
    parsed = parse_stdio_test(example.failing_public_test)
    if parsed is None:
        return None
    stdin, gold = parsed
    task = MBPPTask(
        task_id=example.task_id,
        description=example.description,
        canonical_source=example.target_source,
        tests=(example.failing_public_test,),
        setup_source=example.setup_source,
    )
    mutation = MBPPMutation("canonical", 0, example.target_source)
    events = build_events(
        task,
        mutation,
        (example.failing_public_test,),
        max_events_per_test=max_events_per_test,
        timeout_seconds=timeout_seconds,
    )
    if not events:
        return None
    intermediate = filter_intermediate(events, gold)
    return SimExample(
        example_id=f"sim_{example.example_id}",
        task_id=example.task_id,
        source=example.target_source,
        stdin=stdin,
        gold_stdout=gold,
        events=intermediate,
        event_count=len(intermediate),
        source_hash=example.source_hash,
    )


def example_to_json(example: SimExample) -> dict:
    payload = asdict(example)
    payload["events"] = [asdict(event) for event in example.events]
    return payload


def example_from_json(row: dict) -> SimExample:
    events = tuple(
        EvidenceEvent(int(event["test_id"]), int(event["role_id"]), str(event["content"]))
        for event in row.get("events") or []
    )
    return SimExample(
        example_id=str(row["example_id"]),
        task_id=int(row["task_id"]),
        source=str(row["source"]),
        stdin=str(row["stdin"]),
        gold_stdout=str(row["gold_stdout"]),
        events=events,
        event_count=int(row.get("event_count", len(events))),
        source_hash=str(row["source_hash"]),
    )


def read_examples(path: str | Path) -> list[SimExample]:
    return [
        example_from_json(json.loads(line))
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]
