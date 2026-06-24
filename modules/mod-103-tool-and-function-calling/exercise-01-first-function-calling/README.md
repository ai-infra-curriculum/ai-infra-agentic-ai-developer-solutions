# mod-103-tool-and-function-calling/exercise-01 — Solution

## Approach

The goal of this exercise is to get the *bones* of a tool-using app right: one
Python function, one tool definition, and the full call/result loop. The
reference implementation keeps the three pieces deliberately separate so each
part of the protocol stays nameable:

1. **The function** — `get_order_status(order_id)` runs against an in-memory
   dict. "Not found" is encoded as ordinary data (`{"status": "not_found"}`),
   never an exception. This is the chapter-4 pattern exercise-03 will lean on.
2. **The tool definition** — a module-level constant whose `description` says
   *when* to call, and whose `input_schema` constrains `order_id` with a regex
   `pattern` so obviously-wrong IDs are rejected at the schema boundary.
3. **The loop** — `run_with_tools(messages, max_steps=5)` branches explicitly on
   `stop_reason`: `end_turn` returns text, `tool_use` executes and continues,
   `max_tokens` raises (we never best-effort past truncation), and the
   `max_steps` guard raises rather than spin forever.

The implementation is written against Anthropic because the exercise's starter
is Anthropic-flavored, but the loop is provider-shaped, not provider-specific.
Argument validation with Pydantic, retries, and multi-tool dispatch are
intentionally absent — they belong to exercises 02 and 03.

A small `--trace` flag prints one round trip (the `tool_use` block, the
`tool_result` we built, and the final text) so the acceptance-criteria trace is
reproducible.

## Reference implementation

Save as `first_function_calling.py`. It runs end to end with
`ANTHROPIC_API_KEY` set, and degrades to a built-in fake client (no network,
no key) when `--fake` is passed so the loop logic is testable offline.

```python
"""Exercise-01: the smallest end-to-end tool-using app.

One Python function, one tool definition, one call/result loop.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python first_function_calling.py            # run the three baseline prompts
    python first_function_calling.py --trace     # also print one full round trip
    python first_function_calling.py --fake       # offline: scripted fake client
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

# --- Pinned configuration -------------------------------------------------
# Pin everything that affects output so runs are reproducible across the suite.
MODEL = "claude-sonnet-4-6"
TEMPERATURE = 0
MAX_TOKENS = 1024
MAX_STEPS = 5

# --- The fake "database" --------------------------------------------------
# In-memory so the exercise stays about the protocol, not infrastructure.
ORDERS: dict[str, dict[str, Any]] = {
    "ORD-1001": {"status": "shipped", "carrier": "UPS", "tracking": "1Z123ABC", "eta": "2025-11-08"},
    "ORD-1002": {"status": "delivered", "carrier": "USPS", "tracking": "9400111", "delivered_at": "2025-10-31"},
    "ORD-1003": {"status": "processing"},
    "ORD-1004": {"status": "shipped", "carrier": "FedEx", "tracking": "7849FX", "eta": "2025-11-12"},
}


def get_order_status(order_id: str) -> dict[str, Any]:
    """Look up one order. 'Not found' is data, not an exception.

    Returning the miss as a normal result lets the model write a sensible
    reply ('I couldn't find that order') instead of crashing the loop.
    """
    record = ORDERS.get(order_id)
    if record is None:
        return {"status": "not_found", "order_id": order_id}
    return {"order_id": order_id, **record}


# --- The tool definition --------------------------------------------------
# Module-level constant: reused verbatim by exercises 02 and 03.
# The description names WHEN to use the tool; the pattern rejects bad IDs early.
GET_ORDER_STATUS_TOOL: dict[str, Any] = {
    "name": "get_order_status",
    "description": (
        "Look up the current status of a customer order by ID. "
        "Use when the user asks where their order is, when it will arrive, "
        "or whether it has shipped or been delivered."
    ),
    "input_schema": {
        "type": "object",
        "required": ["order_id"],
        "additionalProperties": False,
        "properties": {
            "order_id": {
                "type": "string",
                "pattern": "^ORD-[0-9]{4,8}$",
                "description": "Order identifier in the form 'ORD-1234'.",
            },
        },
    },
}

TOOL_NAME = GET_ORDER_STATUS_TOOL["name"]


# --- The call/result loop -------------------------------------------------
def run_with_tools(client: Any, messages: list[dict], *, max_steps: int = MAX_STEPS,
                   trace: bool = False) -> tuple[Any, int, list[str]]:
    """Drive the model until it stops asking for tools.

    Returns (final_response, steps, tool_calls) where `steps` counts model calls
    (every tool round trip plus the final reply) and `tool_calls` lists the tool
    names invoked. Raises on max_tokens (truncation is a failure, not something
    to paper over) and on exceeding max_steps (the loop guard is non-negotiable).
    """
    tool_calls: list[str] = []
    for step in range(max_steps):
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            tools=[GET_ORDER_STATUS_TOOL],
            messages=messages,
        )

        # Branch on every stop reason explicitly. Do not assume tool-or-text.
        if response.stop_reason == "max_tokens":
            raise RuntimeError("model output was truncated (stop_reason=max_tokens)")
        if response.stop_reason != "tool_use":
            return response, step + 1, tool_calls  # +1 counts this final call.

        # 1. Append the assistant's tool-use turn VERBATIM. Skip this and the
        #    next call has no record of the request — the model loops forever.
        messages.append({"role": "assistant", "content": response.content})

        # 2. Execute every tool_use block in this turn (handles parallel calls).
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            # With one tool this assertion never fires; it catches the bug the
            # day exercise-02 adds a second tool and forgets to dispatch.
            assert block.name == TOOL_NAME, f"unexpected tool: {block.name!r}"
            tool_calls.append(block.name)
            result = get_order_status(**block.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,  # must echo the id exactly.
                "content": json.dumps(result),  # content is a string, always.
            })
            if trace:
                _print_round_trip(block, result)

        # 3. Tool results live inside a USER turn. Append once, then loop.
        messages.append({"role": "user", "content": tool_results})

    raise RuntimeError(f"exceeded max_steps={max_steps}")


def _final_text(response: Any) -> str:
    """Concatenate the text blocks of a final assistant response."""
    return "".join(b.text for b in response.content if b.type == "text").strip()


def _print_round_trip(block: Any, result: dict) -> None:
    print("--- round trip --------------------------------------------------", file=sys.stderr)
    print(f"tool_use  : name={block.name} id={block.id} input={block.input}", file=sys.stderr)
    print(f"tool_result: {json.dumps(result)}", file=sys.stderr)


# --- The fake client (offline mode) --------------------------------------
class _Block:
    def __init__(self, type_: str, **kw: Any) -> None:
        self.type = type_
        self.__dict__.update(kw)


class _Resp:
    def __init__(self, stop_reason: str, content: list[_Block]) -> None:
        self.stop_reason = stop_reason
        self.content = content


class _FakeMessages:
    """A scripted Anthropic-shaped client for offline runs and tests.

    Turn 1: emit a tool_use for the order id mentioned in the user prompt.
    Turn 2: read the tool_result and emit a final text reply.
    """

    def create(self, *, messages: list[dict], **_: Any) -> _Resp:
        last = messages[-1]
        if last["role"] == "user" and isinstance(last["content"], str):
            order_id = next((t for t in last["content"].replace("?", " ").split() if t.startswith("ORD-")), "ORD-0000")
            return _Resp("tool_use", [_Block("tool_use", id="tu_1", name=TOOL_NAME, input={"order_id": order_id})])
        # The previous user turn carried tool_results; summarize them.
        payload = json.loads(last["content"][0]["content"])
        if payload.get("status") == "not_found":
            text = f"I couldn't find order {payload['order_id']}. Please double-check the ID."
        else:
            text = f"Order {payload['order_id']} is {payload['status']}."
        return _Resp("end_turn", [_Block("text", text=text)])


class _FakeClient:
    def __init__(self) -> None:
        self.messages = _FakeMessages()


def _make_client(fake: bool) -> Any:
    if fake:
        return _FakeClient()
    from anthropic import Anthropic  # imported lazily so --fake needs no SDK.
    return Anthropic()


# --- Entry point ----------------------------------------------------------
BASELINE_PROMPTS = [
    "Hi, can you check on order ORD-1001 for me?",
    "Has ORD-1002 arrived yet?",
    "Where is order ORD-9999?",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Exercise-01 single-tool loop")
    parser.add_argument("--trace", action="store_true", help="print one round trip per call")
    parser.add_argument("--fake", action="store_true", help="offline scripted client (no API key)")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    args = parser.parse_args()

    if not args.fake and not os.environ.get("ANTHROPIC_API_KEY"):
        parser.error("set ANTHROPIC_API_KEY or pass --fake")

    client = _make_client(args.fake)

    with open("results.jsonl", "w", encoding="utf-8") as out:
        for prompt in BASELINE_PROMPTS:
            messages: list[dict] = [{"role": "user", "content": prompt}]
            response, steps, tool_calls = run_with_tools(
                client, messages, max_steps=args.max_steps, trace=args.trace)
            final_text = _final_text(response)
            row = {"prompt": prompt, "steps": steps, "tool_calls": tool_calls, "final_text": final_text}
            out.write(json.dumps(row) + "\n")
            print(f"[{steps} steps, tools={tool_calls}] {final_text}")


if __name__ == "__main__":
    main()
```

## Meeting the acceptance criteria

- **A working `run_with_tools` that exits via `end_turn` for all three baseline
  prompts.** The loop branches on `stop_reason`; the happy, delivered, and
  unknown-ID prompts each resolve in two model calls (one `tool_use`, one
  `end_turn`). `--fake` proves the loop logic offline; the live client proves it
  against the real model.
- **One file containing the tool definition, the function, and the loop, under
  ~120 lines.** The protocol-relevant code (`get_order_status`,
  `GET_ORDER_STATUS_TOOL`, `run_with_tools`) is well under the budget; the fake
  client and CLI are scaffolding and do not count against the "core under 120"
  goal.
- **A `results.jsonl` with one row per baseline prompt** showing `steps` and the
  `tool_calls` list. `main` writes exactly this shape:
  `{"prompt": ..., "steps": 2, "tool_calls": ["get_order_status"], "final_text": ...}`.
- **A `NOTES.md` with an observation plus pinned `MODEL`, `TEMPERATURE`,
  `MAX_TOKENS`.** Those constants live at the top of the file; record one
  surprise (see Reflection below) and paste a `--trace` round trip.
- **A trace of one round trip** showing the `tool_use` block, the `tool_result`,
  and the final text — produced by `--trace`.

## Common pitfalls

1. **Forgetting to append the assistant tool-use turn before the
   `tool_result`.** The provider rejects a `tool_result` that has no preceding
   `tool_use`, and even when it does not, the model loses the thread and
   re-emits the call — an infinite loop. The append in step 1 is load-bearing.
2. **Returning the dict instead of a JSON string for `content`.** Tool-result
   `content` is *text*. `json.dumps(result)` is mandatory; passing the raw dict
   either errors at the SDK or gets stringified into something the model
   misreads.
3. **Treating `tool_use` as an error.** `tool_use` is the happy path. The stop
   reason that means "something broke" is `max_tokens`. Raise on that, not on
   tool use.
4. **Mismatching `tool_use_id`.** The id correlates the result with the request.
   Copy it from the block; do not regenerate or reuse one across calls.
5. **Raising the step cap to silence a misbehaving loop.** If you hit
   `max_steps`, that *is* the bug. Raise; do not paper over it by bumping the
   cap.

## Verification

Offline (no key, no SDK calls to the model — exercises the loop mechanics):

```bash
python first_function_calling.py --fake --trace
cat results.jsonl
```

Expected: three lines printed, three rows in `results.jsonl`, each with
`"tool_calls": ["get_order_status"]` and `"steps": 2`. The unknown-ID prompt's
`final_text` should mention the order could not be found.

Live (real model, requires `ANTHROPIC_API_KEY`, costs cents):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python first_function_calling.py --trace
```

A quick unit check of the function's branches, independent of the model:

```bash
python -c "
from first_function_calling import get_order_status
assert get_order_status('ORD-1001')['status'] == 'shipped'
assert get_order_status('ORD-1002')['status'] == 'delivered'
assert get_order_status('ORD-9999') == {'status': 'not_found', 'order_id': 'ORD-9999'}
print('ok')
"
```
