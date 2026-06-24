# mod-103-tool-and-function-calling/exercise-02 — Solution

## Approach

This exercise widens the single-tool loop into a four-tool app and moves the
dispatch decision out of the loop and into a registry. The design goals:

1. **A `ToolRegistry` owns three things in one place** — the JSON schemas
   (passed as `tools=[...]` on every call), the Python callables, and the
   name → callable dispatch. The loop stops hard-coding any function; the
   `assert block.name == ...` placeholder from exercise-01 becomes
   `registry.dispatch(name, args)`.
2. **One clear purpose per tool.** `get_order_status` (read), `search_products`
   (read), `create_refund` (idempotent write), `cancel_order` (guarded write).
   No `mode` flags — the model picks between *names* more reliably than between
   enum values (chapter 5).
3. **Expected outcomes are data, not exceptions.** `cancel_order` on a
   non-`processing` order returns `{"error": "not_cancellable", ...}`; this
   previews chapter 6 without yet adding the full error layer.
4. **The loop already handles parallel tool use.** Because it iterates every
   `tool_use` block and appends all results in one user turn, a single assistant
   turn with two calls (prompt 3) and two serialized round trips both work
   unchanged.

`tool_choice` is exercised on the refund prompt (auto / forced / forbidden) to
show — per chapter 5 — that forcing a call is not a substitute for a sharp
description: a forced `create_refund` invents `amount_cents` and
`idempotency_key`, and a forbidden one risks the model *claiming* to have acted.

Pydantic validation, retries, and budgets are still out of scope; they are
exercise-03.

## Reference implementation

Save as `multi_tool_app.py`. Reuses the exercise-01 loop shape and adds the
registry and three tools. Runs live with `ANTHROPIC_API_KEY`, or offline with
`--fake`.

```python
"""Exercise-02: four tools behind one registry and dispatcher.

Extends exercise-01: same loop, but dispatch goes through a ToolRegistry and
the surface grows to get_order_status / search_products / create_refund /
cancel_order. tool_choice is exercised on the refund prompt.
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from typing import Any, Callable

MODEL = "claude-sonnet-4-6"
TEMPERATURE = 0
MAX_TOKENS = 1024
MAX_STEPS = 10

# --- Fake state -----------------------------------------------------------
ORDERS: dict[str, dict[str, Any]] = {
    "ORD-1001": {"status": "shipped", "carrier": "UPS", "tracking": "1Z123ABC", "eta": "2025-11-08"},
    "ORD-1002": {"status": "delivered", "carrier": "USPS", "tracking": "9400111", "delivered_at": "2025-10-31"},
    "ORD-1003": {"status": "processing"},
    "ORD-1004": {"status": "shipped", "carrier": "FedEx", "tracking": "7849FX", "eta": "2025-11-12"},
}

PRODUCTS: list[dict[str, Any]] = [
    {"sku": "BL-MED-001", "name": "Blue T-Shirt, Medium", "in_stock": True, "price_cents": 1999},
    {"sku": "BL-LG-001", "name": "Blue T-Shirt, Large", "in_stock": False, "price_cents": 1999},
    {"sku": "RD-MED-001", "name": "Red T-Shirt, Medium", "in_stock": True, "price_cents": 1999},
    {"sku": "RD-LG-001", "name": "Red T-Shirt, Large", "in_stock": True, "price_cents": 1999},
    {"sku": "HS-001", "name": "Wireless Headset", "in_stock": True, "price_cents": 6999},
    {"sku": "MG-001", "name": "Ceramic Mug", "in_stock": True, "price_cents": 1299},
]

REFUNDS: dict[str, dict[str, Any]] = {}  # keyed by idempotency_key.


# --- Tool functions (plain Python, independently testable) ----------------
def get_order_status(order_id: str) -> dict[str, Any]:
    record = ORDERS.get(order_id)
    if record is None:
        return {"status": "not_found", "order_id": order_id}
    return {"order_id": order_id, **record}


def search_products(query: str, limit: int = 10) -> dict[str, Any]:
    """Naive case-insensitive substring match over product names."""
    q = query.lower()
    hits = [p for p in PRODUCTS if q in p["name"].lower()][:limit]
    return {"query": query, "count": len(hits), "results": hits}


def create_refund(order_id: str, amount_cents: int, reason: str, idempotency_key: str) -> dict[str, Any]:
    """Idempotent write: the same key always returns the same refund row."""
    if idempotency_key in REFUNDS:
        return REFUNDS[idempotency_key]
    refund = {
        "refund_id": f"rf_{uuid.uuid4().hex[:12]}",
        "order_id": order_id,
        "amount_cents": amount_cents,
        "reason": reason,
    }
    REFUNDS[idempotency_key] = refund
    return refund


def cancel_order(order_id: str) -> dict[str, Any]:
    """Cancel only a 'processing' order; otherwise report why, as data."""
    record = ORDERS.get(order_id)
    if record is None:
        return {"error": "not_found", "order_id": order_id}
    if record["status"] != "processing":
        return {"error": "not_cancellable", "current_status": record["status"]}
    record["status"] = "cancelled"
    return {"order_id": order_id, "status": "cancelled"}


# --- Tool definitions -----------------------------------------------------
GET_ORDER_STATUS_TOOL = {
    "name": "get_order_status",
    "description": (
        "Look up the current status of a customer order by ID. Use when the user "
        "asks where their order is, when it will arrive, or whether it shipped."
    ),
    "input_schema": {
        "type": "object",
        "required": ["order_id"],
        "additionalProperties": False,
        "properties": {"order_id": {"type": "string", "pattern": "^ORD-[0-9]{4,8}$"}},
    },
}

SEARCH_PRODUCTS_TOOL = {
    "name": "search_products",
    "description": (
        "Search the product catalog by free-text query (matches product names). "
        "Use when the user asks whether an item exists, is in stock, its price, or "
        "its SKU. Do NOT use for order status — use get_order_status for that."
    ),
    "input_schema": {
        "type": "object",
        "required": ["query"],
        "additionalProperties": False,
        "properties": {
            "query": {"type": "string", "description": "Free-text product query."},
            "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 10},
        },
    },
}

CREATE_REFUND_TOOL = {
    "name": "create_refund",
    "description": (
        "Issue a refund against an order. Use only after the user has confirmed the "
        "amount and reason. Do NOT use to cancel an unshipped order — use cancel_order."
    ),
    "input_schema": {
        "type": "object",
        "required": ["order_id", "amount_cents", "reason", "idempotency_key"],
        "additionalProperties": False,
        "properties": {
            "order_id": {"type": "string", "pattern": "^ORD-[0-9]{4,8}$"},
            "amount_cents": {"type": "integer", "minimum": 1, "maximum": 100000},
            "reason": {"type": "string", "enum": ["damaged", "wrong_item", "not_received", "other"]},
            "idempotency_key": {"type": "string", "minLength": 8},
        },
    },
}

CANCEL_ORDER_TOOL = {
    "name": "cancel_order",
    "description": (
        "Cancel an order that has not yet shipped (status 'processing'). If the order "
        "has already shipped or been delivered, this reports 'not_cancellable'. Do NOT "
        "use to refund a delivered order — use create_refund."
    ),
    "input_schema": {
        "type": "object",
        "required": ["order_id"],
        "additionalProperties": False,
        "properties": {"order_id": {"type": "string", "pattern": "^ORD-[0-9]{4,8}$"}},
    },
}


# --- Registry -------------------------------------------------------------
class UnknownToolError(Exception):
    pass


class ToolRegistry:
    """A list of schemas plus a name → callable map. Nothing more."""

    def __init__(self) -> None:
        self._tools: list[dict] = []
        self._fns: dict[str, Callable[..., Any]] = {}

    def register(self, definition: dict, fn: Callable[..., Any]) -> None:
        name = definition["name"]
        if name in self._fns:
            raise ValueError(f"tool {name!r} already registered")
        self._tools.append(definition)
        self._fns[name] = fn

    def definitions(self) -> list[dict]:
        return list(self._tools)

    def dispatch(self, name: str, args: dict) -> Any:
        if name not in self._fns:
            raise UnknownToolError(name)
        return self._fns[name](**args)


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(GET_ORDER_STATUS_TOOL, get_order_status)
    registry.register(SEARCH_PRODUCTS_TOOL, search_products)
    registry.register(CREATE_REFUND_TOOL, create_refund)
    registry.register(CANCEL_ORDER_TOOL, cancel_order)
    return registry


# --- The loop (now dispatcher-driven) -------------------------------------
def run_with_tools(client: Any, messages: list[dict], registry: ToolRegistry, *, tool_choice: dict | None = None,
                   max_steps: int = MAX_STEPS) -> tuple[Any, list[dict]]:
    """Drive the model through the registry. Returns (final_response, calls).

    `calls` is the audit list of {name, args} the model actually invoked.
    """
    calls: list[dict] = []
    for _step in range(max_steps):
        kwargs: dict[str, Any] = dict(
            model=MODEL, max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
            tools=registry.definitions(), messages=messages,
        )
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        response = client.messages.create(**kwargs)

        if response.stop_reason == "max_tokens":
            raise RuntimeError("model output was truncated (stop_reason=max_tokens)")
        if response.stop_reason != "tool_use":
            return response, calls

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            result = registry.dispatch(block.name, block.input)  # one line replaces the assert.
            calls.append({"name": block.name, "args": block.input})
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result),
            })
        messages.append({"role": "user", "content": tool_results})

    raise RuntimeError(f"exceeded max_steps={max_steps}")


def _final_text(response: Any) -> str:
    return "".join(b.text for b in response.content if b.type == "text").strip()


# --- Fake client (offline) ------------------------------------------------
class _Block:
    def __init__(self, type_: str, **kw: Any) -> None:
        self.type = type_
        self.__dict__.update(kw)


class _Resp:
    def __init__(self, stop_reason: str, content: list[_Block]) -> None:
        self.stop_reason = stop_reason
        self.content = content


class _FakeMessages:
    """Scripted client: route on keywords in the first user turn, then reply."""

    def create(self, *, messages: list[dict], tool_choice: dict | None = None, **_: Any) -> _Resp:
        last = messages[-1]
        if last["role"] == "user" and isinstance(last["content"], str):
            if tool_choice and tool_choice.get("type") == "none":
                return _Resp("end_turn", [_Block("text", text="Yes, we do offer refunds in general.")])
            text = last["content"].lower()
            if "in stock" in text or "headset" in text or "shirt" in text:
                return _Resp("tool_use", [_Block("tool_use", id="t1", name="search_products",
                                                 input={"query": "headset" if "headset" in text else "shirt"})])
            if "cancel" in text:
                oid = _first_order(text)
                return _Resp("tool_use", [_Block("tool_use", id="t1", name="cancel_order", input={"order_id": oid})])
            if "refund" in text:
                oid = _first_order(text)
                return _Resp("tool_use", [_Block("tool_use", id="t1", name="create_refund", input={
                    "order_id": oid, "amount_cents": 1999, "reason": "damaged", "idempotency_key": "k" * 8})])
            oid = _first_order(text)
            return _Resp("tool_use", [_Block("tool_use", id="t1", name="get_order_status", input={"order_id": oid})])
        return _Resp("end_turn", [_Block("text", text="Done — here is what I found.")])


def _first_order(text: str) -> str:
    for tok in text.replace("?", " ").replace(".", " ").upper().split():
        if tok.startswith("ORD-"):
            return tok
    return "ORD-0000"


class _FakeClient:
    def __init__(self) -> None:
        self.messages = _FakeMessages()


def _make_client(fake: bool) -> Any:
    if fake:
        return _FakeClient()
    from anthropic import Anthropic
    return Anthropic()


# --- Driver ---------------------------------------------------------------
SUITE = [
    "Is the blue t-shirt in medium in stock?",
    "I'd like to cancel order ORD-1003.",
    "Where is ORD-1001, and do you have any wireless headsets?",
    "I need a refund on ORD-1002. It arrived broken.",
    "Do you do refunds in general?",
    "Cancel ORD-1001.",
    "What's the SKU for the red shirt?",
]


def run_suite(client: Any) -> None:
    registry = build_registry()
    with open("results.jsonl", "w", encoding="utf-8") as out:
        for prompt in SUITE:
            messages: list[dict] = [{"role": "user", "content": prompt}]
            response, calls = run_with_tools(client, messages, registry)
            row = {"prompt": prompt, "tool_calls": calls, "final_text": _final_text(response)}
            out.write(json.dumps(row) + "\n")
            print(f"[{[c['name'] for c in calls]}] {prompt}")


def run_tool_choice_probe(client: Any) -> None:
    """Run the refund prompt under auto / forced / forbidden tool_choice."""
    prompt = "I need a refund on ORD-1002. It arrived broken."
    variants = {
        "auto": None,
        "forced": {"type": "tool", "name": "create_refund"},
        "forbidden": {"type": "none"},
    }
    with open("results_tool_choice.jsonl", "w", encoding="utf-8") as out:
        for label, choice in variants.items():
            registry = build_registry()
            messages: list[dict] = [{"role": "user", "content": prompt}]
            response, calls = run_with_tools(client, messages, registry, tool_choice=choice)
            row = {"variant": label, "tool_calls": calls, "final_text": _final_text(response)}
            out.write(json.dumps(row) + "\n")
            print(f"[{label}] calls={[c['name'] for c in calls]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Exercise-02 multi-tool app")
    parser.add_argument("--fake", action="store_true", help="offline scripted client")
    parser.add_argument("--tool-choice", action="store_true", help="run only the tool_choice probe")
    args = parser.parse_args()

    if not args.fake and not os.environ.get("ANTHROPIC_API_KEY"):
        parser.error("set ANTHROPIC_API_KEY or pass --fake")

    client = _make_client(args.fake)
    if args.tool_choice:
        run_tool_choice_probe(client)
    else:
        run_suite(client)
        run_tool_choice_probe(client)


if __name__ == "__main__":
    main()
```

Independently testable tool functions (the acceptance criteria call for
`cancel_order` covering both paths). Save as `test_tools.py`:

```python
import multi_tool_app as app


def test_cancel_order_success():
    app.ORDERS["ORD-1003"] = {"status": "processing"}  # reset for isolation.
    out = app.cancel_order("ORD-1003")
    assert out == {"order_id": "ORD-1003", "status": "cancelled"}


def test_cancel_order_not_cancellable():
    out = app.cancel_order("ORD-1001")  # shipped, not cancellable.
    assert out == {"error": "not_cancellable", "current_status": "shipped"}


def test_create_refund_is_idempotent():
    a = app.create_refund("ORD-1002", 1999, "damaged", "key-12345")
    b = app.create_refund("ORD-1002", 9999, "other", "key-12345")
    assert a == b  # same key -> same row, ignoring the second call's args.


def test_search_products_substring():
    out = app.search_products("headset")
    assert out["count"] == 1 and out["results"][0]["sku"] == "HS-001"
```

## Meeting the acceptance criteria

- **Four tools through one registry and dispatcher.** `build_registry`
  registers all four; the loop calls `registry.dispatch(block.name,
  block.input)` and nothing else. The tool functions are plain Python, tested
  directly in `test_tools.py` (both `cancel_order` paths covered).
- **`results.jsonl` with rows for all seven prompts** — `run_suite` writes
  `{"prompt", "tool_calls": [{name, args}], "final_text"}` per prompt.
- **`results_tool_choice.jsonl` with the three refund runs** —
  `run_tool_choice_probe` writes auto / forced / forbidden variants for
  prompt 4.
- **`NOTES.md`** should record: the before/after of the one description you
  tighten (task 4), a parallel-tool-use observation from prompt 3 (task 5), and
  the pinned `MODEL`/`TEMPERATURE`/`MAX_TOKENS` (top of file).
- **A trace for prompt 3** showing parallel vs. serial — the loop logs each
  `{name, args}` into `calls`; inspect whether prompt 3 produced two entries in
  one assistant turn (parallel) or across two round trips (serial). Both are
  handled by the same loop.

## Common pitfalls

1. **Re-introducing a per-tool branch in the loop.** The whole point of the
   registry is that the loop is tool-agnostic. If you find yourself writing
   `if block.name == "create_refund"` in the loop, push that logic into the
   function or the schema.
2. **A `mode` flag instead of separate tools.** Folding cancel and refund into
   one tool with a `mode` enum degrades selection accuracy. Keep one purpose per
   name; use anti-overlap text in the descriptions instead (see the "Do NOT use
   …" clauses).
3. **Sending a partial result set on a parallel turn.** When the model emits two
   `tool_use` blocks, you must execute *both* and append *both* results in one
   user turn. Returning one confuses the next turn. The `for block in
   response.content` loop plus a single append handles this.
4. **Treating `tool_choice` as a description fix.** Forcing `create_refund` makes
   the model fabricate `amount_cents` and `idempotency_key`; forbidding tools
   risks a dishonest "I've issued the refund." The fix for bad selection is the
   description, not the choice override.
5. **Non-idempotent refund writes.** If `create_refund` does not key on
   `idempotency_key`, a model that retries (or a parallel duplicate) double-
   refunds. The dict-keyed-by-key pattern is the guard.

## Verification

Offline suite + probe (no key):

```bash
python multi_tool_app.py --fake
cat results.jsonl results_tool_choice.jsonl
```

Expected: seven rows in `results.jsonl` (the "Do you do refunds in general?"
row shows an empty `tool_calls` list only under a `tool_choice=none` run; under
the scripted `auto` client it routes to `search_products`), and three rows in
`results_tool_choice.jsonl` where the `forbidden` variant produced text and no
calls.

Unit tests for the tool functions:

```bash
pip install pytest
pytest test_tools.py -q
```

Live run (costs under a dollar):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python multi_tool_app.py
```
