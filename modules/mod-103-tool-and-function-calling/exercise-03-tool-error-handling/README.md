# mod-103-tool-and-function-calling/exercise-03 — Solution

## Approach

This exercise hardens the exercise-02 surface so the four failure modes from
chapter 6 — malformed arguments, expected business outcomes, transient
failures, and internal bugs — become recoverable feedback instead of crashes.
Three layers, added in order:

1. **Validation at the boundary.** Each tool gets a Pydantic args model with
   `extra="forbid"`. The dispatcher validates *first*, then calls the function
   with the validated, typed args. A `ValidationError` is *not* raised to the
   loop; it is converted into `{"error": "invalid_arguments", "detail":
   err.errors()}` so the model can read which field was wrong and self-correct.
2. **Error-as-feedback via `safe_dispatch`.** A single wrapper turns every
   outcome into `{"is_error": bool, "payload": ...}`. It distinguishes
   `invalid_arguments` (model's fault), `transient` (retriable), and `internal`
   (our bug). Business outcomes (`not_eligible`, `not_cancellable`) are returned
   as ordinary data with `is_error=False` — they are facts, not failures.
3. **A `ToolBudget`.** Per-tool and total caps; exhaustion raises
   `BudgetExceeded` *out* of the loop. We deliberately do not swallow it — the
   whole point of the budget is to fail loudly and surface a clean final message.

Failures are injected on purpose behind flags so the exercise-02 happy path
still works: `FLAKE_SEARCH=1` makes the first `search_products` of each
conversation raise a `RetriableError`; `create_refund` returns `not_eligible`
as data for a cancelled order. The Anthropic `is_error: true` flag rides on each
tool result; for OpenAI it has no protocol field and the `error` key in the
payload carries the signal.

## Reference implementation

Save as `tool_error_handling.py`. Imports the registry, tools, and loop shape
from exercise-02's design and adds the validation, `safe_dispatch`, budget, and
audit layers. Runs live with `ANTHROPIC_API_KEY` or offline with `--fake`.

```python
"""Exercise-03: validation + error-as-feedback + retry budgets.

Hardens the exercise-02 four-tool surface. Every tool gets a Pydantic args
model; safe_dispatch converts failures into structured tool results; ToolBudget
caps retries and fails loudly on exhaustion.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import uuid
from collections import Counter
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field, ValidationError

log = logging.getLogger("tools")

MODEL = "claude-sonnet-4-6"
TEMPERATURE = 0
MAX_TOKENS = 1024
MAX_STEPS = 12

# --- Fake state -----------------------------------------------------------
ORDERS: dict[str, dict[str, Any]] = {
    "ORD-1001": {"status": "shipped"},
    "ORD-1002": {"status": "delivered"},
    "ORD-1003": {"status": "processing"},
    "ORD-1004": {"status": "shipped"},
    "ORD-5005": {"status": "cancelled"},
}

PRODUCTS = [
    {"sku": "BL-MED-001", "name": "Blue T-Shirt, Medium", "in_stock": True, "price_cents": 1999},
    {"sku": "RD-MED-001", "name": "Red T-Shirt, Medium", "in_stock": True, "price_cents": 1999},
    {"sku": "HS-001", "name": "Wireless Headset", "in_stock": True, "price_cents": 6999},
]

REFUNDS: dict[str, dict[str, Any]] = {}


# --- Errors ---------------------------------------------------------------
class RetriableError(Exception):
    """A transient failure the model may recover from by retrying."""


class UnknownToolError(Exception):
    pass


class BudgetExceeded(Exception):
    """Raised out of the loop when retry caps are hit. Never swallowed."""


# --- Tool functions -------------------------------------------------------
# A per-conversation flake flag, reset before each conversation.
_SEARCH_STATE = {"flaked": False}


def get_order_status(order_id: str) -> dict[str, Any]:
    record = ORDERS.get(order_id)
    if record is None:
        return {"status": "not_found", "order_id": order_id}
    return {"order_id": order_id, **record}


def search_products(query: str, limit: int = 10) -> dict[str, Any]:
    """Flaky-on-first-call when FLAKE_SEARCH=1; succeeds afterwards."""
    if os.environ.get("FLAKE_SEARCH") == "1" and not _SEARCH_STATE["flaked"]:
        _SEARCH_STATE["flaked"] = True
        raise RetriableError("upstream unavailable")
    q = query.lower()
    hits = [p for p in PRODUCTS if q in p["name"].lower()][:limit]
    return {"query": query, "count": len(hits), "results": hits}


def create_refund(order_id: str, amount_cents: int, reason: str, idempotency_key: str) -> dict[str, Any]:
    """Business outcome as DATA: a cancelled order is not_eligible (not raised)."""
    record = ORDERS.get(order_id)
    if record is None:
        return {"error": "not_eligible", "detail": "no such order"}
    if record["status"] == "cancelled":
        return {"error": "not_eligible", "detail": "cannot refund a cancelled order"}
    if idempotency_key in REFUNDS:
        return REFUNDS[idempotency_key]
    refund = {"refund_id": f"rf_{uuid.uuid4().hex[:12]}", "order_id": order_id,
              "amount_cents": amount_cents, "reason": reason}
    REFUNDS[idempotency_key] = refund
    return refund


def cancel_order(order_id: str) -> dict[str, Any]:
    record = ORDERS.get(order_id)
    if record is None:
        return {"error": "not_found", "order_id": order_id}
    if record["status"] != "processing":
        return {"error": "not_cancellable", "current_status": record["status"]}
    record["status"] = "cancelled"
    return {"order_id": order_id, "status": "cancelled"}


# --- Pydantic args models (validation at the boundary) --------------------
class GetOrderStatusArgs(BaseModel):
    model_config = {"extra": "forbid"}
    order_id: str = Field(pattern=r"^ORD-[0-9]{4,8}$")


class SearchProductsArgs(BaseModel):
    model_config = {"extra": "forbid"}
    query: str = Field(min_length=1)
    limit: int = Field(default=10, ge=1, le=20)


class CreateRefundArgs(BaseModel):
    model_config = {"extra": "forbid"}
    order_id: str = Field(pattern=r"^ORD-[0-9]{4,8}$")
    amount_cents: int = Field(ge=1, le=100_000)
    reason: Literal["damaged", "wrong_item", "not_received", "other"]
    idempotency_key: str = Field(min_length=8)


class CancelOrderArgs(BaseModel):
    model_config = {"extra": "forbid"}
    order_id: str = Field(pattern=r"^ORD-[0-9]{4,8}$")


# --- Tool definitions (schemas unchanged from exercise-02) ----------------
def _order_id_schema() -> dict:
    return {"type": "string", "pattern": "^ORD-[0-9]{4,8}$"}


TOOLS = [
    {"name": "get_order_status",
     "description": "Look up the status of a customer order by ID.",
     "input_schema": {"type": "object", "required": ["order_id"], "additionalProperties": False,
                      "properties": {"order_id": _order_id_schema()}}},
    {"name": "search_products",
     "description": "Search the product catalog by free text. Use for stock, price, or SKU questions.",
     "input_schema": {"type": "object", "required": ["query"], "additionalProperties": False,
                      "properties": {"query": {"type": "string"},
                                     "limit": {"type": "integer", "minimum": 1, "maximum": 20}}}},
    {"name": "create_refund",
     "description": "Issue a refund. Use only after the user confirms amount and reason.",
     "input_schema": {"type": "object",
                      "required": ["order_id", "amount_cents", "reason", "idempotency_key"],
                      "additionalProperties": False,
                      "properties": {"order_id": _order_id_schema(),
                                     "amount_cents": {"type": "integer", "minimum": 1, "maximum": 100000},
                                     "reason": {"type": "string",
                                                "enum": ["damaged", "wrong_item", "not_received", "other"]},
                                     "idempotency_key": {"type": "string", "minLength": 8}}}},
    {"name": "cancel_order",
     "description": "Cancel an order in 'processing'. Reports not_cancellable otherwise.",
     "input_schema": {"type": "object", "required": ["order_id"], "additionalProperties": False,
                      "properties": {"order_id": _order_id_schema()}}},
]


# --- Registry with args models --------------------------------------------
class ToolRegistry:
    def __init__(self) -> None:
        self._fns: dict[str, Callable[..., Any]] = {}
        self._arg_models: dict[str, type[BaseModel]] = {}

    def register(self, definition: dict, fn: Callable[..., Any], args_model: type[BaseModel]) -> None:
        self._fns[definition["name"]] = fn
        self._arg_models[definition["name"]] = args_model

    def dispatch(self, name: str, raw_args: dict) -> Any:
        """Validate first (may raise ValidationError), then call with typed args."""
        if name not in self._fns:
            raise UnknownToolError(name)
        model = self._arg_models[name]
        validated = model.model_validate(raw_args)  # raises ValidationError
        return self._fns[name](**validated.model_dump())


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(TOOLS[0], get_order_status, GetOrderStatusArgs)
    registry.register(TOOLS[1], search_products, SearchProductsArgs)
    registry.register(TOOLS[2], create_refund, CreateRefundArgs)
    registry.register(TOOLS[3], cancel_order, CancelOrderArgs)
    return registry


# --- Budget ---------------------------------------------------------------
class ToolBudget:
    def __init__(self, max_per_tool: int = 3, max_total: int = 10) -> None:
        self.per_tool: Counter = Counter()
        self.total = 0
        self.max_per_tool = max_per_tool
        self.max_total = max_total

    def charge(self, name: str) -> None:
        self.per_tool[name] += 1
        self.total += 1
        if self.per_tool[name] > self.max_per_tool:
            raise BudgetExceeded(f"too many calls to {name!r}")
        if self.total > self.max_total:
            raise BudgetExceeded("too many tool calls overall")


# --- safe_dispatch: every outcome becomes a structured result -------------
def safe_dispatch(name: str, raw_args: dict, registry: ToolRegistry, budget: ToolBudget) -> dict:
    """Return {"is_error": bool, "payload": ...}. Budget charge raises out."""
    budget.charge(name)  # BudgetExceeded propagates by design.
    try:
        result = registry.dispatch(name, raw_args)
        return {"is_error": False, "payload": result}
    except ValidationError as err:
        return {"is_error": True, "payload": {"error": "invalid_arguments", "detail": err.errors()}}
    except RetriableError as err:
        return {"is_error": True, "payload": {"error": "transient", "detail": str(err)}}
    except Exception:
        log.exception("tool %s crashed", name)
        return {"is_error": True, "payload": {"error": "internal", "detail": "internal error"}}


# --- The hardened loop ----------------------------------------------------
def run_with_tools(client: Any, messages: list[dict], registry: ToolRegistry, budget: ToolBudget,
                   *, audit: list[dict] | None = None, prompt_index: int = 0,
                   max_steps: int = MAX_STEPS) -> tuple[Any, list[dict], bool]:
    """Returns (final_response, per_call_records, budget_exhausted)."""
    records: list[dict] = []
    for step in range(max_steps):
        response = client.messages.create(model=MODEL, max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
                                          tools=TOOLS, messages=messages)
        if response.stop_reason == "max_tokens":
            raise RuntimeError("model output was truncated (stop_reason=max_tokens)")
        if response.stop_reason != "tool_use":
            return response, records, False

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            t0 = time.monotonic()
            try:
                out = safe_dispatch(block.name, block.input, registry, budget)
            except BudgetExceeded:
                return None, records, True  # surface a clean message upstream.
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            payload = out["payload"]
            category = payload.get("error") if isinstance(payload, dict) and "error" in payload else None
            rec = {"name": block.name, "args": block.input, "is_error": out["is_error"], "category": category}
            records.append(rec)
            if audit is not None:
                audit.append({"prompt_index": prompt_index, "step": step, **rec, "elapsed_ms": elapsed_ms})
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(payload),
                "is_error": out["is_error"],  # Anthropic-specific; OpenAI relies on the payload's error key.
            })
        messages.append({"role": "user", "content": tool_results})

    raise RuntimeError(f"exceeded max_steps={max_steps}")


def _final_text(response: Any) -> str:
    if response is None:
        return "I wasn't able to complete this request — please try again or contact support."
    return "".join(b.text for b in response.content if b.type == "text").strip()


# --- Fake client (offline, scripted to exercise each failure bucket) ------
class _Block:
    def __init__(self, type_: str, **kw: Any) -> None:
        self.type = type_
        self.__dict__.update(kw)


class _Resp:
    def __init__(self, stop_reason: str, content: list[_Block]) -> None:
        self.stop_reason = stop_reason
        self.content = content


class _FakeMessages:
    """Routes on the latest user text; on an error result, retries once cleanly."""

    def create(self, *, messages: list[dict], **_: Any) -> _Resp:
        last = messages[-1]
        if last["role"] == "user" and isinstance(last["content"], list):
            # Previous turn was tool_results: read the first payload and react.
            payload = json.loads(last["content"][0]["content"])
            err = payload.get("error") if isinstance(payload, dict) else None
            if err == "invalid_arguments":
                return _Resp("tool_use", [_Block("tool_use", id="t2", name="create_refund", input={
                    "order_id": "ORD-1004", "amount_cents": 5000, "reason": "damaged",
                    "idempotency_key": "fix-12345"})])
            if err == "transient":
                return _Resp("tool_use", [_Block("tool_use", id="t2", name="search_products",
                                                 input={"query": "headset"})])
            return _Resp("end_turn", [_Block("text", text=f"Result: {json.dumps(payload)}")])
        text = last["content"].lower()
        if "refund order 1004" in text:  # malformed: no ORD- prefix, "fifty bucks", no key.
            return _Resp("tool_use", [_Block("tool_use", id="t1", name="create_refund", input={
                "order_id": "1004", "amount_cents": "fifty", "reason": "damaged"})])
        if "headset" in text or "stock" in text:
            return _Resp("tool_use", [_Block("tool_use", id="t1", name="search_products",
                                             input={"query": "headset"})])
        if "how are you" in text or "check in" in text:
            return _Resp("end_turn", [_Block("text", text="Doing well, thanks for checking in!")])
        return _Resp("end_turn", [_Block("text", text="Understood.")])


class _FakeClient:
    def __init__(self) -> None:
        self.messages = _FakeMessages()


def _make_client(fake: bool) -> Any:
    if fake:
        return _FakeClient()
    from anthropic import Anthropic
    return Anthropic()


# --- Adversarial suite ----------------------------------------------------
SUITE = [
    "Refund order 1004 for fifty bucks because it's damaged.",
    "Refund order ORD-5005 for $5 because it's damaged.",
    "Is the wireless headset in stock?",
    "Search for a blue shirt. If you don't find one, try a red shirt. "
    "If you don't find that either, try a green shirt. Then a yellow shirt.",
    "Cancel ORD-1001, then refund it.",
    "Hi, just wanted to check in. How are you doing today?",
]


def run_suite(client: Any) -> None:
    audit: list[dict] = []
    with open("results.jsonl", "w", encoding="utf-8") as out:
        for i, prompt in enumerate(SUITE):
            _SEARCH_STATE["flaked"] = False  # reset the per-conversation flake.
            budget = ToolBudget(max_per_tool=3, max_total=10)  # fresh budget per conversation.
            messages: list[dict] = [{"role": "user", "content": prompt}]
            response, records, exhausted = run_with_tools(
                client, messages, build_registry(), budget, audit=audit, prompt_index=i)
            row = {"prompt": prompt, "tool_calls": records,
                   "final_text": _final_text(response), "budget_exhausted": exhausted}
            out.write(json.dumps(row) + "\n")
            print(f"[exhausted={exhausted}] {[r['name'] for r in records]}")
    with open("audit.jsonl", "w", encoding="utf-8") as out:
        for entry in audit:
            out.write(json.dumps(entry) + "\n")


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description="Exercise-03 hardened tool loop")
    parser.add_argument("--fake", action="store_true", help="offline scripted client")
    args = parser.parse_args()
    if not args.fake and not os.environ.get("ANTHROPIC_API_KEY"):
        parser.error("set ANTHROPIC_API_KEY or pass --fake")
    run_suite(_make_client(args.fake))


if __name__ == "__main__":
    main()
```

Model-free tests for `safe_dispatch` (stretch goal — assert each error category
without invoking the model). Save as `test_safe_dispatch.py`:

```python
import tool_error_handling as app


def _budget():
    return app.ToolBudget(max_per_tool=5, max_total=20)


def test_invalid_arguments_category():
    out = app.safe_dispatch("create_refund", {"order_id": "1004"}, app.build_registry(), _budget())
    assert out["is_error"] and out["payload"]["error"] == "invalid_arguments"


def test_transient_category(monkeypatch):
    monkeypatch.setenv("FLAKE_SEARCH", "1")
    app._SEARCH_STATE["flaked"] = False
    out = app.safe_dispatch("search_products", {"query": "headset"}, app.build_registry(), _budget())
    assert out["is_error"] and out["payload"]["error"] == "transient"


def test_business_outcome_is_not_error():
    out = app.safe_dispatch("create_refund",
                            {"order_id": "ORD-5005", "amount_cents": 500, "reason": "damaged",
                             "idempotency_key": "k" * 8}, app.build_registry(), _budget())
    assert out["is_error"] is False and out["payload"]["error"] == "not_eligible"


def test_budget_exceeded_raises():
    budget = app.ToolBudget(max_per_tool=1, max_total=10)
    app.safe_dispatch("get_order_status", {"order_id": "ORD-1001"}, app.build_registry(), budget)
    try:
        app.safe_dispatch("get_order_status", {"order_id": "ORD-1001"}, app.build_registry(), budget)
        assert False, "expected BudgetExceeded"
    except app.BudgetExceeded:
        pass
```

## Meeting the acceptance criteria

- **Every tool has a Pydantic args model registered with the registry.**
  `register(definition, fn, args_model)` stores all three; `dispatch` validates
  before calling. Invalid args produce a structured `invalid_arguments` result,
  never a raised exception (caught in `safe_dispatch`).
- **`safe_dispatch` distinguishes ≥ 3 error categories plus a business
  outcome.** It returns `invalid_arguments`, `transient`, and `internal`; the
  business `not_eligible` / `not_cancellable` outcomes ride through as data with
  `is_error=False`.
- **`ToolBudget` enforces per-tool and total caps.** Prompt 4 forces a fourth
  `search_products` against `max_per_tool=3`, raising `BudgetExceeded`; the loop
  returns `(None, records, True)` and `_final_text` emits the clean
  "I wasn't able to complete this request" message instead of crashing.
- **`results.jsonl` with rows for all six prompts** including per-call
  `is_error` and `category`. **`audit.jsonl`** with one row per call
  (`prompt_index`, `step`, `name`, `args`, `is_error`, `category`,
  `elapsed_ms`).
- **`NOTES.md`** should summarize per-tool call counts and error rates from
  `audit.jsonl`, plus two prompts where the model *recovered* (prompt 1: it saw
  `invalid_arguments` and re-issued a clean `create_refund`; prompt 3: it saw
  `transient` and retried search successfully).

## Common pitfalls

1. **Catching `ValidationError` inside the tool function.** Validation lives at
   the dispatcher so the *same* boundary serves every tool and the model gets
   one consistent `invalid_arguments` shape. A function that catches its own
   `ValidationError` hides the field-level detail the model needs.
2. **Swallowing `BudgetExceeded`.** The budget exists to fail loudly. If you
   catch it and continue, a stuck conversation calls `create_refund` thirteen
   times. Let it propagate and convert to a user-facing message at the top.
3. **Coercing types silently.** If the model sends `"amount_cents": "5000"`,
   let Pydantic reject it — `extra="forbid"` plus typed fields make the error
   feedback the lesson. Silent `int(...)` coercion masks the model's mistake.
4. **Returning business outcomes as `is_error=True`.** `not_eligible` and
   `not_cancellable` are facts, not failures. Marking them as errors invites the
   model to retry a call that can never succeed. Keep them `is_error=False`.
5. **Leaking internals into `detail`.** Stack traces and raw exception strings
   in the `internal` payload reach the model and, via the model, sometimes the
   user. The `internal` branch returns a fixed `"internal error"` and logs the
   trace server-side only.

## Verification

Offline adversarial suite (no key):

```bash
python tool_error_handling.py --fake
cat results.jsonl
cat audit.jsonl
```

Expected: six rows in `results.jsonl`. Prompt 1 shows an `invalid_arguments`
call followed by a clean one; prompt 6 shows an empty `tool_calls` list and
`budget_exhausted=false`. `audit.jsonl` has one entry per tool call with timing.

Transient-failure path (forces the flake):

```bash
FLAKE_SEARCH=1 python tool_error_handling.py --fake
```

The first `search_products` of the stock-check prompt records
`category="transient"`; the retry succeeds.

Model-free unit tests for the error layer:

```bash
pip install pydantic pytest
pytest test_safe_dispatch.py -q
```

Live run (costs cents):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python tool_error_handling.py
```
