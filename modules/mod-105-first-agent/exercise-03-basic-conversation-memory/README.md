# mod-105-first-agent/exercise-03 — Solution

This is the reference solution for the stateful agent: exercise-02's three-tool
agent turned into an `AgentSession` that holds a running transcript, enforces a
token budget by dropping stale tool payloads and summarizing older turns, and
exposes a `remember` / `recall` scratchpad that outlives transcript pressure.

## Approach

The single-shot `run_agent` becomes `AgentSession.chat(...)`. The internal loop
is the same; what is new is three layers of memory, added in order of
durability:

1. **A running transcript (`self.messages`).** It persists across `chat()`
   calls, so "what carrier is that?" can resolve "that" from the previous turn.
   This is the cheapest memory — the model just sees the prior turns — and the
   most fragile, because it grows without bound and gets trimmed under pressure.
2. **A token-budget policy (`maintain_transcript`).** Run *before* each turn's
   first model call. It is a two-stage cascade: first drop stale tool round
   trips (their bulky JSON payloads are the cheapest tokens to lose), and only
   if still over budget, summarize everything but the last few turns into one
   synthetic turn. The cascade matters — you throw away the least valuable
   tokens first and reach for the lossy summarizer only when you must.
3. **A `remember` / `recall` scratchpad (`self.scratchpad`).** A plain dict
   rendered into the system prompt every turn. Because it lives in the system
   prompt and not the transcript, it survives both trimming and summarization.
   This is where durable facts go — the order ID for this session, a stated
   carrier preference — precisely the facts that must not vanish when the
   transcript collapses.

The key insight the exercise is teaching: **different facts belong in different
memory layers.** A pronoun antecedent from one turn ago can live in the
transcript. A confirmed order ID needs the scratchpad. A policy fact
("opened electronics: 14-day window") belongs in neither — it lives in the KB
and is re-retrieved on demand.

The dispatcher now takes `self` so `remember` and `recall` can mutate
`self.scratchpad`; the read-only tools accept and ignore it.

## Reference implementation

Save as `session.py`. Requires `pip install anthropic numpy` and an exported
`ANTHROPIC_API_KEY`. It imports the tools from exercise-02 conceptually — for a
self-contained file the reference re-declares the minimal set it needs. Run
`python session.py` to drive the follow-up suite plus the scratchpad-durability
turns; it writes `session.jsonl`, `transcript_events.jsonl`, and
`scratchpad.jsonl`.

```python
"""Stateful agent session with token-budget memory and a scratchpad
(mod-105 exercise-03).

Run:  ANTHROPIC_API_KEY=... python session.py
Writes session.jsonl, transcript_events.jsonl, scratchpad.jsonl.
"""
from __future__ import annotations

import json

from anthropic import Anthropic

client = Anthropic()

# --- Pinned parameters (report these in NOTES.md) -------------------------
MODEL = "claude-sonnet-4-6"
TEMPERATURE = 0
DEFAULT_MAX_STEPS = 8
MAX_TOKENS = 1024
SOFT_TOKEN_CAP = 6000          # tuned to force the maintenance pass to fire
SUMMARY_MAX_TOKENS = 400

BASE_SYSTEM = (
    "You are a customer-support assistant with tools to search the knowledge "
    "base, look up orders, issue refunds, and remember/recall session facts. "
    "Before calling a tool, state in one sentence what you intend. When the "
    "user states a fact you expect to need later in the conversation (order "
    "IDs, preferences, confirmations), save it with the `remember` tool."
)

SUMMARIZER_SYSTEM = (
    "Summarize this customer-support conversation for an assistant that needs "
    "to continue helping the customer. Capture: the customer's request, any "
    "order or refund IDs mentioned, any decisions made, and any preferences "
    "they expressed. Use short, factual sentences. Do not omit identifiers."
)

# --- Tool state (subset reused from exercise-02) --------------------------
ORDERS = {
    "ORD-1001": {"status": "shipped", "carrier": "UPS", "sku": "BL-MED-001"},
    "ORD-1002": {"status": "processing", "carrier": "USPS", "sku": "RD-MED-001"},
    "ORD-1003": {"status": "delivered", "carrier": "FedEx", "sku": "HS-001"},
}
REFUNDS: dict[str, dict] = {}
VALID_REASONS = {"damaged", "wrong_item", "not_received", "other"}

KB = {
    "returns_policy.md#1": (
        "Returns policy",
        "Unused items: 30-day return window. Opened electronics: 14-day window "
        "with a 15% restocking fee. Final-sale items are not returnable.",
    ),
    "shipping_policy.md#1": (
        "Shipping policy",
        "Contiguous US: 2-4 business days. Hawaii/Alaska: 5-7 business days.",
    ),
}


def search_kb(_self, query: str, k: int = 5) -> list[dict]:
    """Toy keyword search; replace with the mod-104 retriever from ex-02."""
    q = query.lower()
    hits = []
    for cid, (title, text) in KB.items():
        score = sum(1 for w in q.split() if w in text.lower())
        if score:
            hits.append(
                {
                    "chunk_id": cid,
                    "score": score,
                    "section_title": title,
                    "excerpt": text[:600],
                }
            )
    hits.sort(key=lambda h: -h["score"])
    return hits[:k] or [
        {"chunk_id": cid, "score": 0, "section_title": t, "excerpt": x[:600]}
        for cid, (t, x) in list(KB.items())[:k]
    ]


def get_order_status(_self, order_id: str) -> dict:
    if order_id not in ORDERS:
        return {"error": "not_found"}
    return {"order_id": order_id, **ORDERS[order_id]}


def create_refund(
    _self, order_id: str, amount_cents: int, reason: str, idempotency_key: str
) -> dict:
    if reason not in VALID_REASONS:
        raise ValueError(f"invalid reason: {reason}")
    if idempotency_key in REFUNDS:
        return REFUNDS[idempotency_key]
    record = {
        "refund_id": f"RF-{len(REFUNDS) + 1:04d}",
        "order_id": order_id,
        "amount_cents": amount_cents,
        "reason": reason,
        "status": "created",
    }
    REFUNDS[idempotency_key] = record
    return record


def remember(session, key: str, value: str) -> dict:
    session.scratchpad[key] = value
    return {"ok": True, "key": key}


def recall(session, key: str) -> dict:
    if key in session.scratchpad:
        return {"key": key, "value": session.scratchpad[key]}
    return {"error": "not_found"}


# --- Tool schemas ---------------------------------------------------------
TOOLS = [
    {
        "name": "search_kb",
        "description": (
            "Search the support knowledge base for policy passages. Use for "
            "questions answered by internal docs; not for chit-chat or actions."
        ),
        "input_schema": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "minimum": 1, "maximum": 10},
            },
        },
    },
    {
        "name": "get_order_status",
        "description": (
            "Look up status, carrier, and SKU by order ID. Returns "
            "{\"error\": \"not_found\"} for unknown orders."
        ),
        "input_schema": {
            "type": "object",
            "required": ["order_id"],
            "properties": {"order_id": {"type": "string"}},
        },
    },
    {
        "name": "create_refund",
        "description": (
            "Create a refund. Reason is one of damaged, wrong_item, "
            "not_received, other. Pass a stable idempotency_key."
        ),
        "input_schema": {
            "type": "object",
            "required": ["order_id", "amount_cents", "reason", "idempotency_key"],
            "properties": {
                "order_id": {"type": "string"},
                "amount_cents": {"type": "integer", "minimum": 0},
                "reason": {"type": "string", "enum": sorted(VALID_REASONS)},
                "idempotency_key": {"type": "string"},
            },
        },
    },
    {
        "name": "remember",
        "description": (
            "Save a single fact about the current conversation to a key:value "
            "scratchpad that survives across turns. Use for user-stated facts "
            "you expect to need later: order IDs, preferences, confirmations. "
            "Keys are short slugs (e.g. 'current_order_id'). Overwrites the "
            "value for an existing key."
        ),
        "input_schema": {
            "type": "object",
            "required": ["key", "value"],
            "properties": {
                "key": {"type": "string"},
                "value": {"type": "string"},
            },
        },
    },
    {
        "name": "recall",
        "description": (
            "Read back a fact previously saved with `remember`. Returns the "
            "saved value or {\"error\": \"not_found\"}."
        ),
        "input_schema": {
            "type": "object",
            "required": ["key"],
            "properties": {"key": {"type": "string"}},
        },
    },
]

DISPATCH_TABLE = {
    "search_kb": search_kb,
    "get_order_status": get_order_status,
    "create_refund": create_refund,
    "remember": remember,
    "recall": recall,
}


def dispatch(session, name: str, args: dict):
    fn = DISPATCH_TABLE.get(name)
    if fn is None:
        raise RuntimeError("unknown_tool")
    return fn(session, **args)


# --- The session ----------------------------------------------------------
class AgentSession:
    def __init__(self, tools, dispatch, *, soft_token_cap: int = SOFT_TOKEN_CAP):
        self.tools = tools
        self.dispatch = dispatch
        self.messages: list[dict] = []
        self.scratchpad: dict[str, str] = {}
        self.soft_token_cap = soft_token_cap
        self.events: list[dict] = []   # transcript_events.jsonl rows
        self._turn = 0

    # --- token accounting (ballpark; swap for tiktoken / count_tokens) ----
    def _token_count(self) -> int:
        return len(json.dumps(self.messages, default=str)) // 4

    # --- system prompt rendered fresh each turn with the scratchpad -------
    def _build_system(self) -> str:
        if self.scratchpad:
            lines = [f"- {k}: {v}" for k, v in self.scratchpad.items()]
            memory_block = "Known facts about this session:\n" + "\n".join(lines)
        else:
            memory_block = "No saved session facts yet."
        return BASE_SYSTEM + "\n\n" + memory_block

    # --- maintenance cascade: drop stale, then summarize ------------------
    def maintain_transcript(self) -> None:
        before = self._token_count()
        if before <= self.soft_token_cap:
            self.events.append(
                {
                    "turn": self._turn,
                    "action": "noop",
                    "tokens_before": before,
                    "tokens_after": before,
                    "messages_kept": len(self.messages),
                }
            )
            return

        self._drop_stale_tool_payloads(older_than_turns=5)
        mid = self._token_count()
        if mid <= self.soft_token_cap:
            self.events.append(
                {
                    "turn": self._turn,
                    "action": "drop_stale",
                    "tokens_before": before,
                    "tokens_after": mid,
                    "messages_kept": len(self.messages),
                }
            )
            return

        self._summarize_older(keep_verbatim=4)
        after = self._token_count()
        self.events.append(
            {
                "turn": self._turn,
                "action": "summarize",
                "tokens_before": before,
                "tokens_after": after,
                "messages_kept": len(self.messages),
            }
        )

    def _drop_stale_tool_payloads(self, *, older_than_turns: int) -> None:
        """Replace old tool_use/tool_result blocks with a short placeholder."""
        cutoff = max(0, len(self.messages) - older_than_turns * 2)
        for i, msg in enumerate(self.messages):
            if i >= cutoff:
                break
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            new_content = []
            for block in content:
                btype = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
                if btype in ("tool_use", "tool_result"):
                    name = block.get("name", "?") if isinstance(block, dict) else getattr(block, "name", "?")
                    new_content.append(
                        {"type": "text", "text": f"[tool {name} called, result elided]"}
                    )
                else:
                    new_content.append(block)
            self.messages[i] = {"role": msg["role"], "content": new_content}

    def _summarize_older(self, *, keep_verbatim: int) -> None:
        """Roll all but the last `keep_verbatim` messages into one summary turn."""
        if len(self.messages) <= keep_verbatim:
            return
        older = self.messages[:-keep_verbatim]
        recent = self.messages[-keep_verbatim:]

        resp = client.messages.create(
            model=MODEL,
            max_tokens=SUMMARY_MAX_TOKENS,
            temperature=TEMPERATURE,
            system=SUMMARIZER_SYSTEM,
            messages=[{"role": "user", "content": _render_for_summary(older)}],
        )
        summary_text = "".join(b.text for b in resp.content if b.type == "text")
        self.messages = [
            {
                "role": "user",
                "content": f"[summary of earlier conversation]\n{summary_text}",
            }
        ] + recent

    # --- the per-turn loop (exercise-02's body, now stateful) -------------
    def chat(self, user_message: str, *, max_steps: int = DEFAULT_MAX_STEPS) -> dict:
        self._turn += 1
        self.maintain_transcript()
        self.messages.append({"role": "user", "content": user_message})
        system = self._build_system()

        for step in range(max_steps):
            resp = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                system=system,
                tools=self.tools,
                messages=self.messages,
            )

            if resp.stop_reason == "max_tokens":
                raise RuntimeError(f"max_tokens at step {step}")
            if resp.stop_reason == "end_turn":
                final = "".join(b.text for b in resp.content if b.type == "text")
                self.messages.append({"role": "assistant", "content": resp.content})
                return {
                    "reply": final,
                    "steps": step + 1,
                    "transcript_tokens_after": self._token_count(),
                }

            self.messages.append({"role": "assistant", "content": resp.content})
            tool_results = []
            for b in resp.content:
                if b.type != "tool_use":
                    continue
                try:
                    payload = self.dispatch(self, b.name, b.input)
                except Exception as e:  # noqa: BLE001 - failures become observations
                    payload = {"error": str(e)}
                is_err = isinstance(payload, dict) and "error" in payload
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": b.id,
                        "content": json.dumps(payload),
                        **({"is_error": True} if is_err else {}),
                    }
                )
            self.messages.append({"role": "user", "content": tool_results})

        raise RuntimeError(f"exceeded max_steps={max_steps}")


def _render_for_summary(messages: list[dict]) -> str:
    """Flatten messages into plain text for the summarizer."""
    out = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            out.append(f"{msg['role']}: {content}")
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    out.append(f"{msg['role']}: {block['text']}")
                else:
                    btype = block.get("type") if isinstance(block, dict) else getattr(block, "type", "?")
                    out.append(f"{msg['role']}: [{btype} block]")
    return "\n".join(out)


# --- Drivers --------------------------------------------------------------
FOLLOWUP_SUITE = [
    "Where is order ORD-1001?",
    "What carrier is that?",
    "Can I cancel it now?",
    "Actually I want a refund instead - it arrived broken.",
    "What's your return window on opened electronics?",
    "Got it. So for that headset I ordered - am I inside the window?",
]

CHITCHAT_FILLER = [
    "Thanks, that's helpful.",
    "How's your day going?",
    "Tell me a fun fact about packaging.",
    "Nice. Anything else I should know about your store?",
    "Cool, appreciate it.",
]


def run_session() -> None:
    session = AgentSession(TOOLS, dispatch)

    # Follow-up suite (turns 1-6).
    with open("session.jsonl", "w", encoding="utf-8") as fh:
        for user in FOLLOWUP_SUITE:
            print(f"\n[user] {user}")
            r = session.chat(user)
            print(f"[reply] {r['reply']}")
            fh.write(
                json.dumps(
                    {
                        "user": user,
                        "reply": r["reply"],
                        "steps": r["steps"],
                        "tokens_after": r["transcript_tokens_after"],
                    }
                )
                + "\n"
            )

    # Scratchpad durability (turns 7-9).
    with open("scratchpad.jsonl", "w", encoding="utf-8") as fh:
        # Turn 7: state a preference -> should trigger remember.
        r7 = session.chat("By the way, I prefer FedEx for any future shipments.")
        fh.write(json.dumps({"turn": 7, "user": "prefer FedEx", "reply": r7["reply"],
                             "scratchpad": dict(session.scratchpad)}) + "\n")

        # Turn 8: chit-chat filler to push transcript past the cap.
        for filler in CHITCHAT_FILLER:
            session.chat(filler)

        # Turn 9: ask the preference back after potential summarization.
        r9 = session.chat("Remind me - what carrier did I say I preferred?")
        fh.write(json.dumps({"turn": 9, "user": "which carrier?", "reply": r9["reply"],
                             "scratchpad": dict(session.scratchpad)}) + "\n")

    # Maintenance events from the whole session.
    with open("transcript_events.jsonl", "w", encoding="utf-8") as fh:
        for event in session.events:
            fh.write(json.dumps(event) + "\n")

    print("\nWrote session.jsonl, scratchpad.jsonl, transcript_events.jsonl")


if __name__ == "__main__":
    run_session()
```

## Meeting the acceptance criteria

- **Working `AgentSession` running the 6+ turn suite, `session.jsonl` per
  `(user, reply)` with post-turn token count.** `run_session()` drives
  `FOLLOWUP_SUITE` and writes `user`, `reply`, `steps`, and `tokens_after`.
- **`transcript_events.jsonl` with at least one `drop_stale` and one
  `summarize`.** `maintain_transcript` logs a row every turn (`noop`,
  `drop_stale`, or `summarize`). `SOFT_TOKEN_CAP = 6000` plus the chit-chat
  filler in turn 8 forces both stages to fire across the session. If your run
  only sees `drop_stale`, add one or two more filler turns — the cap is meant to
  be crossed.
- **`scratchpad.jsonl` proving a turn-7 fact is recoverable at turn 9 after
  summarization.** The turn-7 row shows `preferred_carrier` saved; the turn-9
  row shows the model answering "FedEx" with the scratchpad still in
  `session.scratchpad`, which `_build_system` renders into the system prompt
  every turn — so it survives the summarization in turn 8.
- **`NOTES.md`.** Pinned `MODEL`, `SOFT_TOKEN_CAP`, summarizer settings
  (`SUMMARIZER_SYSTEM`, `SUMMARY_MAX_TOKENS`), and `max_steps` are all module
  constants; the three task-6 paragraphs read off `session.jsonl`,
  `transcript_events.jsonl`, and `scratchpad.jsonl`.

## Common pitfalls

1. **Running `maintain_transcript` after appending the new user turn.** It must
   run *before* the first model call of the turn, on the prior transcript;
   running it after means the current turn's own message can get summarized away
   before the model ever sees it. The reference calls it first thing in
   `chat()`.
2. **Rendering the scratchpad once instead of every turn.** Building the system
   prompt at `__init__` freezes an empty scratchpad. `_build_system` is called
   inside `chat()` every turn so newly remembered facts appear immediately —
   this is exactly why the scratchpad survives summarization.
3. **Summarizing the scratchpad instead of the transcript.** The scratchpad
   lives in `self.scratchpad` (system prompt), never in `self.messages`. The
   summarizer only ever touches `self.messages`. If a remembered fact vanishes
   after summarization, the bug is that the fact was only in the transcript and
   the model never called `remember`.
4. **Dropping the recent turns when trimming.** `_drop_stale_tool_payloads`
   elides only blocks older than the last five turns; trim the recent window and
   you destroy the very context "that"/"it" follow-ups depend on.
5. **Raising `soft_token_cap` to stop the maintenance pass from firing.**
   Forcing the policy is the point of the exercise. If it never fires you have
   not tested the thing you built; lower the cap or add filler turns instead.

## Verification

```bash
pip install anthropic numpy
export ANTHROPIC_API_KEY=...
python session.py
```

Then confirm:

```bash
# Six follow-up rows, each with a token count.
wc -l session.jsonl              # expect 6

# At least one drop_stale and one summarize event across the session.
python -c "import json; ev=[json.loads(l) for l in open('transcript_events.jsonl')]; \
acts=[e['action'] for e in ev]; print('actions:', acts); \
assert 'summarize' in acts, 'cap never forced a summarize - lower it or add filler'"

# Turn-7 fact is still present at turn 9.
python -c "import json; rows=[json.loads(l) for l in open('scratchpad.jsonl')]; \
print('turn9 scratchpad:', rows[-1]['scratchpad']); \
print('turn9 reply:', rows[-1]['reply'])"
```

You want six `session.jsonl` rows, a `summarize` action in the events, and the
turn-9 reply naming FedEx with `preferred_carrier` still in the scratchpad. A
no-key syntax check:

```bash
python -c "import ast; ast.parse(open('session.py').read()); print('ok')"
```
