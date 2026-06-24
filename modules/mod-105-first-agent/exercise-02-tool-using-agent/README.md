# mod-105-first-agent/exercise-02 — Solution

This is the reference solution for the three-tool support agent: the `mod-104`
retriever wired in as `search_kb` alongside `get_order_status` and
`create_refund`, all routed through one dispatcher, with citation validation
on top.

## Approach

The loop does not change from exercise-01. What changes is the tool surface and
the dispatcher — the whole exercise is a demonstration that adding capability to
a ReAct agent means adding tools, not rewriting the loop.

Three ideas carry it:

1. **Retrieval is just another tool.** `search_kb` wraps the `mod-104`
   retriever (`embed_query` + `search`) and returns a list of
   `{chunk_id, score, section_title, excerpt}`. The model decides when to call
   it based on the description, exactly as it decides whether to call
   `get_order_status`. There is no special "RAG mode" — retrieval competes for
   the model's attention against the action tools, and the description is what
   wins that competition.
2. **Errors return as data where it reads better.** `get_order_status` returns
   `{"error": "not_found"}` for unknown orders instead of raising. Compare with
   exercise-01's `get_weather`, which raised: a "not found" lookup is a normal
   business outcome the model should reason about, not an exception. The
   dispatcher still wraps everything so a genuine bug surfaces as an observation
   too.
3. **Citations are validated after the fact.** The system prompt asks the model
   to cite chunks as `[chunk_id: <id>]`. A separate `validate_citations`
   function cross-checks every cited id against the chunks the agent actually
   retrieved (recorded in the trace). A cited id that was never retrieved is a
   **fabrication** — the single most important finding in a grounded support
   agent. The trace records each `search_kb` observation precisely so this check
   is possible.

To keep the solution self-contained and runnable without standing up a
SQLite/NumPy index, the reference embeds the small support corpus once at import
time into an in-memory NumPy matrix and does brute-force cosine search. The
`embed_query` / `search` seam is identical to `mod-104` exercise-02; swap in your
persistent index and nothing else changes.

## Reference implementation

Save as `agent.py`. Requires `pip install anthropic numpy` and an exported
`ANTHROPIC_API_KEY`. The reference uses a deterministic local embedder behind
`embed_texts` so the file runs with no external index; if your `mod-104`
retriever used a real embedder, drop it in there. Run `python agent.py` for the
routing suite plus the citation re-runs.

```python
"""Three-tool support agent with retrieval and citation validation
(mod-105 exercise-02).

Run:  ANTHROPIC_API_KEY=... python agent.py
Writes routing.jsonl and citations.jsonl.
"""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np
from anthropic import Anthropic

client = Anthropic()

# --- Pinned parameters (report these in NOTES.md) -------------------------
MODEL = "claude-sonnet-4-6"
TEMPERATURE = 0
DEFAULT_MAX_STEPS = 10
MAX_TOKENS = 1024
EMBED_DIM = 256          # deterministic local embedder; swap for mod-104's
CORPUS_ID = "mod105-support-corpus-v1"

BASE_SYSTEM = (
    "You are a customer-support assistant. You may call tools to search the "
    "knowledge base, look up orders, and issue refunds. Before calling a tool, "
    "state in one sentence what you are looking for and which tool you intend "
    "to use. After a tool result, state what you learned before the next step. "
    "When your answer relies on a knowledge-base passage, cite it inline as "
    "[chunk_id: <id>] using the chunk_id from the search result. Do not cite a "
    "chunk you did not retrieve."
)

# --- Corpus ---------------------------------------------------------------
CORPUS = {
    "returns_policy.md#1": (
        "Returns policy",
        "We accept returns of unused items within 30 days of delivery. "
        "Opened electronics: 14-day return window with a 15% restocking fee "
        "(clause 3). Final-sale items are not returnable.",
    ),
    "shipping_policy.md#1": (
        "Shipping policy",
        "Standard shipping to the contiguous US: 2-4 business days. Shipping to "
        "Hawaii and Alaska (non-contiguous US states): 5-7 business days. "
        "International orders ship via UPS Worldwide with tracking included.",
    ),
    "refunds_policy.md#1": (
        "Refunds policy",
        "Refunds appear on the original payment method within 5-10 business "
        "days. For damaged goods, we will refund the full purchase price and "
        "return shipping.",
    ),
}


# --- Embedding seam (replace with your mod-104 embedder) ------------------
def _hash_embed(text: str) -> np.ndarray:
    """Deterministic bag-of-words hashing embedder.

    Stands in for a real embedding model so this file runs with no external
    index. It ranks these three chunks sensibly. Replace `embed_texts` with
    your mod-104 `embed_query`-backed embedder for real use.
    """
    vec = np.zeros(EMBED_DIM, dtype=np.float32)
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        vec[h % EMBED_DIM] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


def embed_texts(texts: list[str]) -> np.ndarray:
    return np.vstack([_hash_embed(t) for t in texts])


def embed_query(query: str) -> np.ndarray:
    return _hash_embed(query)


# Build the index once at import (mirrors mod-104 exercise-02).
_CHUNK_IDS = list(CORPUS.keys())
_MATRIX = embed_texts([CORPUS[cid][1] for cid in _CHUNK_IDS])


def search(q_vec: np.ndarray, k: int) -> list[dict]:
    """Brute-force cosine search over the in-memory matrix."""
    scores = _MATRIX @ q_vec
    order = np.argsort(-scores)[:k]
    out = []
    for idx in order:
        cid = _CHUNK_IDS[idx]
        title, text = CORPUS[cid]
        out.append(
            {
                "chunk_id": cid,
                "score": float(scores[idx]),
                "section_title": title,
                "text": text,
            }
        )
    return out


# --- Action-tool state ----------------------------------------------------
ORDERS = {
    "ORD-1001": {"status": "shipped", "carrier": "UPS", "sku": "BL-MED-001"},
    "ORD-1002": {"status": "processing", "carrier": "USPS", "sku": "RD-MED-001"},
    "ORD-1003": {"status": "delivered", "carrier": "FedEx", "sku": "HS-001"},
}

REFUNDS: dict[str, dict] = {}
VALID_REASONS = {"damaged", "wrong_item", "not_received", "other"}


# --- Tool implementations -------------------------------------------------
def search_kb(query: str, k: int = 5) -> list[dict]:
    """Return up to k chunks; excerpt truncated to ~600 chars."""
    chunks = search(embed_query(query), min(k, 10))
    return [
        {
            "chunk_id": c["chunk_id"],
            "score": round(c["score"], 3),
            "section_title": c["section_title"],
            "excerpt": c["text"][:600],
        }
        for c in chunks
    ]


def get_order_status(order_id: str) -> dict:
    """Look up an order; return the error as data, not an exception."""
    if order_id not in ORDERS:
        return {"error": "not_found"}
    return {"order_id": order_id, **ORDERS[order_id]}


def create_refund(
    order_id: str, amount_cents: int, reason: str, idempotency_key: str
) -> dict:
    """Idempotent refund creator keyed by idempotency_key."""
    if reason not in VALID_REASONS:
        raise ValueError(f"invalid reason: {reason}")
    if idempotency_key in REFUNDS:
        return REFUNDS[idempotency_key]  # idempotent replay
    record = {
        "refund_id": f"RF-{len(REFUNDS) + 1:04d}",
        "order_id": order_id,
        "amount_cents": amount_cents,
        "reason": reason,
        "idempotency_key": idempotency_key,
        "status": "created",
    }
    REFUNDS[idempotency_key] = record
    return record


# --- Tool schemas ---------------------------------------------------------
SEARCH_KB_TOOL = {
    "name": "search_kb",
    "description": (
        "Search the customer support knowledge base for passages relevant to a "
        "natural-language query. Use this when the user is asking a question "
        "whose answer is likely in our internal documentation (policies, FAQs, "
        "product manuals). Do NOT use it for chit-chat, math, or actions like "
        "placing an order. If the top hit has a weak score (< 0.6) and the "
        "answer is unclear, refine the query and search again. Returns up to 5 "
        "passages with chunk_id, score, section_title, and an excerpt."
    ),
    "input_schema": {
        "type": "object",
        "required": ["query"],
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "A focused natural-language question or phrase. Prefer "
                    "keyword-rich rephrasings of the user's words over the "
                    "literal user text. On a weak first result, try synonyms "
                    "(e.g. 'opened electronics' instead of 'headphones')."
                ),
            },
            "k": {"type": "integer", "minimum": 1, "maximum": 10},
        },
    },
}

GET_ORDER_STATUS_TOOL = {
    "name": "get_order_status",
    "description": (
        "Look up the status, carrier, and SKU of an order by its order ID "
        "(e.g. 'ORD-1001'). Use only when the user gives a concrete order ID. "
        "Returns {\"error\": \"not_found\"} for unknown orders."
    ),
    "input_schema": {
        "type": "object",
        "required": ["order_id"],
        "properties": {
            "order_id": {"type": "string", "description": "Order ID like 'ORD-1001'."}
        },
    },
}

CREATE_REFUND_TOOL = {
    "name": "create_refund",
    "description": (
        "Create a refund for an order. Use only after confirming the order and "
        "the reason. Reason must be one of: damaged, wrong_item, not_received, "
        "other. Pass a stable idempotency_key (e.g. the order ID plus reason) so "
        "retries do not double-refund."
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
}

TOOLS = [SEARCH_KB_TOOL, GET_ORDER_STATUS_TOOL, CREATE_REFUND_TOOL]

# --- Single dispatcher: name -> callable ----------------------------------
DISPATCH_TABLE = {
    "search_kb": search_kb,
    "get_order_status": get_order_status,
    "create_refund": create_refund,
}


def dispatch(name: str, args: dict):
    fn = DISPATCH_TABLE.get(name)
    if fn is None:
        raise RuntimeError("unknown_tool")
    return fn(**args)


# --- The loop (unchanged from exercise-01 except TOOLS and trace shape) ----
def run_agent(user_message: str, *, max_steps: int = DEFAULT_MAX_STEPS) -> dict:
    """One-shot ReAct loop; records each search_kb observation in the trace."""
    messages = [{"role": "user", "content": user_message}]
    trace: list[dict] = []

    for step in range(max_steps):
        resp = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            system=BASE_SYSTEM,
            tools=TOOLS,
            messages=messages,
        )

        text = "".join(b.text for b in resp.content if b.type == "text")
        tool_uses = [b for b in resp.content if b.type == "tool_use"]

        if text:
            print(f"[step {step}] Thought: {text}")
        for b in tool_uses:
            print(f"[step {step}] Action:  {b.name}({b.input})")

        step_record = {
            "step": step,
            "stop_reason": resp.stop_reason,
            "text": text,
            "tool_uses": [
                {"id": b.id, "name": b.name, "input": b.input} for b in tool_uses
            ],
        }

        if resp.stop_reason == "max_tokens":
            trace.append(step_record)
            raise RuntimeError(f"hit max_tokens at step {step}")
        if resp.stop_reason == "end_turn":
            trace.append(step_record)
            return {"final_text": text, "trace": trace, "steps": step + 1}

        messages.append({"role": "assistant", "content": resp.content})

        tool_results = []
        for b in tool_uses:
            try:
                payload = dispatch(b.name, b.input)
            except Exception as e:  # noqa: BLE001 - failures become observations
                payload = {"error": str(e)}
            # Record the observation on the matching tool_use for validation.
            for tu in step_record["tool_uses"]:
                if tu["id"] == b.id:
                    tu["observation"] = payload
            is_err = isinstance(payload, dict) and "error" in payload
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": b.id,
                    "content": json.dumps(payload),
                    **({"is_error": True} if is_err else {}),
                }
            )

        trace.append(step_record)
        messages.append({"role": "user", "content": tool_results})

    raise RuntimeError(f"exceeded max_steps={max_steps}")


def tools_called(trace: list[dict]) -> list[dict]:
    """Flatten the trace into an ordered list of {name, input} tool calls."""
    calls = []
    for step in trace:
        for tu in step.get("tool_uses", []):
            calls.append({"name": tu["name"], "input": tu["input"]})
    return calls


# --- Citation validation --------------------------------------------------
def validate_citations(final_text: str, trace: list[dict]) -> dict:
    """Cross-check cited chunk_ids against chunks the agent actually retrieved."""
    cited = set(re.findall(r"\[chunk_id:\s*([\w\-#\.]+)\]", final_text))
    seen: set[str] = set()
    for step in trace:
        for tu in step.get("tool_uses", []):
            if tu["name"] != "search_kb":
                continue
            for chunk in tu.get("observation", []) or []:
                if isinstance(chunk, dict) and "chunk_id" in chunk:
                    seen.add(chunk["chunk_id"])
    return {
        "cited": sorted(cited),
        "seen": sorted(seen),
        "fabricated": sorted(cited - seen),
        "uncited_relevant": sorted(seen - cited),
    }


# --- Suites ---------------------------------------------------------------
ROUTING_SUITE = [
    "What's your return policy on opened headphones?",
    "Where is order ORD-1001?",
    "Can I cancel an order that already shipped?",
    "I need a refund on ORD-1002 - it arrived broken.",
    "Hi there!",
    "What's the SKU on ORD-1003 and is that item covered by free returns?",
    "How long does shipping to Hawaii take, and is my order ORD-1002 going there?",
]

CITATION_RERUN_INDICES = [0, 2, 3, 5, 6]  # prompts 1, 3, 4, 6, 7


def run_routing(path: str = "routing.jsonl") -> list[dict]:
    results = []
    with open(path, "w", encoding="utf-8") as fh:
        for prompt in ROUTING_SUITE:
            print(f"\n=== {prompt} ===")
            try:
                r = run_agent(prompt)
                row = {
                    "prompt": prompt,
                    "trace": r["trace"],
                    "tools_called": tools_called(r["trace"]),
                    "final_text": r["final_text"],
                }
            except RuntimeError as e:
                row = {"prompt": prompt, "error": str(e)}
            results.append(row)
            fh.write(json.dumps(row) + "\n")
    return results


def run_citations(path: str = "citations.jsonl") -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for idx in CITATION_RERUN_INDICES:
            prompt = ROUTING_SUITE[idx]
            print(f"\n=== [citations] {prompt} ===")
            r = run_agent(prompt)
            validation = validate_citations(r["final_text"], r["trace"])
            row = {
                "prompt": prompt,
                "final_text": r["final_text"],
                "validation": validation,
            }
            if validation["fabricated"]:
                print(f"  !! FABRICATED: {validation['fabricated']}")
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    run_routing()
    run_citations()
    print("\nWrote routing.jsonl and citations.jsonl")
```

## Meeting the acceptance criteria

- **Three tools, one dispatcher, loop unchanged.** `DISPATCH_TABLE` maps names
  to callables; `run_agent` is exercise-01's loop with a different `TOOLS` list
  and the per-call observation recorded on the trace.
- **`routing.jsonl` over the seven prompts.** `run_routing()` writes a row with
  `trace`, `tools_called`, and `final_text` for each suite prompt.
- **`iterative.jsonl`.** The `search_kb` description's `query` parameter already
  tells the model to retry weak hits with synonyms ("opened electronics" instead
  of "headphones"). For the iterative task, run prompt 1, capture the first and
  (if any) second `search_kb` query from the trace, and write them side by side;
  if the model answered after one weak hit, the description change to make is in
  that `query` field — record the before/after there.
- **`citations.jsonl` with the validator output for five prompts.**
  `run_citations()` re-runs prompts 1, 3, 4, 6, 7 and writes
  `validate_citations(...)` output. Zero `fabricated` is the bar; any non-empty
  `fabricated` list is printed and saved as a finding rather than hidden.
- **`parallelism.jsonl`.** Prompt 7 is two unrelated lookups. Inspect its trace:
  if both `tool_use` blocks appear in one assistant turn (`step` 0 has two
  entries in `tool_uses`), the calls were parallel. Re-run measuring step count,
  model-call count, and wall-clock to fill `parallelism.jsonl`.
- **`NOTES.md`.** The pinned `MODEL`, `TEMPERATURE`, `max_steps`, embedding
  model, and `CORPUS_ID` are all module constants; the before/after for the
  tightened description is the line you edit in task 6.

## Common pitfalls

1. **Not recording the `search_kb` observation in the trace.** The citation
   validator needs the list of retrieved chunks, not just the call's arguments.
   The loop writes `tu["observation"] = payload` onto the matching `tool_use`;
   without it, `validate_citations` sees an empty `seen` set and every real
   citation looks fabricated.
2. **Making `get_order_status` raise on unknown orders.** A "not found" lookup
   is a normal outcome the model should explain to the user. Returning
   `{"error": "not_found"}` as data keeps the trajectory alive; raising forces
   the dispatcher's generic error path and produces a worse answer.
3. **A non-idempotent `create_refund`.** If the model retries the refund (it
   sometimes does after a confused turn) and the key is not respected, you
   double-refund. The reference keys `REFUNDS` by `idempotency_key` and replays
   the existing record on a repeat.
4. **Over-tuning the system prompt instead of the tool description.** The
   highest-leverage knob in this exercise is the `search_kb` description (task 6
   asks you to edit *only* a description). Reaching for the system prompt first
   hides which change actually moved behavior.
5. **Confusing `seen` vs. `uncited_relevant`.** `fabricated` (cited but never
   retrieved) is the real defect. `uncited_relevant` (retrieved but not cited)
   is usually fine — the model cited the strongest chunk and ignored a weaker
   one. Only `fabricated` should block a response.

## Verification

```bash
pip install anthropic numpy
export ANTHROPIC_API_KEY=...
python agent.py
```

Then confirm:

```bash
# Seven routing rows.
wc -l routing.jsonl              # expect 7

# Routing sanity: prompt 2 should call get_order_status, prompt 5 no tools.
python -c "import json; rows=[json.loads(l) for l in open('routing.jsonl')]; \
print([(r['prompt'][:20], [c['name'] for c in r.get('tools_called',[])]) for r in rows])"

# Zero fabricated citations across the five re-runs is the bar.
python -c "import json; rows=[json.loads(l) for l in open('citations.jsonl')]; \
print('fabricated:', [r['validation']['fabricated'] for r in rows])"
```

You want `get_order_status` in prompt 2's calls, an empty `tools_called` for
prompt 5, and every `fabricated` list empty. A no-key syntax check:

```bash
python -c "import ast; ast.parse(open('agent.py').read()); print('ok')"
```
