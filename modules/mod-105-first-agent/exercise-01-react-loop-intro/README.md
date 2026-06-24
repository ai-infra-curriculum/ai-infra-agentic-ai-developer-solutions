# mod-105-first-agent/exercise-01 — Solution

This is the reference solution for the minimal ReAct loop: two toy tools, a
step cap, errors surfaced as observations, and a printed Thought/Action trace
for every tool-using step.

## Approach

The exercise asks for the smallest honest agent: a single `run_agent` function
that loops `model → tool → model` until the model emits a final answer or a
guardrail fires. Nothing is hidden behind a class or a framework — the whole
protocol is visible in one file so you can point at every line and say why it
is there.

Four design decisions carry the exercise:

1. **The loop is bounded.** `max_steps=8` caps the number of model calls. The
   loop is a `for step in range(max_steps)` — when it falls out the bottom it
   raises, rather than running forever. A step cap is the single most important
   guardrail in any agent; without it a confused model can spin until your bill
   does.
2. **Stop reasons are handled explicitly.** Every turn checks
   `resp.stop_reason`. `end_turn` returns the final text; `max_tokens` raises
   (a truncated turn is a bug, not a result); anything else means the model
   wants to call tools and we continue the loop. Nothing is left to fall
   through silently.
3. **Tool errors become observations, not crashes.** The dispatcher catches
   every exception and returns `{"error": str(e)}` with `is_error: True` on the
   `tool_result`. The model sees the error in its next turn and recovers in
   natural language. An exception that escapes the dispatcher kills the whole
   trajectory; an error fed back as an observation is just another thing the
   model reasons about.
4. **Every step is traced.** The system prompt nudges the model to write a
   one-sentence Thought before each call, and the loop prints
   `[step N] Thought:` and `[step N] Action:` lines. The same data is appended
   to a structured `trace` list and written to `runs.jsonl`. You cannot debug
   what you cannot see.

The two tools are deliberately tiny — one read-only (`get_weather`), one
mutating (`set_thermostat`) — because the point is to exercise routing,
multi-step chains, and error recovery, not to build a broad surface.

## Reference implementation

Save as `agent.py`. Requires `pip install anthropic` and an exported
`ANTHROPIC_API_KEY`. Run `python agent.py` to execute the full suite plus the
two adversarial probes; it writes `runs.jsonl` and `adversarial.jsonl`.

```python
"""Minimal ReAct loop over two toy tools (mod-105 exercise-01).

Run:  ANTHROPIC_API_KEY=... python agent.py
Writes runs.jsonl (suite) and adversarial.jsonl (guardrail probes).
"""
from __future__ import annotations

import json

from anthropic import Anthropic

client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment

# --- Pinned parameters (report these in NOTES.md) -------------------------
MODEL = "claude-sonnet-4-6"
TEMPERATURE = 0
DEFAULT_MAX_STEPS = 8
MAX_TOKENS = 1024

SYSTEM = (
    "You are a helpful assistant. You may call tools to look up information or "
    "perform actions. Before calling a tool, briefly state in one sentence what "
    "you are looking for and which tool you intend to use. After receiving a "
    "tool result, briefly state what you learned before deciding the next step."
)

# --- Tool implementations -------------------------------------------------
WEATHER = {
    "Seattle": {"tempF": 54, "conditions": "rain"},
    "Phoenix": {"tempF": 102, "conditions": "sun"},
    "Boston": {"tempF": 41, "conditions": "snow"},
    "Tokyo": {"tempF": 68, "conditions": "cloudy"},
}

THERMOSTAT_LOG: list[dict] = []

MIN_TARGET_F = 60
MAX_TARGET_F = 80


def get_weather(city: str) -> dict:
    """Return a fake weather record for a known city; raise if unknown."""
    if city not in WEATHER:
        raise ValueError(f"unknown city: {city}")
    return {"city": city, **WEATHER[city]}


def set_thermostat(target_f: int, note: str | None = None) -> dict:
    """Record a thermostat change; raise if target is outside 60-80F."""
    if not (MIN_TARGET_F <= target_f <= MAX_TARGET_F):
        raise ValueError(
            f"target_f {target_f} out of range "
            f"({MIN_TARGET_F}-{MAX_TARGET_F})"
        )
    THERMOSTAT_LOG.append({"target_f": target_f, "note": note})
    return {"ok": True, "set_to_f": target_f}


# --- Tool schemas ---------------------------------------------------------
GET_WEATHER_TOOL = {
    "name": "get_weather",
    "description": (
        "Return current weather (temperature in F, conditions) for a known "
        "city. Use for any weather-related question. Raises if the city is "
        "unknown."
    ),
    "input_schema": {
        "type": "object",
        "required": ["city"],
        "properties": {
            "city": {"type": "string", "description": "City name, e.g. 'Seattle'."}
        },
    },
}

SET_THERMOSTAT_TOOL = {
    "name": "set_thermostat",
    "description": (
        "Set the thermostat to a target temperature in Fahrenheit. Use when "
        "the user explicitly asks to change the temperature. Acceptable range "
        "is 60-80F."
    ),
    "input_schema": {
        "type": "object",
        "required": ["target_f"],
        "properties": {
            "target_f": {"type": "integer", "description": "Target temperature in F."},
            "note": {"type": "string", "description": "Optional reason for the change."},
        },
    },
}

TOOLS = [GET_WEATHER_TOOL, SET_THERMOSTAT_TOOL]


# --- Dispatcher: errors become data, never exceptions that escape ---------
def dispatch(name: str, args: dict) -> dict:
    """Route a tool call to its implementation by name."""
    if name == "get_weather":
        return get_weather(**args)
    if name == "set_thermostat":
        return set_thermostat(**args)
    # Unknown tool is itself an error-as-observation, not a crash.
    raise RuntimeError("unknown_tool")


# --- The loop -------------------------------------------------------------
def run_agent(
    user_message: str,
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
    seed_messages: list[dict] | None = None,
) -> dict:
    """Run a one-shot ReAct loop over a single user message.

    Returns {"final_text", "trace", "steps"}. Raises on max_tokens or when the
    step cap is exceeded. `seed_messages` lets a caller pre-inject turns (used
    by the unknown-tool adversarial probe).
    """
    messages: list[dict] = list(seed_messages or [])
    messages.append({"role": "user", "content": user_message})
    trace: list[dict] = []

    for step in range(max_steps):
        resp = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            system=SYSTEM,
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

        # max_tokens means the turn was truncated mid-thought: a bug, not a result.
        if resp.stop_reason == "max_tokens":
            trace.append(step_record)
            raise RuntimeError(f"hit max_tokens at step {step}")

        # end_turn means the model is done: return its final answer.
        if resp.stop_reason == "end_turn":
            trace.append(step_record)
            return {"final_text": text, "trace": trace, "steps": step + 1}

        # Otherwise the model wants to act. Echo its turn, then run the tools.
        messages.append({"role": "assistant", "content": resp.content})

        tool_results = []
        observations = []
        for b in tool_uses:
            try:
                payload = dispatch(b.name, b.input)
            except Exception as e:  # noqa: BLE001 - every failure is an observation
                payload = {"error": str(e)}
            observations.append({"tool_use_id": b.id, "observation": payload})
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": b.id,
                    "content": json.dumps(payload),
                    **({"is_error": True} if "error" in payload else {}),
                }
            )

        step_record["observations"] = observations
        trace.append(step_record)
        messages.append({"role": "user", "content": tool_results})

    raise RuntimeError(f"exceeded max_steps={max_steps}")


# --- Suites ---------------------------------------------------------------
SUITE = [
    "What's the weather in Seattle?",
    "How does Phoenix compare to Boston right now?",
    "Set the thermostat to 72 because it's cold.",
    "It's snowing in Boston - turn the heat up a little for me.",
    "Hi there.",
    "What's the weather in Atlantis?",
    "Set the thermostat to 95.",
]


def run_suite(path: str = "runs.jsonl") -> None:
    """Run every suite prompt and append one JSONL row per prompt."""
    with open(path, "w", encoding="utf-8") as fh:
        for prompt in SUITE:
            print(f"\n=== {prompt} ===")
            try:
                result = run_agent(prompt)
                row = {
                    "prompt": prompt,
                    "steps": result["steps"],
                    "trace": result["trace"],
                    "final_text": result["final_text"],
                }
            except RuntimeError as e:
                # A guardrail fired; record the partial run rather than crashing.
                row = {"prompt": prompt, "error": str(e)}
            fh.write(json.dumps(row) + "\n")


def run_adversarial(path: str = "adversarial.jsonl") -> None:
    """Two probes that should make the guardrails fire."""
    rows = []

    # Probe 1: step-cap. A long chain of calls should exceed a tight cap.
    step_probe = (
        "Tell me the weather in Seattle, then Phoenix, then Boston, then Tokyo, "
        "then Seattle again, then Phoenix again, then write a haiku about all four."
    )
    print(f"\n=== step-cap probe (max_steps=4) ===")
    try:
        result = run_agent(step_probe, max_steps=4)
        rows.append(
            {
                "probe": "step_cap",
                "outcome": "completed_unexpectedly",
                "steps": result["steps"],
                "final_text": result["final_text"],
            }
        )
    except RuntimeError as e:
        rows.append(
            {
                "probe": "step_cap",
                "outcome": "guardrail_fired",
                "error": str(e),
                "note": "Step cap fired before the model finished the chain.",
            }
        )

    # Probe 2: unknown tool. Pre-inject an assistant turn calling summon_dragon.
    print(f"\n=== unknown-tool probe ===")
    seed = [
        {
            "role": "user",
            "content": "Please summon a dragon for me.",
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_probe_dragon",
                    "name": "summon_dragon",
                    "input": {"size": "large"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_probe_dragon",
                    "content": json.dumps(_safe_dispatch("summon_dragon", {})),
                    "is_error": True,
                }
            ],
        },
    ]
    result = run_agent(
        "Given that error, please respond to me.", seed_messages=seed
    )
    rows.append(
        {
            "probe": "unknown_tool",
            "outcome": "recovered",
            "dispatcher_result": _safe_dispatch("summon_dragon", {}),
            "final_text": result["final_text"],
            "note": "Dispatcher returned an error observation; the model recovered.",
        }
    )

    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _safe_dispatch(name: str, args: dict) -> dict:
    """Dispatch but always return a dict, mirroring the in-loop error path."""
    try:
        return dispatch(name, args)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


if __name__ == "__main__":
    run_suite()
    run_adversarial()
    print("\nWrote runs.jsonl and adversarial.jsonl")
```

The `_safe_dispatch` helper exists only so the unknown-tool probe can show the
exact observation the dispatcher produces (`{"error": "unknown_tool"}`) inside
the seeded transcript — the in-loop path uses the same try/except.

## Meeting the acceptance criteria

- **Runs all seven suite prompts, writes `runs.jsonl`.** `run_suite()` iterates
  `SUITE` and writes one row per prompt with `prompt`, `steps`, `trace`, and
  `final_text`.
- **Thought + Action line per tool-using step.** The loop prints
  `[step N] Thought:` whenever the turn has text and `[step N] Action:` for each
  `tool_use` block.
- **`MODEL`, `TEMPERATURE`, `max_steps` pinned and reportable.** All three are
  module constants (`MODEL`, `TEMPERATURE`, `DEFAULT_MAX_STEPS`); quote them in
  `NOTES.md`.
- **`adversarial.jsonl` with the two probes.** `run_adversarial()` writes a
  `step_cap` row (guardrail fired) and an `unknown_tool` row (dispatcher
  returned `{"error": "unknown_tool"}` and the model recovered).
- **`NOTES.md` content.** The trace list and printed lines give you everything
  the three task-5 paragraphs and the readability observation need; the pinned
  parameters are the constants above.

The structured `trace` (each step records `text`, `tool_uses`, and
`observations`) is what you read back to write the task-5 paragraphs: the pivot
Thought, the post-error Thought on prompts 6 and 7, and whether prompt 2 issued
one assistant turn with two `tool_use` blocks (parallel) or two turns (serial).

## Common pitfalls

1. **Forgetting to echo the assistant turn before sending tool results.** The
   API requires the `assistant` message containing the `tool_use` blocks to be
   in `messages` before you append the matching `tool_result` blocks. Skip it
   and you get a 400 about an unmatched `tool_use_id`. The loop appends
   `{"role": "assistant", "content": resp.content}` immediately after deciding
   to continue.
2. **Letting a tool exception escape the dispatcher.** If `get_weather` raises
   and nothing catches it, the whole trajectory dies on prompt 6. The bare
   `except Exception` around `dispatch(...)` is intentional here — every failure
   must come back as an observation so the model can apologize gracefully.
3. **Raising `max_steps` to make the adversarial prompt "pass."** That defeats
   the guardrail you are testing. If the step-cap probe batches its calls in
   parallel and finishes under 8, lower `max_steps` (the probe uses 4) — do not
   raise it.
4. **Treating `max_tokens` as a normal stop.** A `max_tokens` turn is a
   truncated thought, not a final answer. Returning its partial text as the
   result silently corrupts the run; the loop raises instead.
5. **Missing `is_error` on error `tool_result` blocks.** Omitting it still
   works, but flagging errors helps the model (and your trace reader)
   distinguish a real result from a failure. The loop sets it whenever the
   payload has an `error` key.

## Verification

```bash
pip install anthropic
export ANTHROPIC_API_KEY=...   # a real key; spend stays well under a dollar
python agent.py
```

Then confirm:

```bash
# Seven suite rows, each with a prompt and either steps or an error.
wc -l runs.jsonl                 # expect 7

# Every tool-using row carries a trace; prompt 5 ("Hi there.") should have 1 step.
python -c "import json; rows=[json.loads(l) for l in open('runs.jsonl')]; \
print([(r['prompt'][:24], r.get('steps')) for r in rows])"

# Adversarial: step-cap fired, unknown-tool recovered.
python -c "import json; rows=[json.loads(l) for l in open('adversarial.jsonl')]; \
print([(r['probe'], r['outcome']) for r in rows])"
```

You should see the `Thought:` / `Action:` lines stream to stdout during the
run, `runs.jsonl` with seven rows, and `adversarial.jsonl` showing
`('step_cap', 'guardrail_fired')` and `('unknown_tool', 'recovered')`. A quick
syntax check without an API key:

```bash
python -c "import ast; ast.parse(open('agent.py').read()); print('ok')"
```
