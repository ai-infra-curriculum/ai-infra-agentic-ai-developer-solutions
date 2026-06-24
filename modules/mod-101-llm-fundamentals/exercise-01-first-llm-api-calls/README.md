# mod-101-llm-fundamentals/exercise-01-first-llm-api-calls — Solution

## Approach

The exercise asks for a single CLI program that demonstrates single-turn calls,
a behavior-shaping system prompt, a stateful multi-turn REPL, and the sampling
and length knobs that govern an LLM call. The cleanest way to satisfy all of
that without copying boilerplate is to keep one thin client builder, give each
task its own function, and let `argparse` dispatch between a `single` and a
`chat` mode. Every call funnels through one `_create` helper so the model, the
system prompt, and the sampling parameters are set in exactly one place — that
single choke point is also where we read and print the stop reason on every
response, which builds the habit the exercise wants.

The reference implementation targets Anthropic's Messages API because the
exercise's skeleton does, but the structure ports directly to OpenAI by
swapping the client, the response field names (`stop_reason` →
`finish_reason`, `content[0].text` → `choices[0].message.content`), and the
message-history shape. The API key is read from the environment at construction
time and never appears in source. The chat history lives in a single list that
we reset only at process start (or on an optional `/reset` command), which makes
the "where is history reset and why" reflection answerable by pointing at one
line.

## Reference implementation

```python
#!/usr/bin/env python3
"""exercise-01: first LLM API calls (Anthropic Messages API).

Reflection answers:
1. Chat history is reset exactly once, in `multi_turn`, where `history = []`
   is created before the REPL loop. It lives for the life of one chat session
   so each turn sees every prior turn. `/reset` rebinds it to a new empty list.
2. If you forget to append the assistant message before the next turn, the
   model never sees its own previous answer, so follow-ups that depend on it
   ("And of South Korea?") lose context and the model guesses or asks again.
3. The smallest model that still answered the sample prompts acceptably was
   Haiku; it handled the capital-city follow-ups but was weaker on multi-step
   reasoning. Record your own observation here after running --model swaps.

Usage:
    export ANTHROPIC_API_KEY=...        # never commit this
    python solution.py "Explain TCP in one sentence."
    python solution.py --mode chat
    python solution.py --temperature 0.9 --max-tokens 32 "Write a long poem."
"""

from __future__ import annotations

import argparse
import os
import sys

from anthropic import Anthropic

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_SYSTEM = "Reply in fewer than 30 words. Always end with a period."


def build_client() -> Anthropic:
    """Construct a client from the environment. The key never appears in source."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ANTHROPIC_API_KEY is not set. Export it and retry.")
    return Anthropic(api_key=api_key)


def _create(
    client: Anthropic,
    *,
    model: str,
    system: str | None,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
):
    """Single choke point for every call: one place sets every knob."""
    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages,
    }
    if system:
        kwargs["system"] = system
    return client.messages.create(**kwargs)


def _text(response) -> str:
    """Join the text content blocks of a Messages response."""
    return "".join(block.text for block in response.content if block.type == "text")


def single_turn(
    client: Anthropic,
    model: str,
    system: str | None,
    prompt: str,
    *,
    temperature: float,
    max_tokens: int,
) -> None:
    """Task 1 + 5: one call, print the text and the stop reason."""
    response = _create(
        client,
        model=model,
        system=system,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    print(_text(response))
    # `stop_reason` == "max_tokens" means the answer was truncated by the cap;
    # "end_turn" means the model finished cleanly.
    print(f"[stop_reason={response.stop_reason}]", file=sys.stderr)


def multi_turn(
    client: Anthropic,
    model: str,
    system: str | None,
    *,
    temperature: float,
    max_tokens: int,
) -> None:
    """Task 3: a stateful REPL that maintains context across turns."""
    history: list[dict] = []  # <-- reset point: one session = one history.
    print("Chat mode. Ctrl-D to exit, /reset to clear history.", file=sys.stderr)
    while True:
        try:
            line = input("you> ").strip()
        except EOFError:
            print(file=sys.stderr)  # clean newline on Ctrl-D
            return
        if not line:
            continue
        if line == "/reset":
            history = []  # clear without exiting
            print("[history cleared]", file=sys.stderr)
            continue

        history.append({"role": "user", "content": line})
        response = _create(
            client,
            model=model,
            system=system,
            messages=history,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        reply = _text(response)
        print(f"bot> {reply}")
        print(f"[stop_reason={response.stop_reason}]", file=sys.stderr)
        # Append the assistant turn BEFORE looping so the next user message
        # sees it. Drop this line and follow-up questions lose context.
        history.append({"role": "assistant", "content": reply})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="First LLM API calls.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--mode", choices=["single", "chat"], default="single")
    parser.add_argument(
        "--system",
        default=DEFAULT_SYSTEM,
        help="System prompt, or a path prefixed with @ to load from a file.",
    )
    parser.add_argument("prompt", nargs="?", help="Prompt for --mode single.")
    return parser


def resolve_system(raw: str | None) -> str | None:
    """Stretch goal: `--system @path/to/file` loads the prompt from disk."""
    if raw and raw.startswith("@"):
        with open(raw[1:], encoding="utf-8") as handle:
            return handle.read().strip()
    return raw


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    client = build_client()
    system = resolve_system(args.system)

    if args.mode == "chat":
        multi_turn(
            client,
            args.model,
            system,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        return 0

    if not args.prompt:
        print("error: --mode single requires a prompt argument.", file=sys.stderr)
        return 2

    single_turn(
        client,
        args.model,
        system,
        args.prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

To exercise the sampling-knob task (Task 4), run the same prompt repeatedly and
watch variability collapse at `temperature=0`:

```bash
for i in 1 2 3; do python solution.py --temperature 0 "Name one prime under 10."; done
for i in 1 2 3; do python solution.py --temperature 0.9 "Name one prime under 10."; done
```

## Meeting the acceptance criteria

- **Key read from the environment; never in source.** `build_client` reads
  `os.environ["ANTHROPIC_API_KEY"]` and exits with a clear message if it is
  missing. No literal key string exists anywhere in the file.
- **Single-turn call returns parsed response *and* prints the stop reason.**
  `single_turn` prints `_text(response)` then
  `[stop_reason=...]`. Every call routes through `_create`, so the stop reason
  is always available.
- **System prompt visibly changes behavior across prompts.** `DEFAULT_SYSTEM`
  caps replies at 30 words and forces a trailing period; run two long-answer
  prompts and observe both rules honored.
- **Multi-turn REPL maintains context.** `multi_turn` sends the full `history`
  list every turn and appends the assistant reply before looping, so "What's
  the capital of Japan?" → "And of South Korea?" resolves correctly.
- **Temperature 0 vs 0.9 variability.** The `temperature` flag threads into
  `_create`; the loops above show near-identical output at 0 and varied output
  at 0.9.
- **Constrained `max_tokens` truncates with a matching stop reason.** Run with
  `--max-tokens 32` on a long-answer prompt: the printed `stop_reason` is
  `max_tokens`. Raise it and it becomes `end_turn`.

## Common pitfalls

- **Forgetting to append the assistant turn.** The most common multi-turn bug.
  Without the `history.append({"role": "assistant", ...})` line, the model never
  sees its own prior answers and follow-ups fail. The comment in `multi_turn`
  flags exactly this line.
- **Hardcoding the key or printing it for debugging.** Read it once from the
  environment and never echo it. A leaked key in shell history or a log is a
  real incident.
- **Assuming `content[0].text` always exists.** A Messages response can contain
  multiple blocks (and non-text block types later). `_text` iterates and filters
  on `block.type == "text"` instead of indexing `[0]`.
- **Treating `temperature=0` as a guarantee of identical output.** It produces
  *highly similar*, often identical, output — not a hard determinism contract.
  The exercise's wording ("highly similar (often identical)") is deliberate.
- **Confusing `max_tokens` (output cap) with the context window (input + output
  limit).** Setting `--max-tokens 32` truncates the *reply*, not the prompt.

## Verification

```bash
pip install anthropic
export ANTHROPIC_API_KEY=...        # your funded key, small spend cap

# Single-turn + stop reason on stderr
python solution.py "Explain TCP in one sentence."

# System prompt enforces the 30-word / trailing-period rule
python solution.py "Tell me everything about the Roman Empire."

# Multi-turn context check
printf "What is the capital of Japan?\nAnd of South Korea?\n" | python solution.py --mode chat

# Truncation: stop_reason should print max_tokens
python solution.py --max-tokens 32 "Write a 500-word essay on rivers."
```

A green run shows: text on stdout, `[stop_reason=...]` on stderr, the system
rule honored, the follow-up answered as "Seoul", and `max_tokens` as the stop
reason on the constrained call.
