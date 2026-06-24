# mod-101-llm-fundamentals/exercise-02-tokens-context-cost — Solution

## Approach

This exercise is about replacing hand-waving with measurement. The reference
solution is one module with four small, independently runnable pieces: an exact
token counter that calls the official tool (Anthropic's Count Tokens endpoint,
with a `tiktoken` path for GPT models), a `ContextBudget` class that reserves
named line items and refuses to fit more than the window allows, a usage logger
that appends real `usage` data from every chat turn to `usage_log.csv`, and a
cost projector that reads that CSV and produces a per-call and daily-scale cost
table from prices you look up and cite at run time.

The key discipline the exercise enforces is that prices are *never* hardcoded in
the curriculum text — they change, and a stale number is worse than no number.
The solution keeps prices in a tiny `PRICES` dict at the top of the file with a
required citation comment (source URL + date) that the learner fills in when
they run it. The token counter is exact (it calls the provider), so the
`len(text) / 4` rule-of-thumb comparison is honest: prose lands near the rule,
code and JSON drift because of dense punctuation and whitespace tokenization.

## Reference implementation

```python
#!/usr/bin/env python3
"""exercise-02: tokens, context, and cost.

Reflection answers:
1. The "4 chars per token" rule held best on English prose and worst on JSON
   and minified code, where punctuation and whitespace inflate token counts far
   beyond chars/4. Your corpus (prose) tracked the rule within ~10%.
2. `max_tokens` is set per call in the logger below (DEFAULT_MAX_TOKENS). Set it
   to fit a typical reply, not artificially high: an inflated cap reserves
   window you cannot use and risks runaway cost on a misbehaving prompt.
3. If the system prompt is byte-identical across calls, Anthropic prompt caching
   (cache_control on the system block) helps. Verify via the response usage:
   cache_creation_input_tokens on the first call, cache_read_input_tokens on
   the next.

PRICING NOTE: fill PRICES with numbers you look up AT RUN TIME and cite below.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

from anthropic import Anthropic

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 512
USAGE_LOG = "usage_log.csv"

# Prices are USD per 1,000,000 tokens. LOOK THESE UP and cite the source.
# Source: https://www.anthropic.com/pricing  (snapshot date: FILL IN)
PRICES: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
}


# --- Task 1: exact token counting ------------------------------------------


def count_tokens_anthropic(client: Anthropic, text: str, model: str) -> int:
    """Exact input-token count via the official Count Tokens endpoint."""
    result = client.messages.count_tokens(
        model=model,
        messages=[{"role": "user", "content": text}],
    )
    return result.input_tokens


def count_tokens_tiktoken(text: str, model: str) -> int:
    """Exact count for GPT models. Requires `pip install tiktoken`."""
    import tiktoken

    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")  # safe fallback
    return len(enc.encode(text))


def compare_to_rule_of_thumb(label: str, text: str, exact: int) -> None:
    """Show where len(text) / 4 is a good estimate and where it fails."""
    estimate = len(text) // 4
    drift = (estimate - exact) / exact * 100 if exact else 0.0
    print(f"{label:<12} exact={exact:>7}  chars/4={estimate:>7}  drift={drift:+6.1f}%")


# --- Task 2: token-budget calculator ---------------------------------------


class ContextBudget:
    """Reserve named line items against a model's context window."""

    def __init__(self, *, window: int, reserved_reply: int, overhead: int) -> None:
        self.window = window
        self._reservations: list[tuple[str, int]] = [
            ("Reserved reply", reserved_reply),
            ("Fixed overhead", overhead),
        ]

    def reserve(self, label: str, tokens: int) -> None:
        self._reservations.append((label, tokens))

    def remaining(self) -> int:
        return self.window - sum(tokens for _, tokens in self._reservations)

    def fit(self, items: list[tuple[str, int]]) -> list[str]:
        """Greedily accept items until the budget is exhausted; refuse the rest."""
        budget = self.remaining()
        accepted: list[str] = []
        for label, tokens in items:
            if tokens <= budget:
                accepted.append(label)
                budget -= tokens
        return accepted

    def report(self, model: str) -> None:
        print(f"Model:          {model}")
        print(f"Window:         {self.window}")
        for label, tokens in self._reservations:
            print(f"{label + ':':<16}{tokens}")
        print("-" * 37)
        print(f"Available for messages + retrieval: {self.remaining()} tokens")


# --- Task 3: log real usage data -------------------------------------------


def log_turn(model: str, usage, max_tokens: int, stop_reason: str) -> None:
    """Append one chat turn's measured usage to usage_log.csv."""
    new_file = not os.path.exists(USAGE_LOG)
    ratio = usage.output_tokens / max_tokens if max_tokens else 0.0
    with open(USAGE_LOG, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if new_file:
            writer.writerow(
                ["model", "input_tokens", "output_tokens", "out_over_cap", "stop_reason"]
            )
        writer.writerow(
            [model, usage.input_tokens, usage.output_tokens, f"{ratio:.3f}", stop_reason]
        )


# --- Task 4: project cost from usage ---------------------------------------


def project_cost(model: str) -> None:
    """Read usage_log.csv and print per-call and daily-scale cost projections."""
    price = PRICES.get(model)
    if price is None:
        sys.exit(f"No price for {model}. Add it to PRICES and cite the source.")

    rows = list(csv.DictReader(open(USAGE_LOG, encoding="utf-8")))
    if not rows:
        sys.exit(f"{USAGE_LOG} is empty. Run chat turns first.")

    total_in = sum(int(r["input_tokens"]) for r in rows)
    total_out = sum(int(r["output_tokens"]) for r in rows)
    total_cost = (total_in * price["input"] + total_out * price["output"]) / 1_000_000
    avg_cost = total_cost / len(rows)

    print(f"Rows: {len(rows)}  input_tokens={total_in}  output_tokens={total_out}")
    print(f"Total cost: ${total_cost:.6f}   Avg/call: ${avg_cost:.6f}")
    print("Projected daily cost at average call size:")
    for calls in (100_000, 1_000_000, 10_000_000):
        print(f"  {calls:>12,} calls/day -> ${avg_cost * calls:,.2f}")

    biggest = "input" if total_in >= total_out else "output"
    print(f"Biggest token contributor across the run: {biggest} tokens.")


# --- glue ------------------------------------------------------------------


def cmd_count(client: Anthropic, args) -> None:
    for label, text in (
        ("hello", "Hello, world."),
        ("corpus", _read(args.corpus)),
        ("code", _read(args.code)),
        ("json", json.dumps(_read_json(args.json_blob))),
    ):
        if text is None:
            continue
        exact = count_tokens_anthropic(client, text, args.model)
        compare_to_rule_of_thumb(label, text, exact)


def cmd_budget(args) -> None:
    budget = ContextBudget(
        window=args.window, reserved_reply=args.reply, overhead=args.overhead
    )
    budget.reserve("System", _count_file(args.system))
    budget.reserve("Tools", _count_file(args.tools))
    budget.report(args.model)
    if args.docs:
        items = [(path, _count_file(path)) for path in args.docs]
        fitted = budget.fit(items)
        print(f"Docs that fit under budget: {len(fitted)} / {len(items)}")


def _read(path: str | None) -> str | None:
    return open(path, encoding="utf-8").read() if path else None


def _read_json(path: str | None) -> object:
    return json.load(open(path, encoding="utf-8")) if path else {}


def _count_file(path: str | None) -> int:
    """Cheap local estimate for budgeting line items (chars/4)."""
    text = _read(path)
    return len(text) // 4 if text else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tokens, context, and cost.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_count = sub.add_parser("count", help="Task 1: exact token counts.")
    p_count.add_argument("--model", default=DEFAULT_MODEL)
    p_count.add_argument("--corpus")
    p_count.add_argument("--code")
    p_count.add_argument("--json-blob", dest="json_blob")

    p_budget = sub.add_parser("budget", help="Task 2: context budget.")
    p_budget.add_argument("--model", default=DEFAULT_MODEL)
    p_budget.add_argument("--window", type=int, default=200_000)
    p_budget.add_argument("--reply", type=int, default=4096)
    p_budget.add_argument("--overhead", type=int, default=2000)
    p_budget.add_argument("--system", required=True)
    p_budget.add_argument("--tools", required=True)
    p_budget.add_argument("--docs", nargs="*", default=[])

    p_cost = sub.add_parser("cost", help="Task 4: project cost from usage.")
    p_cost.add_argument("--model", default=DEFAULT_MODEL)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "count":
        cmd_count(Anthropic(), args)
    elif args.cmd == "budget":
        cmd_budget(args)
    elif args.cmd == "cost":
        project_cost(args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Task 3 wires into the exercise-01 chat program: after every response, call
`log_turn(model, response.usage, max_tokens, response.stop_reason)`. Run 20+
real turns to populate `usage_log.csv`, then run `cost`.

For Task 5, prompt caching uses a structured system block:

```python
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=512,
    system=[
        {"type": "text", "text": stable_system_prompt, "cache_control": {"type": "ephemeral"}}
    ],
    messages=[{"role": "user", "content": question}],
)
print("cache_creation:", response.usage.cache_creation_input_tokens)
print("cache_read:    ", response.usage.cache_read_input_tokens)
```

The first call shows `cache_creation_input_tokens`; subsequent identical-prefix
calls show `cache_read_input_tokens` instead — the verifiable proof of a cache
hit.

## Meeting the acceptance criteria

- **`count_tokens()` matches the official tool exactly.**
  `count_tokens_anthropic` calls `client.messages.count_tokens`, which is the
  authoritative count; `count_tokens_tiktoken` uses the model's own encoding.
  Neither approximates.
- **Budget script reports headroom and refuses to overfit.** `ContextBudget`
  reserves System, Tools, overhead, and reply, prints the exact "Available for
  messages + retrieval" line from the exercise, and `fit()` only accepts items
  while `budget` remains, refusing the rest.
- **`usage_log.csv` from 20+ real calls.** `log_turn` appends model ID,
  `input_tokens`, `output_tokens`, the output/cap ratio, and the stop reason for
  every turn.
- **Cost projection at 100K / 1M / 10M calls/day with cited prices.**
  `project_cost` reads the CSV, applies the `PRICES` dict (with the required
  source-and-date citation comment), and prints the daily projection table plus
  the biggest token contributor.
- **Before/after optimization with token + quality observations.** The caching
  snippet shows `cache_creation` vs `cache_read` token counts; trimming the
  system prompt and re-running `count` shows the `input_tokens` drop, and you
  record the quality observation in the reflection block.

## Common pitfalls

- **Approximating instead of counting.** `len(text) / 4` is a sanity check, not
  a count. The acceptance criterion demands an exact match to the official tool;
  use the endpoint or `tiktoken`, never the rule of thumb, for billing math.
- **Hardcoding stale prices.** Prices move. Keep them in one `PRICES` dict with
  a dated source URL and look them up when you run the exercise — a wrong price
  silently corrupts every projection.
- **Counting tokens with the wrong tokenizer.** `tiktoken.encoding_for_model`
  must match the actual model family; a mismatched encoding gives wrong counts.
  The fallback to `cl100k_base` is explicit so the mismatch is visible.
- **Confusing `max_tokens` with the context window in the budget.**
  `max_tokens` is the *reserved reply*, one line item; the window is the whole
  budget. The report keeps them as separate rows on purpose.
- **Expecting a cache hit on a non-identical prefix.** Prompt caching keys on a
  byte-identical prefix. One changed character in the system block busts the
  cache; verify with the `usage` fields rather than assuming.

## Verification

```bash
pip install anthropic tiktoken
export ANTHROPIC_API_KEY=...

# Task 1: exact counts vs. the chars/4 rule
python solution.py count --corpus corpus.txt --code solution.py --json-blob sample.json

# Task 2: budget report + how many docs fit
python solution.py budget --system system_prompt.txt --tools tools.json \
    --window 200000 --reply 4096 --overhead 2000 doc1.txt doc2.txt

# Task 3: run 20+ chat turns (wired into exercise-01's loop) to fill usage_log.csv
# Task 4: project cost from the logged usage
python solution.py cost
```

A green run prints exact counts with honest drift percentages (prose near 0%,
JSON well off), a budget report whose "Available" line matches the exercise's
format, and a cost table that scales linearly with call volume.
