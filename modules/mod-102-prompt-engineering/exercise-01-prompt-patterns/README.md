# mod-102-prompt-engineering/exercise-01 — Solution

## Approach

The exercise asks for the *same* triage task expressed four ways — bare
instruction (v0), role + format (v1), few-shot (v2), and decomposition
(v3) — run over one frozen input set so that every output difference is
attributable to a prompt change rather than to sampling noise.

The reference solution treats prompts as data: each version's text lives
in a `prompts/` file (or, for v3, two files), and a single `run.py`
loads the relevant prompt(s), iterates over `inputs.txt`, calls the model
at `temperature=0`, parses defensively, and appends one row per
`(input, version)` to `results.csv`. The same model ID and temperature
are used everywhere and recorded in `NOTES.md`. No validation library,
no vendor JSON mode, no retries — those belong to exercises 02 and 03.

Design decisions worth calling out:

- **One provider abstraction, two prompts in, one record out.** The
  model call is wrapped so that v0–v2 share a single-call path and v3
  reuses it twice (classify, then summarize). That keeps the diff
  between versions in the *prompt*, not the *plumbing*.
- **Parse defensively, do not validate.** We strip code fences and pull
  the outer-most JSON object (`extract_json_object`, from chapter 6) but
  deliberately do **not** enforce the enum or summary length yet — the
  point of v0 is to *watch* it fail.
- **Everything that affects output is logged.** Latency and
  `stop_reason` go into every row so the `NOTES.md` recommendation can
  cite numbers.

Model IDs follow the curriculum's pinned references
(`claude-sonnet-4-6`), overridable via the `MODEL` environment variable
so the same script runs against a cheaper tier without edits.

## Reference implementation

The full layout matches the exercise's suggested structure. Below are
the load-bearing files: the five prompt texts, the runner, and the
analysis stub. Everything is runnable as-is given
`pip install anthropic` and `ANTHROPIC_API_KEY`.

### `prompts/v0_bare.txt`

```text
Classify the following message and produce category, urgency, summary:

{message}
```

### `prompts/v1_role_format.txt`

```text
You are a support-inbox triage assistant for an e-commerce store.

For each customer message, output a SINGLE JSON object and nothing else
— no prose, no code fences, no commentary.

Schema:
{
  "category": one of ["billing", "shipping", "returns", "product", "other", "unknown"],
  "urgency":  one of ["low", "normal", "high"],
  "summary":  a string of at most 140 characters, suitable as a ticket title
}

Rules:
- Use ONLY the labels listed above. Never invent a label such as "critical".
- If the message is empty, unintelligible, or you cannot tell what it is
  about, return "category": "unknown" and "urgency": "low".
- The summary must be one sentence and must not exceed 140 characters.

Output the JSON object only.
```

### `prompts/v2_few_shot.txt`

```text
You are a support-inbox triage assistant for an e-commerce store.

For each customer message, output a SINGLE JSON object and nothing else
— no prose, no code fences, no commentary.

Schema:
{
  "category": one of ["billing", "shipping", "returns", "product", "other", "unknown"],
  "urgency":  one of ["low", "normal", "high"],
  "summary":  a string of at most 140 characters, suitable as a ticket title
}

Rules:
- Use ONLY the labels listed above. Never invent a label such as "critical".
- If the message is empty, unintelligible, or you cannot tell what it is
  about, return "category": "unknown" and "urgency": "low".
- The summary must be one sentence and must not exceed 140 characters.

Examples:

Message: I was charged twice for the same purchase yesterday — please fix this asap.
{"category": "billing", "urgency": "high", "summary": "Customer double-charged for one purchase, requests urgent fix."}

Message: Do you carry the blue color in size medium?
{"category": "product", "urgency": "low", "summary": "Asks whether blue is available in size medium."}

Message: hola, mi pedido nunca llegó y necesito el reembolso ya
{"category": "shipping", "urgency": "high", "summary": "Order never arrived; customer demands an immediate refund."}

Message:
{"category": "unknown", "urgency": "low", "summary": "Empty message; no actionable content."}

Output the JSON object only.
```

### `prompts/v3_classify.txt`

```text
You are a support-inbox triage classifier for an e-commerce store.

Output a SINGLE JSON object and nothing else:
{
  "category": one of ["billing", "shipping", "returns", "product", "other", "unknown"],
  "urgency":  one of ["low", "normal", "high"]
}

Use ONLY the labels listed. If the message is empty or unintelligible,
return "category": "unknown" and "urgency": "low". Output JSON only.
```

### `prompts/v3_summarize.txt`

```text
You summarize a single e-commerce support message into one short ticket
title. Output ONLY the title as plain text — no quotes, no JSON, no
prefix. Maximum 140 characters. If the message is empty, output:
(no content)
```

### `run.py`

```python
"""Run one prompt version of the triage task over inputs.txt.

Usage:
    python run.py --version v0          # or v1, v2, v3
    MODEL=claude-haiku-4-5 python run.py --version v2

Writes/extends results.csv with one row per (input, version) pair.
Requires: pip install anthropic ; export ANTHROPIC_API_KEY=...
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

from anthropic import Anthropic

# Pinned per the curriculum; override with MODEL=... for a cheaper tier.
MODEL = os.environ.get("MODEL", "claude-sonnet-4-6")
MAX_TOKENS = 512
HERE = Path(__file__).resolve().parent
PROMPTS = HERE / "prompts"

client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_object(s: str) -> dict:
    """Strip a single code fence (if any) and parse the outermost JSON object.

    This is the chapter-6 defensive parser. In this exercise we only parse;
    we do NOT validate the enum or summary length — that is the point of v0.
    """
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s).strip()
    match = _JSON_OBJECT.search(s)
    if not match:
        raise ValueError("no JSON object found in model output")
    return json.loads(match.group(0))


def load_prompt(name: str) -> str:
    return (PROMPTS / name).read_text(encoding="utf-8")


def call_model(system: str, user: str, *, temperature: float) -> tuple[str, str, int]:
    """Return (text, stop_reason, latency_ms) for one single-turn call."""
    start = time.perf_counter()
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=temperature,
        system=system or "You are a helpful assistant.",
        messages=[{"role": "user", "content": user}],
    )
    latency_ms = int((time.perf_counter() - start) * 1000)
    text = "".join(b.text for b in response.content if b.type == "text")
    return text, response.stop_reason, latency_ms


def run_single_call(version: str, message: str) -> dict:
    """v0, v1, v2 — one call that should yield a JSON object."""
    if version == "v0":
        system = ""
        user = load_prompt("v0_bare.txt").format(message=message)
    else:
        template = {"v1": "v1_role_format.txt", "v2": "v2_few_shot.txt"}[version]
        system = load_prompt(template)
        user = message

    text, stop_reason, latency_ms = call_model(system, user, temperature=0)
    record = {"category": "", "urgency": "", "summary": "", "parse_error": ""}
    try:
        parsed = extract_json_object(text)
        record["category"] = parsed.get("category", "")
        record["urgency"] = parsed.get("urgency", "")
        record["summary"] = parsed.get("summary", "")
    except (ValueError, json.JSONDecodeError) as err:
        record["parse_error"] = str(err)
    record["latency_ms"] = latency_ms
    record["stop_reason"] = stop_reason
    return record


def run_decomposed(message: str) -> dict:
    """v3 — two calls (classify strict, summarize loose), assembled in Python."""
    cls_text, cls_stop, cls_ms = call_model(
        load_prompt("v3_classify.txt"), message, temperature=0
    )
    sum_text, sum_stop, sum_ms = call_model(
        load_prompt("v3_summarize.txt"), message, temperature=0.2
    )
    record = {"category": "", "urgency": "", "summary": "", "parse_error": ""}
    try:
        parsed = extract_json_object(cls_text)
        record["category"] = parsed.get("category", "")
        record["urgency"] = parsed.get("urgency", "")
    except (ValueError, json.JSONDecodeError) as err:
        record["parse_error"] = f"classify: {err}"
    record["summary"] = sum_text.strip()[:140]
    record["latency_ms"] = cls_ms + sum_ms
    # Surface the worst stop reason of the two stages.
    record["stop_reason"] = cls_stop if cls_stop != "end_turn" else sum_stop
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, choices=["v0", "v1", "v2", "v3"])
    args = parser.parse_args()

    inputs = (HERE / "inputs.txt").read_text(encoding="utf-8").splitlines()
    results_path = HERE / "results.csv"
    fieldnames = [
        "input", "version", "category", "urgency", "summary",
        "latency_ms", "stop_reason", "parse_error",
    ]
    new_file = not results_path.exists()

    with results_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        for line in inputs:
            message = line  # keep blank lines: they exercise the empty case
            if args.version == "v3":
                record = run_decomposed(message)
            else:
                record = run_single_call(args.version, message)
            record["input"] = message
            record["version"] = args.version
            writer.writerow(record)
            tag = record["parse_error"] or record["category"]
            print(f"[{args.version}] {message[:40]!r:44} -> {tag}")


if __name__ == "__main__":
    main()
```

### `analyze.py` (helper used while writing `NOTES.md`)

```python
"""Summarize results.csv: parse-failure counts and enum violations per version.

This is the evidence-gathering step the NOTES.md recommendation must cite.
Run after collecting all four versions:  python analyze.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

ALLOWED_CATEGORY = {"billing", "shipping", "returns", "product", "other", "unknown"}
ALLOWED_URGENCY = {"low", "normal", "high"}


def main() -> None:
    rows = list(csv.DictReader((Path(__file__).parent / "results.csv").open()))
    stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for row in rows:
        version = row["version"]
        stats[version]["n"] += 1
        if row["parse_error"]:
            stats[version]["parse_fail"] += 1
        if row["category"] and row["category"] not in ALLOWED_CATEGORY:
            stats[version]["bad_category"] += 1
        if row["urgency"] and row["urgency"] not in ALLOWED_URGENCY:
            stats[version]["bad_urgency"] += 1
        if len(row["summary"]) > 140:
            stats[version]["long_summary"] += 1
        stats[version]["latency_sum"] += int(row["latency_ms"])

    header = f"{'ver':4} {'n':>3} {'parse_fail':>10} {'bad_cat':>8} {'bad_urg':>8} {'long_sum':>9} {'mean_ms':>8}"
    print(header)
    for version in sorted(stats):
        s = stats[version]
        mean_ms = s["latency_sum"] // max(s["n"], 1)
        print(
            f"{version:4} {s['n']:>3} {s['parse_fail']:>10} "
            f"{s['bad_category']:>8} {s['bad_urgency']:>8} "
            f"{s['long_summary']:>9} {mean_ms:>8}"
        )


if __name__ == "__main__":
    main()
```

### `NOTES.md` (the deliverable that interprets the runs)

The grader's `NOTES.md` is prose; the structure below is what a strong
submission contains. The point is that every claim is backed by a count
or a concrete input, never by preference.

```markdown
# Triage prompt patterns — notes

- Model: claude-sonnet-4-6 (pinned). Temperature: 0 for all classifying
  calls; 0.2 only for the v3 summarize stage. Recorded here per the
  acceptance criteria.

## What each version changed (with evidence)

- v0 -> v1: `"Where is my order!!!!!!"` returned `urgency: "critical"`
  under v0 (an invented label) and `urgency: "high"` under v1. The
  named, closed label set in the system prompt is what fixed it. The
  empty line went from a parse failure (prose apology) to a clean
  `unknown/low` once the rule "if empty, return unknown" was explicit.
- v1 -> v2: the Spanish line stabilized to `shipping/high` because the
  multilingual few-shot example anchored it; under v1 it oscillated
  between `shipping` and `returns`. REGRESSION: the polite
  "love the new packaging" line, which v1 read as `other/low`, was
  pulled toward `product/low` in v2 because every worked example had a
  concrete product/issue — over-fitting to the example distribution.
- v2 -> v3: decomposition did not change any *category*. It made the
  summary stage independently tunable (temperature 0.2) without
  destabilizing classification, and it roughly doubled latency
  (two calls). It also let a wrong classification and a fine summary
  coexist on the same record, which a human reviewer can now spot.

## Recommendation

Ship v2 for production. It has zero parse failures and zero enum
violations across the input set (see analyze.py output), matching v1 on
robustness while improving the two hardest inputs (multilingual, empty).
v3's only win is maintainability, at 2x latency and 2x cost — not worth
it until the summary needs independent tuning.

## Reflection

1. Few-shot hurt the polite-no-issue case (pushed `other` -> `product`).
   Adding an `other/low` example with no product mention would fix it.
2. v3 improved maintainability, not correctness — same categories as v2.
3. The few-shot block earned the most per token: four short examples
   removed the remaining enum drift that paragraphs of rules did not.
```

## Meeting the acceptance criteria

| Criterion | Where it is satisfied |
| --- | --- |
| Four prompt versions (v0–v3) as separate files | `prompts/v0_bare.txt`, `v1_role_format.txt`, `v2_few_shot.txt`, plus `v3_classify.txt` + `v3_summarize.txt` |
| `results.csv`, one row per `(input, version)` with parsed fields, latency, stop reason | `run.py` appends exactly that via `csv.DictWriter` |
| `NOTES.md` identifies, per step, one input whose output changed and why | "What each version changed" section, with named inputs |
| Reasoned ship recommendation grounded in data | "Recommendation" section cites `analyze.py` counts |
| Same model ID and `temperature=0`, recorded in `NOTES.md` | `MODEL` constant, temperature pinned in `call_model`, both noted |

## Common pitfalls

- **Letting plumbing differ between versions.** If v2 uses a different
  `max_tokens` or model than v1, you cannot attribute the change to the
  prompt. Keep `call_model` identical and vary only the prompt text.
- **Stripping blank lines from `inputs.txt`.** The empty-message edge
  case lives in those blank lines; `splitlines()` preserves them and the
  runner must not filter them out.
- **Validating in v0.** The instinct to enforce the enum immediately
  hides the very failure v0 is meant to expose. Parse only; record the
  raw category, even when it is `critical`.
- **Editing the prompt to chase a single input.** Few-shot over-fitting
  (the polite-packaging regression) is a finding, not a bug to patch
  away mid-run. Record it.
- **Calling the v3 summarize stage at `temperature=0`.** The exercise
  asks for `0.2` there specifically so you can observe that the loose
  stage does not destabilize the strict classify stage.

## Verification

```bash
cd exercise-01-prompt-patterns
python -m pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...      # your key

# Generate the four versions (each appends to results.csv).
for v in v0 v1 v2 v3; do python run.py --version "$v"; done

# Gather the evidence the NOTES.md recommendation cites.
python analyze.py

# Sanity checks on the artifacts.
test -f results.csv && echo "results.csv present"
wc -l results.csv                         # 1 header + (inputs * 4) rows
ls prompts/                               # five prompt files
```

Expected: `analyze.py` shows v1 and v2 with `parse_fail=0` and
`bad_cat=0`, v0 with at least one parse failure and/or an invented label,
and v3's `mean_ms` roughly double v2's.
</content>
</invoke>
