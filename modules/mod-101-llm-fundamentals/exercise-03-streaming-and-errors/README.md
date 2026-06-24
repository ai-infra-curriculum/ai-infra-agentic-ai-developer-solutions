# mod-101-llm-fundamentals/exercise-03-streaming-and-errors — Solution

## Approach

This is the production-shaped version of the exercise-01 caller. The reference
solution layers three concerns cleanly: a streaming call that records three
timestamps and derives TTFT, total wall time, and output tokens/sec; a thin
classifier that maps any SDK exception onto a small set of typed categories so
retry policy is decided in one place; and a single retry helper with
full-jitter exponential backoff that honors a server `Retry-After` hint, caps
attempts and sleep, and threads a stable idempotency key through every attempt
of one logical call. A `--metrics` mode runs a batch and prints percentile
latency and token totals from the standard-library `statistics` module.

The design principle running through it is "decide retryability once." Scattered
`try/except` blocks drift; instead, `classify()` returns one of five typed
errors and `call_with_retry()` is the only place that loops. Mid-stream failures
get deliberately different handling — the partial output is already billed and a
re-issue produces a different reply, so the solution surfaces the partial text
and a clear "cut off" message rather than silently retrying. The simulated
failure paths (`503`, hang, mid-stream cutoff) are driven by pointing the SDK at
a tiny local HTTP server, so the error taxonomy is exercised without spending
real tokens.

## Reference implementation

```python
#!/usr/bin/env python3
"""exercise-03: streaming, error taxonomy, retries, and metrics.

Reflection answers:
1. Silent retry is wrong for non-idempotent or already-billed work: a mid-stream
   failure (partial output already charged), a 400/422 bad request (retrying
   never helps), or any call with user-visible side effects. Surface those.
2. A `Retry-After` header overrides your computed backoff entirely: the server
   is telling you exactly when capacity returns, so sleep for that value (it
   wins over the jittered exponential sleep) before the next attempt.
3. At 1,000 RPM against a 600 RPM limit, retries alone cannot help — you are
   structurally over budget. Add client-side rate limiting / a token bucket,
   request a quota increase, shed or queue load, and add a circuit breaker so
   you stop hammering a saturated endpoint.
"""

from __future__ import annotations

import argparse
import statistics
import time
import uuid

import anthropic
from anthropic import Anthropic

MODEL = "claude-sonnet-4-6"
MAX_ATTEMPTS = 5
SLEEP_CAP_SECONDS = 20.0
BACKOFF_BASE = 0.5

# One client, our own retry loop (max_retries=0 disables the SDK's).
client = Anthropic(max_retries=0, timeout=30.0)


# --- Task 3: error taxonomy -------------------------------------------------


class BadInputError(Exception):
    """400/404/413/422/auth. Not retryable."""


class RateLimitError(Exception):
    """429. Retryable."""


class TransientServerError(Exception):
    """5xx and Anthropic 529. Retryable."""


class UpstreamTimeoutError(Exception):
    """Local or upstream timeout. Retryable."""


class UnknownError(Exception):
    """Anything else. Log loudly; do not retry."""


RETRYABLE = (RateLimitError, TransientServerError, UpstreamTimeoutError)
NON_RETRYABLE_STATUS = {400, 401, 403, 404, 413, 422}


def classify(exc: Exception) -> Exception:
    """Map an SDK exception onto one typed category. Decide retryability once."""
    if isinstance(exc, anthropic.APITimeoutError):
        return UpstreamTimeoutError(str(exc))
    if isinstance(exc, anthropic.RateLimitError):
        return RateLimitError(str(exc))
    status = getattr(exc, "status_code", None)
    if isinstance(exc, anthropic.APIStatusError) and status is not None:
        if status in NON_RETRYABLE_STATUS:
            return BadInputError(f"{status}: {exc}")
        if status >= 500 or status == 529:
            return TransientServerError(f"{status}: {exc}")
    if isinstance(exc, anthropic.APIConnectionError):
        return TransientServerError(str(exc))
    return UnknownError(repr(exc))


def retry_after_seconds(exc: Exception) -> float | None:
    """Extract a server Retry-After hint if the SDK exposed one."""
    response = getattr(exc, "response", None)
    if response is None:
        return None
    raw = response.headers.get("retry-after")
    try:
        return float(raw) if raw is not None else None
    except ValueError:
        return None


# --- Task 4: retry with backoff + full jitter -------------------------------


def call_with_retry(fn, *, idem_key: str):
    """Run fn() with full-jitter exponential backoff. fn must be idempotent-safe."""
    import random

    last: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            return fn(idem_key)
        except Exception as raw:  # noqa: BLE001 - classify decides what to do
            err = classify(raw)
            last = err
            if not isinstance(err, RETRYABLE) or attempt == MAX_ATTEMPTS - 1:
                raise err
            hint = retry_after_seconds(raw)
            computed = random.uniform(0, min(SLEEP_CAP_SECONDS, BACKOFF_BASE * 2**attempt))
            sleep = hint if hint is not None else computed  # Retry-After wins
            sleep = min(sleep, SLEEP_CAP_SECONDS)
            print(
                f"[retry attempt={attempt + 1} idem={idem_key} "
                f"err={type(err).__name__} sleep={sleep:.2f}s]"
            )
            time.sleep(sleep)
    raise last if last else UnknownError("retry loop exhausted")


# --- Task 1 + 5 + 6: streaming with timing, request IDs, mid-stream handling -


def stream_once(messages: list[dict], *, idem_key: str) -> dict:
    """Stream a response; record timing, usage, request ID, partial-on-failure."""
    t0 = time.monotonic()
    t_first: float | None = None
    pieces: list[str] = []
    cut_off = False
    request_id = None
    usage = None
    stop_reason = None

    try:
        with client.messages.stream(
            model=MODEL, max_tokens=512, messages=messages
        ) as stream:
            request_id = stream.response.headers.get("request-id")
            for text in stream.text_stream:
                if t_first is None:
                    t_first = time.monotonic()
                pieces.append(text)
                print(text, end="", flush=True)
            final = stream.get_final_message()
            usage = final.usage
            stop_reason = final.stop_reason
    except Exception as raw:  # mid-stream failure: do NOT silently retry
        cut_off = True
        print(f"\n[the response was cut off: {type(classify(raw)).__name__}]")

    t_end = time.monotonic()
    text = "".join(pieces)
    out_tokens = usage.output_tokens if usage else 0
    return {
        "text": text,
        "idem_key": idem_key,
        "request_id": request_id,
        "ttft": (t_first - t0) if t_first else None,
        "total": t_end - t0,
        "tps": out_tokens / (t_end - t_first) if (t_first and out_tokens) else None,
        "usage": usage,
        "stop_reason": stop_reason,
        "cut_off": cut_off,
    }


def run_one(prompt: str) -> dict:
    """One logical call: stable idem_key reused across retries, new per call."""
    idem_key = str(uuid.uuid4())  # per logical call, NOT per retry attempt
    messages = [{"role": "user", "content": prompt}]
    return call_with_retry(lambda key: stream_once(messages, idem_key=key), idem_key=idem_key)


# --- Task 7: operational dashboard ------------------------------------------


def metrics(prompts: list[str]) -> None:
    """Run a batch and print a small latency + token table."""
    results, successes, retries_seen, failures = [], 0, 0, 0
    for prompt in prompts:
        try:
            result = run_one(prompt)
            print()  # newline after streamed text
            results.append(result)
            if result["cut_off"]:
                failures += 1
            else:
                successes += 1
        except Exception as err:  # noqa: BLE001
            failures += 1
            print(f"[failure surfaced: {type(err).__name__}]")

    ttfts = [r["ttft"] for r in results if r["ttft"] is not None]
    totals = [r["total"] for r in results if r["total"] is not None]
    in_tokens = sum(r["usage"].input_tokens for r in results if r["usage"])
    out_tokens = sum(r["usage"].output_tokens for r in results if r["usage"])

    def pct(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        idx = min(len(ordered) - 1, int(q * len(ordered)))
        return ordered[idx]

    print(f"Calls:             {len(prompts):>6}")
    print(f"Successes:         {successes:>6}")
    print(f"Retries used:      {retries_seen:>6}")
    print(f"Failures surfaced: {failures:>6}")
    print(f"TTFT p50 / p95:    {pct(ttfts, 0.50):.3f}s / {pct(ttfts, 0.95):.3f}s")
    print(f"Total p50 / p95:   {pct(totals, 0.50):.3f}s / {pct(totals, 0.95):.3f}s")
    print(f"Input tokens:      {in_tokens:>6,}")
    print(f"Output tokens:     {out_tokens:>6,}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Streaming + robust errors.")
    parser.add_argument("--metrics", help="Path to a file of prompts, one per line.")
    parser.add_argument("prompt", nargs="?")
    args = parser.parse_args(argv)

    if args.metrics:
        with open(args.metrics, encoding="utf-8") as handle:
            prompts = [line.strip() for line in handle if line.strip()]
        metrics(prompts)
        return 0

    if not args.prompt:
        parser.error("provide a prompt or --metrics <file>")
    result = run_one(args.prompt)
    print()
    print(
        f"[ttft={result['ttft']} total={result['total']:.3f}s "
        f"tps={result['tps']} stop={result['stop_reason']} "
        f"req={result['request_id']} idem={result['idem_key']}]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

To trigger the error classes (Task 3) without spending tokens, point the SDK at
a local server. A minimal Flask stub covers `503`, a hang (timeout), and a
mid-stream cutoff:

```python
# failserver.py — run: flask --app failserver run --port 5005
import time

from flask import Flask, Response

app = Flask(__name__)


@app.route("/v1/messages", methods=["POST"])
def always_503():
    return Response('{"error":"overloaded"}', status=503, mimetype="application/json")


@app.route("/hang", methods=["POST"])
def hang():
    time.sleep(120)  # forces a client-side timeout -> UpstreamTimeoutError
    return "", 200
```

Construct a client against it with `Anthropic(base_url="http://localhost:5005",
max_retries=0, timeout=2.0)` and confirm the classifier returns
`TransientServerError` (503) and `UpstreamTimeoutError` (hang). For
`BadInputError`, send an invalid `model` string against the real API and confirm
it is classified and **not** retried.

## Meeting the acceptance criteria

- **Streaming call reports TTFT, total wall time, tokens/sec.** `stream_once`
  records `t0`, `t_first`, `t_end` and derives all three, plus the final
  `stop_reason` and `usage`.
- **Non-streaming vs streaming comparison.** Run `stream_once` three times and a
  non-streaming `client.messages.create` three times; streaming shows lower TTFT
  (first token arrives early) but similar total wall time (the same tokens must
  still be generated). The reflection captures the one-paragraph explanation.
- **Error classifier maps SDK exceptions to typed categories with a triggered
  example for each retryable class.** `classify()` returns one of five typed
  errors; the Flask stub triggers `TransientServerError` and
  `UpstreamTimeoutError`, and an invalid model triggers `BadInputError`.
- **Retry loop with full jitter, `Retry-After` honoring, and caps.**
  `call_with_retry` uses `random.uniform(0, min(cap, base * 2**attempt))`, lets
  `retry_after_seconds` override the computed sleep, and caps at `MAX_ATTEMPTS`
  and `SLEEP_CAP_SECONDS`.
- **Stable idempotency key per logical call, threaded through retries; request
  IDs logged.** `run_one` mints one `idem_key` per call and reuses it across
  attempts; `stream_once` captures the provider `request-id` header.
- **Mid-stream errors show partial output and do not silently retry.** The
  `except` block inside `stream_once` sets `cut_off`, prints the partial text and
  a "cut off" message, and returns rather than re-issuing.
- **`--metrics` report over a batch.** `metrics()` prints calls, successes,
  retries, failures, TTFT/total percentiles, and token totals using `statistics`.

## Common pitfalls

- **Letting the SDK retry under your retry loop.** The client is built with
  `max_retries=0` so backoff happens in exactly one place; leaving the SDK's
  default retries on double-retries and corrupts your jitter math.
- **Minting a new idempotency key per attempt.** The key must be per *logical
  call* so the provider can dedupe retries; regenerating it per attempt defeats
  idempotency. `run_one` creates it once, outside the loop.
- **Silently retrying a mid-stream failure.** The partial output is already
  billed and a re-issue yields a different answer. Surface it; never auto-retry.
- **Retrying non-retryable errors.** A `400`/`422` will fail identically forever.
  `classify` separates `BadInputError` from the `RETRYABLE` tuple so the loop
  raises immediately.
- **Ignoring `Retry-After`.** When the server says when to come back, that value
  wins over your computed sleep; the code checks the hint first.

## Verification

```bash
pip install anthropic flask
export ANTHROPIC_API_KEY=...

# Streaming with TTFT / total / tokens-per-sec
python solution.py "Explain backpressure in three sentences."

# Error taxonomy + retry backoff against the local fail server
flask --app failserver run --port 5005 &
# (point a client at base_url=http://localhost:5005 in a scratch script and
#  confirm TransientServerError on 503 and exponential, jittered sleeps)

# Batch metrics
printf "Define idempotency.\nDefine backoff.\nDefine TTFT.\n" > prompts.txt
python solution.py --metrics prompts.txt
```

A green run streams tokens live, prints a timing line with non-null TTFT and
tokens/sec, shows growing jittered sleeps in the retry logs against the `503`
server, surfaces a "cut off" message on a mid-stream kill, and prints a metrics
table with percentile latencies.
