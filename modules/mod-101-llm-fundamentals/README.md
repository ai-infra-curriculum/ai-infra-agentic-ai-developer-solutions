# mod-101-llm-fundamentals — Solutions

Reference solutions for the LLM Fundamentals module. Each exercise has a full
walkthrough: approach, an annotated runnable implementation, a mapping to the
acceptance criteria, common pitfalls, and verification steps.

The implementations target Anthropic's Messages API (matching the exercise
skeletons) and port directly to OpenAI by swapping the client, the response
field names, and the message shape.

## Exercise solutions

- [exercise-01-first-llm-api-calls](exercise-01-first-llm-api-calls/README.md)
  — single-turn calls, a behavior-shaping system prompt, a stateful multi-turn
  REPL, and the temperature / max-tokens / model knobs with stop reasons.
- [exercise-02-tokens-context-cost](exercise-02-tokens-context-cost/README.md)
  — exact token counting, a `ContextBudget` calculator, real `usage` logging to
  CSV, and cost projection at 100K / 1M / 10M calls/day.
- [exercise-03-streaming-and-errors](exercise-03-streaming-and-errors/README.md)
  — streaming with TTFT and tokens/sec, a typed error taxonomy, full-jitter
  retry with `Retry-After`, idempotency keys, mid-stream error handling, and a
  `--metrics` dashboard.

## How to use these

Attempt each exercise first, then read the matching solution. The value is in
comparing your structure to the reference: where the retry policy lives, where
the chat history resets, and how prices and token counts are measured rather
than guessed. Bring a funded API key with a small spend cap to run the
verification steps.
