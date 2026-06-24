# mod-103-tool-and-function-calling — Solutions

Reference solutions for the Tool & Function Calling module. Each exercise builds
on the previous one: a single-tool loop, then a multi-tool registry, then the
error-handling layer that makes the surface shippable.

## Exercises

- [exercise-01 — First Function Calling](exercise-01-first-function-calling/README.md):
  one Python function, one tool definition, and the full `tool_use` →
  `tool_result` → continuation loop with explicit `stop_reason` handling and a
  `max_steps` guard.
- [exercise-02 — Multi-Tool App](exercise-02-multi-tool-app/README.md): four
  tools behind a `ToolRegistry` and dispatcher, `tool_choice` (auto / forced /
  forbidden), parallel tool use, and one-purpose-per-tool descriptions.
- [exercise-03 — Tool Error Handling](exercise-03-tool-error-handling/README.md):
  Pydantic argument validation at the boundary, `safe_dispatch` converting the
  four failure modes into structured feedback, and a `ToolBudget` that fails
  loudly on exhaustion.

## Conventions

- Code targets Anthropic (the lessons' default), but the loop is
  provider-shaped — porting to OpenAI changes only field names (`tool_use` →
  `tool_calls`, parsed `input` → JSON-string `arguments`, `tool_result` in a
  user turn → a `role: "tool"` message).
- Every reference implementation pins `MODEL`, `temperature=0`, and `max_tokens`
  at module top for reproducibility, and ships a `--fake` offline mode so the
  loop logic is verifiable without an API key or spend.
