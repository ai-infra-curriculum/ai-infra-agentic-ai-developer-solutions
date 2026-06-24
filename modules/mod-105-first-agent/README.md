# mod-105-first-agent: Your First Agent: The Reason-Act Loop — Solutions

Reference implementations and per-exercise walkthroughs for the
[mod-105 learning module](../../../ai-infra-agentic-ai-developer-learning/lessons/mod-105-first-agent/README.md).
Each solution is a single runnable Python script plus an annotated README
covering the approach, how it meets the acceptance criteria, common pitfalls,
and verification steps.

## Exercises

- [exercise-01: ReAct loop intro](exercise-01-react-loop-intro/README.md) — the
  minimal `run_agent` loop over two toy tools (`get_weather`,
  `set_thermostat`), with a step cap, errors-as-observations, a printed
  Thought/Action trace, and the step-cap and unknown-tool adversarial probes.
- [exercise-02: Tool-using agent](exercise-02-tool-using-agent/README.md) — the
  `mod-104` retriever wired in as `search_kb` alongside `get_order_status` and
  `create_refund` through one dispatcher, with a routing suite and after-the-fact
  citation validation (`fabricated` is the bar).
- [exercise-03: Basic conversation memory](exercise-03-basic-conversation-memory/README.md)
  — exercise-02's agent turned into a stateful `AgentSession` with a running
  transcript, a token-budget cascade (drop stale tool payloads, then summarize),
  and a `remember` / `recall` scratchpad that survives transcript pressure.

## Running the solutions

Each exercise's code targets Python 3.10+, the Anthropic SDK, and an exported
`ANTHROPIC_API_KEY`; total spend across all three stays well under a dollar.
Install per-exercise deps (`anthropic`, and `numpy` for exercises 02–03), then
run the script named in that exercise's README. Every README ends with a
no-API-key syntax check so you can validate the code offline first.
