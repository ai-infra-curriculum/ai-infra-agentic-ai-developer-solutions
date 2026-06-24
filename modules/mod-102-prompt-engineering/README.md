# mod-102-prompt-engineering: Prompt Engineering & Structured Output — Solutions

Reference solutions for the prompt-engineering module. Each exercise
folder holds an annotated, runnable walkthrough that maps directly to the
learning module's tasks and acceptance criteria.

## Exercise solutions

- [exercise-01 — Prompt patterns](exercise-01-prompt-patterns/README.md):
  the same triage task as bare instruction, role + format, few-shot, and
  decomposition, run over a frozen input set with per-version evidence
  and a data-grounded ship recommendation.
- [exercise-02 — Structured JSON output](exercise-02-structured-json-output/README.md):
  prompt-only (with Anthropic prefill), vendor JSON mode,
  schema-constrained generation, and forced tool calling, compared on
  shape-correctness, latency, and adversarial inputs from one Pydantic
  source of truth.
- [exercise-03 — Output validation](exercise-03-output-validation/README.md):
  a typed `TriageResult` boundary with a business-rule validator, a
  capped repair loop, structured logging, and a frozen 20-case golden set
  with a CI-gating eval runner.

## Conventions

- Code targets Python 3.10+ and the official `anthropic` (and, where the
  exercise calls for it, `openai`) SDKs.
- Model IDs follow the curriculum's pinned references and are overridable
  via the `MODEL` (or `ANTHROPIC_MODEL` / `OPENAI_MODEL`) environment
  variable so the same scripts run against a cheaper tier in CI.
- API keys come from the environment (`ANTHROPIC_API_KEY`,
  `OPENAI_API_KEY`); nothing is hardcoded.
- Each solution's `## Verification` section lists the exact commands to
  reproduce the results, including offline checks that need no API key.
</content>
