# mod-106-llm-app-deployment — Solutions

Reference solutions for the **Shipping an LLM Application** module. Each exercise
turns a laptop LLM script into one progressively more production-shaped FastAPI
service: the request boundary, the secret boundary, the observability boundary, and
the run boundary.

The three exercises build the same `support-agent/` repo cumulatively — exercise-02
extends exercise-01, exercise-03 extends exercise-02.

## Index

| Exercise | Solution | Focus |
|----------|----------|-------|
| exercise-01 | [exercise-01-fastapi-llm-endpoint](exercise-01-fastapi-llm-endpoint/README.md) | FastAPI `POST /chat` + `GET /healthz`, Pydantic request/response models, centralized exception handlers |
| exercise-02 | [exercise-02-secrets-and-config](exercise-02-secrets-and-config/README.md) | `pydantic-settings` `Settings`, `SecretStr`, `.env`/`.env.example`, fail-loud-at-startup |
| exercise-03 | [exercise-03-logging-and-usage-guardrail](exercise-03-logging-and-usage-guardrail/README.md) | JSON logging middleware, `DailySpendGuardrail` (`429`), multi-stage Dockerfile |

## Conventions

- Code blocks are runnable. The agent ships an `AGENT_FAKE=1` offline path so the
  test suites and validation smoke tests run with no API key, no network, and no
  spend.
- Provider rates in `prices.py` are pinned from the live Anthropic pricing page with
  a source URL and a check date in the code comment; re-confirm on every release.
- No secret is ever committed, logged, or baked into the image — `.gitignore`
  blocks `.env`, `SecretStr` redacts the key, and `.dockerignore` keeps `.env` out
  of the build context.
