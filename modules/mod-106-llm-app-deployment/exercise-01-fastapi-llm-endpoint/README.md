# mod-106-llm-app-deployment/exercise-01-fastapi-llm-endpoint — Solution

## Approach

The exercise wraps an existing `run_agent(question, ...) -> dict` (ported from
`mod-105`) behind a two-route FastAPI service. The whole lesson is the *boundary*,
so the design keeps three responsibilities cleanly separated:

- **`schemas.py`** owns the wire contract. `ChatRequest` enforces field
  constraints (`min_length`, `max_length`, `ge`, `le`) so malformed bodies are
  rejected with `422` *before* the handler — and therefore before the model — runs.
  `ChatResponse`/`Usage` are the response contract, and `response_model=ChatResponse`
  guarantees nothing the agent returns internally (`trace`, `steps` debug, raw
  provider objects) leaks onto the wire.
- **`agent.py`** owns the model call and is the only module that imports the
  provider SDK. It raises the module's named errors (`MaxStepsExceeded`,
  `ProviderTimeout`, `ProviderBadRequest`) instead of letting raw SDK exceptions
  escape. This is what lets the HTTP layer stay provider-agnostic.
- **`main.py`** is thin: a `request_id`, one call into `run_agent`, one
  `ChatResponse`, and four exception handlers that map known failures to honest
  status codes. No `try/except` in the route body — raising lives in `agent.py`,
  mapping lives in the handlers.

Secrets are read once from `os.environ` at startup (exercise-02 replaces this with
a `Settings` object). The handler is synchronous for clarity; the async port is a
stretch goal.

A small but important design choice: `agent.py` ships with a **deterministic
offline stub** path (`AGENT_FAKE=1`) so the test suite and the validation smoke
tests run without an API key or network. The real path uses the Anthropic SDK. The
two share one code path for everything except the single `messages.create` call.

## Reference implementation

### `pyproject.toml`

```toml
[project]
name = "support-agent"
version = "0.1.0"
description = "A small FastAPI service wrapping an LLM support agent."
requires-python = ">=3.10"

# Explicit pins. No requirements.txt — pyproject is the manifest.
dependencies = [
    "fastapi[standard]>=0.115,<1.0",
    "pydantic>=2.9,<3.0",
    "uvicorn>=0.30,<1.0",
    "anthropic>=0.39,<1.0",
]

[project.optional-dependencies]
test = [
    "pytest>=8.0",
    "httpx>=0.27",          # FastAPI TestClient depends on it
]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["."]
include = ["src*"]
```

### `src/__init__.py`

```python
"""support-agent — a small FastAPI service that wraps an LLM agent."""
```

### `src/errors.py`

```python
# src/errors.py
"""Named application errors.

The HTTP layer never sees a raw provider SDK exception. ``agent.py`` catches
those and re-raises one of these, and ``main.py`` maps each to a status code.
Keeping the taxonomy here (not inline in the handler) means two requests that
fail the same way always get the same status code.
"""
from __future__ import annotations


class MaxStepsExceeded(RuntimeError):
    """The agent loop ran out of steps before producing a final answer."""


class ProviderTimeout(RuntimeError):
    """The upstream model did not respond within the client timeout."""


class ProviderBadRequest(RuntimeError):
    """The upstream provider rejected our request (an HTTP 4xx from the API)."""
```

### `src/schemas.py`

```python
# src/schemas.py
"""Wire contract for the service.

These models ARE the validation layer. Every constraint here is enforced by
FastAPI before the route handler runs; a violation becomes a 422 the caller can
read and fix, and the model is never called with garbage.
"""
from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    # min_length=1 rejects {"question": ""} before the agent sees an empty prompt.
    # max_length=4000 bounds input size at the boundary so a 2 MB body can't
    # become a huge token bill.
    question: Annotated[str, Field(min_length=1, max_length=4000)]
    # Optional but bounded: an unbounded session_id is a free cache-key abuse.
    session_id: Annotated[str | None, Field(default=None, max_length=64)]
    # Clamped to the range the agent loop accepts. The handler never re-clamps;
    # Pydantic guarantees a value in [1, 12].
    max_steps: Annotated[int, Field(default=8, ge=1, le=12)]


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    steps: int


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    request_id: str
    usage: Usage
```

### `src/agent.py`

```python
# src/agent.py
"""The application layer: the agent loop and the only place the provider SDK
is imported. ``main.py`` calls ``run_agent`` and never touches Anthropic types.

This is a faithful but compact stand-in for a mod-105 driver. The contract is
the thing that matters: a string in, a dict out with ``final_text``, ``steps``,
and ``usage``. Swap the body for your real mod-105 loop; keep the signature and
the error mapping.
"""
from __future__ import annotations

import os

from .errors import MaxStepsExceeded, ProviderBadRequest, ProviderTimeout

SYSTEM = "You are a concise customer-support agent. Answer in one short paragraph."


def _run_fake(question: str, *, max_steps: int) -> dict:
    """Deterministic offline path for tests and no-key smoke runs.

    Triggered by AGENT_FAKE=1. A question containing the token 'LOOP' simulates
    an agent that never converges, so the 504 path is exercisable without burning
    real API calls.
    """
    if "LOOP" in question.upper():
        raise MaxStepsExceeded(f"exceeded max_steps={max_steps}")
    return {
        "final_text": f"(offline) You asked: {question}",
        "steps": 1,
        "usage": {"input_tokens": len(question.split()), "output_tokens": 12},
    }


def run_agent(
    question: str,
    *,
    max_steps: int = 8,
    session_id: str | None = None,
    request_id: str | None = None,
) -> dict:
    """Run the agent once and return a result dict.

    Returns ``{"final_text": str, "steps": int, "usage": {"input_tokens": int,
    "output_tokens": int}}``.

    Raises one of the named errors from ``errors.py`` on a known failure so the
    HTTP layer can map it to a status code without inspecting SDK types.
    """
    if os.environ.get("AGENT_FAKE") == "1":
        return _run_fake(question, max_steps=max_steps)

    # Imported lazily so the offline path (and module import) never require the
    # SDK. The real provider call is the ONLY thing that differs from the stub.
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key, timeout=30.0)

    messages = [{"role": "user", "content": question}]
    total_in = total_out = 0

    for step in range(1, max_steps + 1):
        try:
            resp = client.messages.create(
                model=os.environ.get("MODEL", "claude-sonnet-4-6"),
                max_tokens=1024,  # never accept the provider default unattended
                system=SYSTEM,
                messages=messages,
            )
        except anthropic.APITimeoutError as exc:
            raise ProviderTimeout("upstream model timed out") from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderTimeout("could not reach upstream model") from exc
        except anthropic.BadRequestError as exc:
            raise ProviderBadRequest("upstream provider rejected the request") from exc

        total_in += resp.usage.input_tokens
        total_out += resp.usage.output_tokens

        # A single-turn support agent: the first text response is the answer.
        # A real mod-105 loop would inspect resp.stop_reason for tool_use here
        # and continue; this compact version returns on the first text block.
        text = "".join(
            block.text for block in resp.content if block.type == "text"
        )
        if text:
            return {
                "final_text": text,
                "steps": step,
                "usage": {"input_tokens": total_in, "output_tokens": total_out},
            }
        messages.append({"role": "assistant", "content": resp.content})

    raise MaxStepsExceeded(f"exceeded max_steps={max_steps}")
```

### `src/main.py`

```python
# src/main.py
from __future__ import annotations

import os
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .agent import run_agent
from .errors import MaxStepsExceeded, ProviderBadRequest, ProviderTimeout
from .schemas import ChatRequest, ChatResponse, Usage

app = FastAPI(title="support-agent", version="0.1.0")

# Quick sanity check at startup; exercise-02 replaces this with Settings.
# AGENT_FAKE=1 lets the offline path boot without a key (used by tests).
if os.environ.get("AGENT_FAKE") != "1" and not os.environ.get("ANTHROPIC_API_KEY"):
    raise RuntimeError("ANTHROPIC_API_KEY is required")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    # No model, no network, no file. Microseconds. Do not make it do work.
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    request_id = uuid.uuid4().hex[:12]
    result = run_agent(
        req.question,
        max_steps=req.max_steps,
        session_id=req.session_id,
        request_id=request_id,
    )
    return ChatResponse(
        answer=result["final_text"],
        session_id=req.session_id or request_id,
        request_id=request_id,
        usage=Usage(
            input_tokens=result["usage"]["input_tokens"],
            output_tokens=result["usage"]["output_tokens"],
            steps=result["steps"],
        ),
    )


# --- exception handlers: one decision each (status code, error class) ---


@app.exception_handler(MaxStepsExceeded)
def _max_steps(request: Request, exc: MaxStepsExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=504,
        content={
            "detail": "agent did not converge within step cap",
            "error_class": "max_steps_exceeded",
        },
    )


@app.exception_handler(ProviderTimeout)
def _provider_timeout(request: Request, exc: ProviderTimeout) -> JSONResponse:
    return JSONResponse(
        status_code=504,
        content={
            "detail": "upstream model timed out",
            "error_class": "provider_timeout",
        },
    )


@app.exception_handler(ProviderBadRequest)
def _provider_bad_request(request: Request, exc: ProviderBadRequest) -> JSONResponse:
    return JSONResponse(
        status_code=502,
        content={
            "detail": "upstream provider rejected the request",
            "error_class": "provider_bad_request",
        },
    )


@app.exception_handler(Exception)
def _unexpected(request: Request, exc: Exception) -> JSONResponse:
    # Honest but blank: no traceback on the wire (it can leak paths or prompts).
    # The traceback belongs in the logs (exercise-03).
    return JSONResponse(
        status_code=500,
        content={
            "detail": "internal server error",
            "error_class": "unexpected",
        },
    )
```

### `tests/test_chat.py`

A test file the exercise does not strictly require but that proves the boundary
without spending a cent (it runs against the `AGENT_FAKE=1` offline path):

```python
# tests/test_chat.py
import os

os.environ["AGENT_FAKE"] = "1"  # set before importing the app

from fastapi.testclient import TestClient  # noqa: E402

from src.main import app  # noqa: E402

client = TestClient(app, raise_server_exceptions=False)


def test_healthz_ok():
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_chat_happy_path():
    r = client.post("/chat", json={"question": "What is your return policy?"})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"answer", "session_id", "request_id", "usage"}
    assert set(body["usage"]) == {"input_tokens", "output_tokens", "steps"}


def test_empty_question_is_422():
    r = client.post("/chat", json={"question": ""})
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"] == ["body", "question"]


def test_missing_question_is_422():
    r = client.post("/chat", json={})
    assert r.status_code == 422


def test_max_steps_out_of_range_is_422():
    r = client.post("/chat", json={"question": "x", "max_steps": 99})
    assert r.status_code == 422


def test_unconverged_agent_is_504():
    r = client.post("/chat", json={"question": "please LOOP forever", "max_steps": 2})
    assert r.status_code == 504
    assert r.json()["error_class"] == "max_steps_exceeded"
```

### `README.md` (the repo's own, with `curl` examples)

````markdown
# support-agent

A small FastAPI service wrapping an LLM support agent.

## Run

```bash
pip install -e ".[test]"
export ANTHROPIC_API_KEY=sk-ant-...
fastapi dev src/main.py
```

## Routes

```bash
# health
curl -s localhost:8000/healthz
# {"status":"ok"}

# chat (200)
curl -s -X POST localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"question":"What is your return policy?"}'

# validation error (422) — empty question
curl -s -X POST localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"question":""}'
```
````

## Meeting the acceptance criteria

| Criterion | Where it is met |
|-----------|-----------------|
| `fastapi dev src/main.py` serves `/healthz` and `/chat` | `src/main.py` declares both routes |
| `runs.md` captures the eight smoke-test requests | reproduce with the commands below; sample output in *Verification* |
| `NOTES.md` has the three task-6 paragraphs | template in *Verification* |
| `openapi.json` shows `min_length`/`max_length`/`ge`/`le` | generated from `schemas.py`; `jq` check in *Verification* |
| `def chat(...)` body is < ~15 lines, no `try/except` | `src/main.py` handler is 11 lines, raising is in `agent.py` |

The handler body is deliberately five operations (request → request_id → call →
build response → return). All four named failure modes map to status codes in the
exception handlers, and the catch-all turns anything unanticipated into a blank `500`.

## Common pitfalls

1. **Importing the provider SDK in `main.py`.** The handler must not know it is
   talking to Anthropic. Keep `import anthropic` inside `agent.py`; the HTTP layer
   imports only `run_agent`. The moment `main.py` catches an `anthropic.*`
   exception, your error map is coupled to one provider.
2. **Wrapping the route in `try/except`.** The exercise explicitly forbids this.
   Raising belongs in `agent.py`; mapping belongs in the registered handlers. A
   `try/except` in the handler silently competes with them and makes the status
   codes unpredictable.
3. **Re-clamping `max_steps` in the handler.** Pydantic already guarantees
   `1 <= max_steps <= 12`. Defensive re-clamping is dead code that hides the fact
   the validation already happened at the boundary.
4. **Returning a bare `dict` instead of `ChatResponse`.** Without
   `response_model=ChatResponse`, internal fields (`trace`, debug data) leak onto
   the wire and a refactor that drops a field fails silently at runtime instead of
   at startup.
5. **Making `/healthz` do work.** Calling the model or hitting the network in the
   health check ties your liveness probe to upstream health — every provider blip
   then marks your container unhealthy.

## Verification

Run the offline test suite (no key, no network, no spend):

```bash
pip install -e ".[test]"
AGENT_FAKE=1 pytest -q
# 6 passed
```

Drive the live service and capture `runs.md`:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
fastapi dev src/main.py &
sleep 2

curl -s localhost:8000/healthz                                   # 200 {"status":"ok"}
curl -s -X POST localhost:8000/chat -H 'content-type: application/json' \
     -d '{"question":"What is your return policy?"}'             # 200 ChatResponse
curl -s -o /dev/null -w '%{http_code}\n' -X POST localhost:8000/chat \
     -H 'content-type: application/json' -d '{"question":""}'    # 422
curl -s -o /dev/null -w '%{http_code}\n' -X POST localhost:8000/chat \
     -H 'content-type: application/json' -d '{}'                 # 422
curl -s -o /dev/null -w '%{http_code}\n' -X POST localhost:8000/chat \
     -H 'content-type: application/json' \
     -d '{"question":"x","max_steps":99}'                        # 422
```

Confirm the OpenAPI schema carries the constraints:

```bash
curl -s localhost:8000/openapi.json \
  | jq '.components.schemas.ChatRequest.properties.question'
# { "type": "string", "maxLength": 4000, "minLength": 1, "title": "Question" }

curl -s localhost:8000/openapi.json \
  | jq '.components.schemas.ChatRequest.properties.max_steps'
# { "type": "integer", "maximum": 12, "minimum": 1, "default": 8, ... }
```

`NOTES.md` template (fill in from your own run):

```text
## Validation surface
Tests 3, 4, and 5 would each have crashed downstream without Pydantic. Test 3
({"question": ""}) reaches client.messages.create with an empty user message;
test 4 ({}) raises KeyError on req.question construction; test 5 (max_steps=99)
spins the loop 99 times. Pydantic rejects all three at the boundary with 422.

## The handler body
~11 lines: request_id, run_agent, ChatResponse. Prompt construction, the loop,
and error raising all live in agent.py. Nothing about Anthropic is visible here.

## The docs page
/docs renders every constraint and lets a teammate "Try it out". It is missing
the auth story (there is none yet) and does not show the 504/502/500 bodies the
exception handlers return, since those are not declared as responses.
```

## Reflection notes

1. **Validating `max_steps` in the handler instead of Pydantic.** A future
   refactor that reorders the handler could call `run_agent` before the manual
   check; a test asserting the 422 would have to mock the handler internals
   instead of just POSTing a body; and in production the clamp could silently
   coerce `99 -> 12` and hide a client bug. Pydantic's boundary rejection makes
   the contract visible in `/openapi.json` and impossible to skip.
2. **A bare `dict` from the handler.** `response_model=ChatResponse` stops you
   leaking `trace`, intermediate `steps`, raw provider objects, or a stray
   `api_key` that ended up in the result dict.
3. **A status code that should not be `500`.** A caller sending
   `content-type: text/plain` instead of JSON should get `415 Unsupported Media
   Type`, not `500` — it is the caller's mistake, not a server bug.
