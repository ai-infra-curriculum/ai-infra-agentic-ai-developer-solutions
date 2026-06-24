# mod-106-llm-app-deployment/exercise-03-logging-and-usage-guardrail — Solution

## Approach

This exercise takes the configured-but-quiet service from exercise-02 and makes it
observable, bounded, and shippable. Three additions:

- **Structured logging.** A `JsonFormatter` emits one JSON document per line to
  stdout. A `@app.middleware("http")` stamps a `request_id`, times the request, and
  emits **exactly one** line per request — including `422`s that never reach the
  handler. The handler stashes its specific fields onto `request.state.extra_log`;
  the middleware merges them in. Prompts are never logged raw: a SHA-256 prefix, a
  char count, and an optional 200-char preview gated on `LOG_PROMPT_PREVIEW`.
- **A daily-spend guardrail.** `DailySpendGuardrail` is a locked, in-memory counter
  that resets on calendar-day change and raises `GuardrailTripped` once the cap is
  hit. The handler **pre-checks** (so a tripped cap blocks the call) and
  **post-charges** (so the actual estimate informs the next request). `prices.py`
  estimates cost from token counts using rates pinned from the live pricing page
  with a date.
- **A multi-stage Dockerfile.** Two stages, `python:3.12-slim`, non-root `app` user,
  a Python-only `HEALTHCHECK` against `/healthz`, `PYTHONUNBUFFERED=1`, and uvicorn
  as PID 1. A `.dockerignore` keeps `.env`, `.git`, tests, and caches out of the
  image so no secret is ever baked in.

The middleware approach is chosen over per-handler logging precisely because a
Pydantic `422` short-circuits before the handler runs; only middleware sees every
request uniformly.

## Reference implementation

### `src/logging_config.py`

```python
# src/logging_config.py
"""One JSON line per log event, written to stdout.

The formatter surfaces everything passed via extra={...} and drops the noisy
internal LogRecord fields so a line carries only what we put on it.
"""
from __future__ import annotations

import json
import logging
import sys

# LogRecord attributes we never want on the wire.
_DROP = {
    "args", "msg", "levelno", "pathname", "filename", "module", "exc_info",
    "exc_text", "stack_info", "lineno", "funcName", "created", "msecs",
    "relativeCreated", "thread", "threadName", "processName", "process",
    "name", "taskName",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S.%03dZ"),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _DROP or key == "message":
                continue
            payload[key] = value
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    # Silence per-request library chatter so the stream stays one-line-per-request.
    logging.getLogger("httpx").setLevel("WARNING")
    logging.getLogger("anthropic").setLevel("WARNING")
    logging.getLogger("uvicorn.access").setLevel("WARNING")
```

### `src/prices.py`

```python
# src/prices.py
"""Token -> USD estimation.

Rates are the standard (non-cached, non-batch) per-million-token Claude API
prices, copied from the Anthropic pricing page. They are an ESTIMATE for the log
line and the guardrail, not your invoice. Re-confirm on every release.

Source:  https://platform.claude.com/docs/en/about-claude/pricing
Checked: 2026-06-24
  Claude Sonnet 4.x (incl. claude-sonnet-4-6):  $3.00 / MTok in,  $15.00 / MTok out
  Claude Haiku 4.5:                              $1.00 / MTok in,   $5.00 / MTok out
"""
from __future__ import annotations

PRICES_USD_PER_MTOK: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6": {"in": 3.00, "out": 15.00},
    "claude-sonnet-4-5": {"in": 3.00, "out": 15.00},
    "claude-haiku-4-5": {"in": 1.00, "out": 5.00},
}


def estimate_cost(usage: dict, model: str) -> float:
    """Estimate request cost in USD. Returns 0.0 for an unpriced model id."""
    rates = PRICES_USD_PER_MTOK.get(model)
    if not rates or rates.get("in") is None:
        return 0.0
    return (
        usage["input_tokens"] * rates["in"] / 1_000_000
        + usage["output_tokens"] * rates["out"] / 1_000_000
    )
```

### `src/errors.py` (gains `GuardrailTripped`)

```python
# src/errors.py  (added in this exercise)
class GuardrailTripped(RuntimeError):
    """The process-level daily spend cap has been reached."""
```

### `src/guardrail.py`

```python
# src/guardrail.py
"""A process-level daily-spend ceiling. In-memory, single-process, fails closed.

Known limits (see chapter 5): does not survive a restart, does not share state
across workers, does not stop a burst inside one request. It is the floor that
catches the runaway you didn't write before the bill arrives.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from functools import lru_cache
from threading import Lock

from .errors import GuardrailTripped
from .settings import get_settings


@dataclass
class DailySpendGuardrail:
    cap_usd: float
    spend_usd: float = 0.0
    day: dt.date = field(default_factory=dt.date.today)
    _lock: Lock = field(default_factory=Lock)

    def check_and_charge(self, estimated_cost_usd: float) -> None:
        """Reset on day change, raise if the cap is hit, then add the charge.

        Call with 0.0 to pre-check (blocks the next request after the cap trips)
        and again with the real estimate to post-charge.
        """
        with self._lock:
            today = dt.date.today()
            if today != self.day:
                self.day = today
                self.spend_usd = 0.0
            if self.spend_usd >= self.cap_usd:
                raise GuardrailTripped(
                    f"daily cap of ${self.cap_usd:.2f} reached"
                )
            self.spend_usd += estimated_cost_usd


@lru_cache
def get_guardrail() -> DailySpendGuardrail:
    return DailySpendGuardrail(cap_usd=get_settings().daily_usd_cap)
```

### `src/settings.py` (two new fields)

```python
# src/settings.py  (added in this exercise)
    # --- configuration ---
    daily_usd_cap: float = Field(default=5.0, ge=0, alias="DAILY_USD_CAP")

    # --- environment ---
    log_prompt_preview: bool = Field(default=False, alias="LOG_PROMPT_PREVIEW")
```

### `src/main.py` (final shape)

```python
# src/main.py
from __future__ import annotations

import hashlib
import logging
import time
import uuid

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from .agent import run_agent
from .errors import (
    GuardrailTripped,
    MaxStepsExceeded,
    ProviderBadRequest,
    ProviderTimeout,
)
from .guardrail import DailySpendGuardrail, get_guardrail
from .logging_config import configure_logging
from .prices import estimate_cost
from .schemas import ChatRequest, ChatResponse, Usage
from .settings import Settings, get_settings

settings = get_settings()
configure_logging(settings.log_level)  # configured BEFORE app is built
log = logging.getLogger("app")

app = FastAPI(
    title="support-agent",
    version="0.1.0",
    docs_url="/docs" if settings.docs_enabled else None,
    redoc_url="/redoc" if settings.docs_enabled else None,
)


def prompt_metadata(prompt: str, preview: bool) -> dict:
    """Hash + length + optional 200-char preview. Never the raw prompt by default."""
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
    payload = {"prompt_sha256_16": digest, "prompt_chars": len(prompt)}
    if preview:
        payload["prompt_preview"] = prompt[:200]
    return payload


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    request.state.extra_log = {}
    started = time.monotonic()

    response = await call_next(request)

    fields = {
        "request_id": request_id,
        "endpoint": f"{request.method} {request.url.path}",
        "status_code": response.status_code,
        "latency_ms": int((time.monotonic() - started) * 1000),
        **getattr(request.state, "extra_log", {}),
    }
    # Default outcome if the handler did not set one (e.g. a 422 from Pydantic).
    fields.setdefault(
        "outcome",
        "ok"
        if response.status_code < 400
        else "validation_error"
        if response.status_code == 422
        else "guardrail_tripped"
        if response.status_code == 429
        else "error",
    )
    level = logging.WARNING if response.status_code == 429 else logging.INFO
    log.log(level, "chat_request", extra=fields)
    return response


@app.on_event("startup")
def on_startup() -> None:
    log.info(
        "startup",
        extra={
            "env": settings.env,
            "model": settings.model,
            "max_steps": settings.max_steps,
            "daily_usd_cap": settings.daily_usd_cap,
            "anthropic_api_key_set": bool(settings.anthropic_api_key.get_secret_value()),
        },
    )


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(
    request: Request,
    req: ChatRequest,
    settings: Settings = Depends(get_settings),
    guardrail: DailySpendGuardrail = Depends(get_guardrail),
) -> ChatResponse:
    guardrail.check_and_charge(0.0)  # pre-check: trips if cap already hit

    result = run_agent(
        req.question,
        max_steps=req.max_steps,
        session_id=req.session_id,
        request_id=request.state.request_id,
        api_key=settings.anthropic_api_key.get_secret_value(),
        model=settings.model,
    )

    cost = estimate_cost(result["usage"], settings.model)
    guardrail.check_and_charge(cost)  # post-charge: record the actual estimate

    request.state.extra_log = {
        "session_id": req.session_id,
        "model": settings.model,
        "input_tokens": result["usage"]["input_tokens"],
        "output_tokens": result["usage"]["output_tokens"],
        "steps": result["steps"],
        "estimated_cost_usd": cost,
        "outcome": "ok",
        **prompt_metadata(req.question, settings.log_prompt_preview),
    }

    return ChatResponse(
        answer=result["final_text"],
        session_id=req.session_id or request.state.request_id,
        request_id=request.state.request_id,
        usage=Usage(
            input_tokens=result["usage"]["input_tokens"],
            output_tokens=result["usage"]["output_tokens"],
            steps=result["steps"],
        ),
    )


# --- exception handlers ---


@app.exception_handler(MaxStepsExceeded)
def _max_steps(request: Request, exc: MaxStepsExceeded) -> JSONResponse:
    request.state.extra_log = {"outcome": "max_steps_exceeded",
                               "error_class": "max_steps_exceeded"}
    return JSONResponse(status_code=504, content={
        "detail": "agent did not converge within step cap",
        "error_class": "max_steps_exceeded",
    })


@app.exception_handler(ProviderTimeout)
def _provider_timeout(request: Request, exc: ProviderTimeout) -> JSONResponse:
    request.state.extra_log = {"outcome": "provider_timeout",
                               "error_class": "provider_timeout"}
    return JSONResponse(status_code=504, content={
        "detail": "upstream model timed out",
        "error_class": "provider_timeout",
    })


@app.exception_handler(ProviderBadRequest)
def _provider_bad_request(request: Request, exc: ProviderBadRequest) -> JSONResponse:
    request.state.extra_log = {"outcome": "provider_bad_request",
                               "error_class": "provider_bad_request"}
    return JSONResponse(status_code=502, content={
        "detail": "upstream provider rejected the request",
        "error_class": "provider_bad_request",
    })


@app.exception_handler(GuardrailTripped)
def _guardrail(request: Request, exc: GuardrailTripped) -> JSONResponse:
    request.state.extra_log = {"outcome": "guardrail_tripped",
                               "error_class": "guardrail_tripped"}
    # Polite: tell the caller when the budget resets (midnight UTC).
    return JSONResponse(
        status_code=429,
        headers={"Retry-After": "86400"},
        content={"detail": "daily usage cap reached",
                 "error_class": "guardrail_tripped"},
    )
```

### `Dockerfile`

```dockerfile
# ---------- stage 1: build dependencies in a fat image ----------
FROM python:3.12-slim AS build
ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the manifest first so a code-only change reuses the cached install layer.
COPY pyproject.toml ./
COPY src/ ./src/
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install .

# ---------- stage 2: runtime image (no compilers) ----------
FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

RUN groupadd --system app && useradd --system --gid app --home /app app

WORKDIR /app
COPY --from=build /opt/venv /opt/venv
COPY --chown=app:app src/ ./src/

USER app
EXPOSE 8000

# Python-only probe; no need to install curl into the image.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz',timeout=2).status==200 else 1)"

# Exec form -> uvicorn is PID 1 -> docker stop reaches it for a clean shutdown.
# DO NOT add `COPY .env .` above this line — that would bake the secret into the
# image. Secrets arrive at run time via --env-file only.
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### `.dockerignore`

```gitignore
.env
.env.*
!.env.example

.git
.gitignore

__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/

.venv/
venv/

tests/
*.md
.DS_Store
.vscode/
.idea/
```

### `tests/test_guardrail.py`

```python
# tests/test_guardrail.py
import pytest

from src.errors import GuardrailTripped
from src.guardrail import DailySpendGuardrail


def test_trips_when_cap_reached():
    g = DailySpendGuardrail(cap_usd=0.01)
    g.check_and_charge(0.0)   # pre-check: under cap, ok
    g.check_and_charge(0.02)  # post-charge: now over cap
    with pytest.raises(GuardrailTripped):
        g.check_and_charge(0.0)  # next pre-check trips


def test_estimate_cost_known_model():
    from src.prices import estimate_cost

    cost = estimate_cost({"input_tokens": 1_000_000, "output_tokens": 1_000_000},
                         "claude-sonnet-4-6")
    assert cost == pytest.approx(18.0)  # $3 in + $15 out per MTok
```

## Meeting the acceptance criteria

| Criterion | Where it is met |
|-----------|-----------------|
| One JSON line per `/chat` with the required fields | `log_requests` middleware + handler `extra_log` |
| Working `DailySpendGuardrail` returning `429` | `src/guardrail.py` + `_guardrail` handler |
| Dockerfile < ~250 MB, non-root, `HEALTHCHECK` | `Dockerfile` (multi-stage, `USER app`) |
| `.dockerignore` excludes `.env`, `.git`, `__pycache__`, tests | `.dockerignore` |
| `runs.md` with the 10 smoke-test steps | reproduce per *Verification* |
| `NOTES.md` with the three task-9 paragraphs | template in *Verification* |
| teammate-runnable in < 5 minutes | runbook in *Verification* |

The log schema carries `request_id`, `model`, `input_tokens`, `output_tokens`,
`latency_ms`, `outcome`, `status_code`, and `estimated_cost_usd`. The guardrail
emits `outcome: guardrail_tripped` at `WARNING`. The prompt is never logged raw —
only a hash, a length, and an opt-in preview.

## Common pitfalls

1. **`COPY .env .` in the Dockerfile.** The single line that silently breaks secret
   isolation. The `.dockerignore` excludes `.env`, but a hand-written `COPY .env .`
   overrides that intent and bakes the key into a shareable image. The comment above
   the `CMD` line flags it for the next maintainer.
2. **Logging the raw prompt by default.** `LOG_PROMPT_PREVIEW` defaults to `False`.
   Hash + length + optional preview is the floor; flipping the default to `True`
   turns every log line into a PII store with all the regulatory weight that brings.
3. **Emitting two log lines per request.** If you log inside the handler *and* in
   the middleware, every request produces two rows and your `sum(estimated_cost_usd)`
   double-counts. The handler stashes onto `request.state`; only the middleware emits.
4. **Dropping the `Lock`.** Under concurrent requests the read-modify-write on
   `spend_usd` races: two requests both read a sub-cap value, both proceed, and the
   cap is overshot. The lock is cheap and the only thing making the counter correct.
5. **Charging a guessed token count instead of the provider's.** `input_tokens` /
   `output_tokens` come back on the response. Pre-counting with a local tokenizer to
   "charge" drifts from the invoice; charge what the provider reported.
6. **Running uvicorn as root** or using the shell form of `CMD` (signals then never
   reach uvicorn, so `docker stop` hangs for 10s before a SIGKILL).

## Verification

```bash
# Offline tests (no key, no network):
AGENT_FAKE=1 pytest -q
# guardrail + prices + settings + chat tests pass

# 1. Build:
docker build -t support-agent:0.1 .

# 2. Image size (expect well under 250 MB):
docker images support-agent --format '{{.Size}}'

# 3. Non-root:
docker run --rm support-agent:0.1 whoami
# app

# 4-5. Health + happy path:
docker run --rm -d -p 8000:8000 --env-file .env --name agent support-agent:0.1
curl -s localhost:8000/healthz                       # {"status":"ok"}
curl -s -X POST localhost:8000/chat \
     -H 'content-type: application/json' -d '{"question":"hi"}'

# 6. Validation produces exactly one JSON line:
curl -s -o /dev/null -w '%{http_code}\n' -X POST localhost:8000/chat \
     -H 'content-type: application/json' -d '{"question":""}'     # 422
docker logs agent 2>&1 | jq -c 'select(.event=="chat_request")' | tail -1

# 7. Guardrail trip (set DAILY_USD_CAP=0.001 in .env, restart, hit twice):
#   first  -> 200
#   second -> 429  {"error_class":"guardrail_tripped"}

# 8. Restart resets the counter (in-memory tradeoff):
docker restart agent
#   next /chat -> 200 again (counter back to zero)

# 9. No secret in the image:
docker history support-agent:0.1 --no-trunc | grep -i anthropic   # (nothing)
docker run --rm support-agent:0.1 env | grep ANTHROPIC            # (nothing without --env-file)

# 10. The log stream is parseable JSONL:
docker logs agent 2>&1 | jq -c 'select(.event=="chat_request") | {request_id, outcome, latency_ms}'

docker rm -f agent
```

Teammate runbook (the README section for the repo):

```bash
cp .env.example .env       # fill in ANTHROPIC_API_KEY
docker build -t support-agent:0.1 .
docker run --rm -p 8000:8000 --env-file .env support-agent:0.1
curl localhost:8000/healthz
curl -X POST localhost:8000/chat -H 'content-type: application/json' \
     -d '{"question":"hi"}'
```

`NOTES.md` template:

```text
The fields you logged
At 2 a.m. with a bill spike, I'd group log lines by day and sum
estimated_cost_usd to confirm the spike, then bucket by session_id (or prompt
hash) to find the heavy caller, then look at steps and output_tokens to see
whether it was many requests or a few runaway loops. I'd add a `provider` field
the day we add a second one, and a hashed user_ip for abuse triage. I'd drop
nothing — every field above answers a real question.

The prompt-and-PII decision
Before flipping LOG_PROMPT_PREVIEW on in prod I'd ask the PM to agree to a
retention window, a deletion path, an access-control list on the log store, and
sign-off from whoever owns GDPR/CCPA exposure — because full prompts turn the log
store into a system-of-record for PII.

The guardrail tradeoff
Restart: the counter resets, so a crash-loop re-grants the full budget each boot;
fix with a sqlite-backed counter keyed by day. Multi-worker: each worker counts
independently, so N workers allow N×cap; fix with a shared store (Redis/sqlite).
Distributed denial-of-funds: a burst drains the cap inside one window before the
next pre-check sees it; fix with a per-key token-bucket rate limit in front.
```

## Reflection notes

1. **Estimate off by 2×.** It happens when the rate table is stale (a price
   change), when a different model is actually called than the one logged, or when
   the request used a feature with different economics (cache reads at 0.1×, batch
   at 0.5×, the 1M-context tier). Operationally, a sustained 2× gap between
   estimated and invoiced spend is a signal to re-pull the pricing page and audit
   which model id and features each request used.
2. **A teammate flips `LOG_PROMPT_PREVIEW=true` in prod.** Cleanup: flip it off,
   rotate the log retention to purge the affected window, audit any downstream sink
   (aggregator, backup) that ingested the previews, and document the exposure. The
   smallest preventive change: gate the flag on `ENV != "prod"` in code so it
   physically cannot be enabled in production without a code change and review.
3. **Deleting the `Lock` under 20 concurrent curls at `DAILY_USD_CAP=0.10`.** The
   read-modify-write races: several requests read a sub-cap `spend_usd`
   simultaneously, all pass the check, all charge, and the cap is overshot — you
   serve and pay for more than the budget. The user would not notice (their request
   succeeds); you would, on the invoice.
4. **The Dockerfile line that breaks secret isolation.** Any `COPY .env .` (or a
   `COPY . .` that the `.dockerignore` fails to cover) bakes the key into a shared
   image. The reference Dockerfile copies only `src/` and carries a comment above
   `CMD` warning against adding a `.env` copy.
