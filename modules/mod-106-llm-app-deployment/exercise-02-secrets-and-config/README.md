# mod-106-llm-app-deployment/exercise-02-secrets-and-config — Solution

## Approach

Exercise-01 read `ANTHROPIC_API_KEY` directly from `os.environ`. This exercise
centralizes every environment read into one `pydantic-settings` `Settings` object
and proves four properties:

- **Required-missing fails loud at startup.** A field with no `default=` is
  required; if the env var is absent, `Settings()` raises `ValidationError` and the
  process dies before serving a request. Better one second at boot than one day at
  a customer report.
- **Optional-missing uses a default.** Fields with `default=` boot fine and the
  startup line shows the value that got used.
- **Typos are rejected.** `extra="forbid"` turns `ANTROPIC_API_KEY` (a typo) into a
  hard startup failure instead of a silently-ignored mystery.
- **Secrets never leak.** `SecretStr` redacts the key from `repr()`, the startup
  line logs only `anthropic_api_key_set: bool`, and the value is retrieved through
  the grep-able `.get_secret_value()` verb.

The wiring change is mechanical: every `os.environ.get(...)` becomes a setting read,
the agent constructs its client with an explicit `api_key=` argument (so a test can
inject a fake), and the handler pulls `Settings` via `Depends(get_settings)` so tests
can override it with `app.dependency_overrides`.

`get_settings` is `@lru_cache`d: it validates once, then returns the cached instance.
Tests clear it with `get_settings.cache_clear()`.

## Reference implementation

### `src/settings.py`

```python
# src/settings.py
"""One typed Settings object for the whole app.

Categories are labeled in the field groups below. Secrets use SecretStr;
configuration and environment flags have defaults and are safe to log. Nothing
else in the codebase calls os.environ — this module is the single boundary.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",  # a mistyped env var is a typo, not a feature
    )

    # --- secrets (never logged; SecretStr-wrapped) ---
    anthropic_api_key: SecretStr = Field(alias="ANTHROPIC_API_KEY")

    # --- configuration (defaults; safe to log) ---
    model: str = Field(default="claude-sonnet-4-6", alias="MODEL")
    max_steps: int = Field(default=8, ge=1, le=12, alias="MAX_STEPS")
    request_timeout_s: int = Field(default=30, ge=1, le=120, alias="REQUEST_TIMEOUT_S")

    # --- environment flags (defaults; safe to log) ---
    env: str = Field(default="dev", alias="ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    docs_enabled: bool = Field(default=True, alias="DOCS_ENABLED")


@lru_cache
def get_settings() -> Settings:
    """Validate once, cache forever. Tests clear with get_settings.cache_clear()."""
    return Settings()
```

`DAILY_USD_CAP` is intentionally *not* here — it arrives in exercise-03 when the
guardrail code first reads it. Adding a setting before the code reads it violates
the "add a row when the app actually reads it" rule.

### `src/agent.py` (the wiring change)

The agent now receives `api_key` and `model` explicitly instead of reading the
environment. Only the signature and the client construction change from
exercise-01:

```python
# src/agent.py  (excerpt — the parts that changed from exercise-01)
def run_agent(
    question: str,
    *,
    max_steps: int = 8,
    session_id: str | None = None,
    request_id: str | None = None,
    api_key: str | None = None,        # NEW: injected, not read from env
    model: str = "claude-sonnet-4-6",  # NEW: injected from Settings
) -> dict:
    if os.environ.get("AGENT_FAKE") == "1":
        return _run_fake(question, max_steps=max_steps)

    import anthropic

    # Explicit key: makes it obvious which credential is used and lets a test
    # inject a different one. We do NOT rely on the SDK's env-var fallback.
    client = anthropic.Anthropic(api_key=api_key, timeout=30.0)
    # ... loop body unchanged, but the create() call uses the injected `model` ...
```

### `src/main.py` (the wiring change)

```python
# src/main.py  (excerpt — the parts that changed from exercise-01)
from fastapi import Depends, FastAPI, Request

from .settings import Settings, get_settings

settings = get_settings()  # validates at import; fails the process if invalid

app = FastAPI(
    title="support-agent",
    version="0.1.0",
    # The config flag drives whether docs are mounted — not `if env == "prod"`.
    docs_url="/docs" if settings.docs_enabled else None,
    redoc_url="/redoc" if settings.docs_enabled else None,
)


@app.on_event("startup")
def on_startup() -> None:
    s = get_settings()
    # Note the last field: a boolean, never the value. This is the line the next
    # maintainer reads at 2 a.m. when the app boots without a key.
    print(
        {
            "event": "startup",
            "env": s.env,
            "model": s.model,
            "max_steps": s.max_steps,
            "request_timeout_s": s.request_timeout_s,
            "log_level": s.log_level,
            "docs_enabled": s.docs_enabled,
            "anthropic_api_key_set": bool(s.anthropic_api_key.get_secret_value()),
        }
    )


@app.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    settings: Settings = Depends(get_settings),
) -> ChatResponse:
    request_id = uuid.uuid4().hex[:12]
    result = run_agent(
        req.question,
        max_steps=req.max_steps,
        session_id=req.session_id,
        request_id=request_id,
        api_key=settings.anthropic_api_key.get_secret_value(),
        model=settings.model,
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
```

### `.gitignore`

```gitignore
# Secrets — never commit. Add this BEFORE you create .env.
.env
.env.local
*.env.local

# Python
__pycache__/
*.pyc
.venv/
venv/
.pytest_cache/
```

### `.env.example` (committed contract)

```bash
# Required
ANTHROPIC_API_KEY=replace-me-with-your-anthropic-key

# Optional (these defaults are fine)
# MODEL=claude-sonnet-4-6
# MAX_STEPS=8
# REQUEST_TIMEOUT_S=30
# LOG_LEVEL=INFO
# ENV=dev
# DOCS_ENABLED=true
```

### `tests/test_settings.py`

```python
# tests/test_settings.py
import pytest
from pydantic import ValidationError

from src.settings import Settings, get_settings


def test_required_field_missing(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    get_settings.cache_clear()
    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # don't read .env in this test


def test_secret_str_redacts():
    s = Settings(ANTHROPIC_API_KEY="sk-test-key-not-real", _env_file=None)
    assert "sk-test-key-not-real" not in repr(s)
    assert s.anthropic_api_key.get_secret_value() == "sk-test-key-not-real"


def test_typo_rejected():
    with pytest.raises(ValidationError):
        Settings(ANTHROPIC_API_KEY="x", ANTROPIC_API_KEY="oops", _env_file=None)
```

## Meeting the acceptance criteria

| Criterion | Where it is met |
|-----------|-----------------|
| `Settings` validates every field; `SecretStr` for every secret | `src/settings.py` |
| `.env` in `.gitignore`, `.env.example` checked in | `.gitignore`, `.env.example` |
| `runs.md` with all seven failure-mode captures | reproduce per *Verification* |
| `tests/test_settings.py` with at least 3 passing tests | missing-required, redacted-repr, typo-rejected |
| `NOTES.md` with the three task-7 paragraphs | template in *Verification* |
| no `os.environ` outside `settings.py` / test helpers | `grep` check in *Verification* |

## Common pitfalls

1. **Forgetting `_env_file=None` in tests.** Without it, `Settings()` in a test
   reads your real `.env`, so `test_required_field_missing` finds the key you
   deleted from the environment still present in the file — and the test that
   should fail passes. In CI (no `.env` present) the behavior differs again.
2. **Using `str` instead of `SecretStr` for the key.** A plain `str` shows up in
   `repr(settings)`, in a `print(settings)`, and in any traceback that formats the
   object — exactly the leak the exercise is testing against.
3. **Leaving a `model` fallback in `agent.py`** (`os.environ.get("MODEL", ...)`).
   Either the value is in `Settings` or it is not configurable. A second source of
   truth means the startup log and the actual call can disagree.
4. **Disabling `extra="forbid"` to silence a third-party env var.** This re-opens
   the typo foot-gun. Rename your config or use a prefix instead.
5. **Logging the key value instead of a boolean.** `anthropic_api_key_set: True`
   is a time-saver that reveals nothing; logging the actual key is the most
   embarrassing leak there is.

## Verification

```bash
# Tests (no key required; _env_file=None isolates them):
pytest -q tests/test_settings.py
# 3 passed

# [1] Required-secret-present:
printf 'ANTHROPIC_API_KEY=sk-ant-real\n' > .env
fastapi dev src/main.py    # boots; startup line shows anthropic_api_key_set: True

# [2] Required-secret-missing:
printf '' > .env
ANTHROPIC_API_KEY= fastapi dev src/main.py
# ValidationError: anthropic_api_key — Field required ; exit 1

# [5] Typo detected:
printf 'ANTHROPIC_API_KEY=x\nANTROPIC_API_KEY=oops\n' > .env
fastapi dev src/main.py
# ValidationError: extra_forbidden — ANTROPIC_API_KEY ; exit 1

# [6] Override at run time:
printf 'ANTHROPIC_API_KEY=sk-ant-real\n' > .env
LOG_LEVEL=DEBUG fastapi dev src/main.py
# startup line shows log_level: DEBUG (env beats .env)

# [7] Secret never leaks:
python -c "from src.settings import Settings; \
print(Settings(ANTHROPIC_API_KEY='sk-secret', _env_file=None))"
# anthropic_api_key=SecretStr('**********') ...

# .gitignore actually blocks .env:
git check-ignore -v .env
# .gitignore:2:.env  .env

# os.environ confined to settings.py (+ agent's AGENT_FAKE test hook):
grep -rn "os.environ" src/ | grep -v "settings.py"
# src/agent.py: only the AGENT_FAKE test-mode check
```

`NOTES.md` template:

```text
Required vs. optional
anthropic_api_key is required: a missing key must fail at boot, not on the first
real request. model/max_steps/timeout/log_level/docs_enabled all have sane
defaults, so a teammate with only a key in .env still boots. The riskiest guess
is request_timeout_s — too low and legitimate slow requests 504; raise it on
evidence.

The extra="forbid" test
With extra="allow", ANTROPIC_API_KEY would have loaded as an ignored extra,
anthropic_api_key would stay unset, and the app would fail at boot (still
required) — or, worse, if someone had added a default, boot and 502 on every
/chat because the real key was never read. forbid turns a silent typo into a
loud stop.

The SecretStr redaction
"Not in repr()" covers print(settings) and most tracebacks. "Not logged anywhere"
is broader: a structured logger that serializes a dict containing the secret, an
exception that includes request context, or a debugger dump can all leak it. The
next path to check is the chapter-4 JSON log formatter — verify it never receives
the raw key.
```

## Reflection notes

1. **Why `_env_file=None`.** It stops Pydantic from reading the real `.env`, so
   the test's environment is exactly what `monkeypatch` set. Without it, CI (which
   has no `.env`) and your laptop (which does) produce different results — the
   classic "passes locally, fails in CI" trap, inverted.
2. **`extra="forbid"` blocking a future optional setting.** It is a feature: a new
   setting fails loudly until you declare it, which keeps `.env.example` and the
   code in lockstep. Flip to `allow` only when an external library you do not
   control injects its own env vars into the same namespace and renaming is
   impractical.
3. **Adding a second secret (an OpenAI fallback key).** `Settings` gains
   `openai_api_key: SecretStr | None = Field(default=None, alias="OPENAI_API_KEY")`
   (optional, since it is a fallback); `.env.example` gains a commented row;
   `agent.py` takes a second `api_key` argument and a provider switch; the tests
   gain a redaction assertion for the new field.
4. **Hot-reload vs. frozen settings.** Draw the line at the `Settings` boundary:
   values read once at import (`MODEL`, used to construct the client) are frozen;
   values read per-request through `Depends(get_settings)` *could* be made
   hot-reloadable by dropping the `@lru_cache` and re-reading, at the cost of
   re-validating on every request.
