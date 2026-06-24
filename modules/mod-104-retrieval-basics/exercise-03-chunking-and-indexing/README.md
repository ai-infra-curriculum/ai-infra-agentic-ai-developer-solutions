# mod-104-retrieval-basics/exercise-03 — Solution

## Approach

This exercise turns "I think retrieval improved" into "recall@5 went from 0.61
to 0.78." The loop shape from exercise-02 does not change; the **chunker** and
**K** do, and a labeled eval set scores each configuration so the winner is
defensible with a number.

Five moving parts:

1. **Content-addressed chunk ids.** Switch from `document_id#ordinal` to the
   stable form `f"{document_id}#{ordinal:04d}#{sha1(text)[:10]}"`. The hash ties
   the id to *content*, so a label recorded against a chunk stays valid only when
   that chunk's text is unchanged — and visibly breaks when it changes. Without
   this, swapping chunkers silently invalidates every label.
2. **A swappable chunker strategy.** `ingest(corpus_dir, db_path, chunker,
   chunker_name)` takes the chunker as a parameter and tags every row with a
   `chunker` column. Three chunkers coexist in one database: `heading`,
   `recursive_800_100`, `tokens_400_40`. Search filters on both `embed_model`
   and `chunker`.
3. **A labeled eval set.** `eval/dataset.jsonl` holds 30+ rows spanning
   direct-hit, paraphrase, multi-doc, and refusal classes, each with
   `answer_chunk_ids` recorded against a real ingested index.
4. **A metrics harness.** `evaluate(conn, dataset, chunker_name, k)` computes
   recall@K, MRR, answer-substring, citation validity, and over-answer /
   over-refuse rates, then aggregates them. The per-metric functions are copied
   from chapter 7.
5. **A sweep + a locked baseline.** `sweep.py` runs the 3×3 cross-product of
   chunkers and K, appends one JSON record per run to `eval_runs/sweep.jsonl`,
   and the chosen winner is frozen into `eval_runs/baseline.json` as a
   regression gate.

The reference code imports the exercise-02 `rag` module for embedding, search,
and the answer loop, and adds only the new pieces: the two extra chunkers, the
content-hash id, the metrics, and the sweep. It runs offline (deterministic
backends) or against real providers, exactly as exercise-02 does.

## Reference implementation

### Stable id and the chunkers (`chunkers.py`)

```python
"""exercise-03: content-addressed ids + three swappable chunkers."""

from __future__ import annotations

import hashlib
import re

from rag import chunk_by_heading as _heading_sections  # exercise-02 splitter


def stable_chunk_id(document_id: str, ordinal: int, text: str) -> str:
    """Tie the id to content so labels break loudly when text changes."""
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{document_id}#{ordinal:04d}#{h}"


def chunk_by_heading(text: str, doc_id: str) -> list[dict]:
    """Sections by markdown heading (exercise-02 behavior)."""
    return _heading_sections(text)


def chunk_recursive(
    text: str, doc_id: str, max_chars: int = 800, overlap: int = 100
) -> list[dict]:
    """Recursive character splitter (chapter 4).

    Split on the largest natural boundary that keeps pieces under max_chars,
    descending paragraph -> line -> sentence -> hard cut, with char overlap
    so a fact straddling a boundary survives in at least one chunk.
    """
    separators = ["\n\n", "\n", ". ", " "]

    def split(segment: str, seps: list[str]) -> list[str]:
        segment = segment.strip()
        if len(segment) <= max_chars or not seps:
            return [segment] if segment else []
        sep, rest = seps[0], seps[1:]
        parts = segment.split(sep) if sep in segment else [segment]
        out, buf = [], ""
        for part in parts:
            candidate = (buf + sep + part) if buf else part
            if len(candidate) <= max_chars:
                buf = candidate
            else:
                if buf:
                    out.extend(split(buf, rest))
                buf = part
        if buf:
            out.extend(split(buf, rest))
        return [p for p in out if p]

    pieces = split(text, separators)
    if overlap <= 0:
        return [{"title": None, "text": p} for p in pieces]

    overlapped: list[dict] = []
    for i, piece in enumerate(pieces):
        prefix = pieces[i - 1][-overlap:] + " " if i > 0 else ""
        overlapped.append({"title": None, "text": (prefix + piece).strip()})
    return overlapped


def chunk_fixed_tokens(
    text: str, doc_id: str, max_tokens: int = 400, overlap: int = 40
) -> list[dict]:
    """Token-aware fixed-size splitter using tiktoken when available.

    Falls back to whitespace 'tokens' offline so the function is always
    runnable; the chunk boundaries differ slightly but the strategy is the same.
    """
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        ids = enc.encode(text)
        decode = enc.decode
    except Exception:  # offline / tiktoken absent
        ids = text.split()
        decode = lambda toks: " ".join(toks)  # noqa: E731

    step = max(max_tokens - overlap, 1)
    out: list[dict] = []
    for start in range(0, len(ids), step):
        window = ids[start : start + max_tokens]
        if not window:
            continue
        body = decode(window).strip()
        if body:
            out.append({"title": None, "text": body})
        if start + max_tokens >= len(ids):
            break
    return out


CHUNKERS = {
    "heading": chunk_by_heading,
    "recursive_800_100": lambda text, doc_id: chunk_recursive(
        text, doc_id, max_chars=800, overlap=100
    ),
    "tokens_400_40": lambda text, doc_id: chunk_fixed_tokens(
        text, doc_id, max_tokens=400, overlap=40
    ),
}
```

### Switchable ingest + the eval harness (`eval.py`)

```python
"""exercise-03: switchable ingest, metrics, and the sweep harness."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from statistics import mean

import numpy as np

import rag  # exercise-02: embed_texts, search backend, answer loop, REFUSAL
from chunkers import CHUNKERS, stable_chunk_id

EMBED_MODEL = rag.EMBED_MODEL
REFUSAL_PHRASES = ["i do not see that in the documents", "i don't see that"]


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id      TEXT PRIMARY KEY,
            document_id   TEXT NOT NULL,
            source_path   TEXT NOT NULL,
            section_title TEXT,
            ordinal       INTEGER NOT NULL,
            text          TEXT NOT NULL,
            embed_model   TEXT NOT NULL,
            chunker       TEXT NOT NULL,
            vector        BLOB NOT NULL
        )
        """
    )
    return conn


def ingest(corpus_dir: str, db_path: str, chunker, chunker_name: str) -> int:
    """Idempotent for a given chunker_name; variants coexist in one DB."""
    conn = connect(db_path)
    root = Path(corpus_dir)
    rows: list[dict] = []
    for path in sorted(root.glob("**/*.md")):
        document_id = path.relative_to(root).with_suffix("").as_posix()
        for ordinal, sec in enumerate(chunker(path.read_text(), document_id)):
            cid = stable_chunk_id(document_id, ordinal, sec["text"])
            rows.append(
                {
                    "chunk_id": cid,
                    "document_id": document_id,
                    "source_path": str(path),
                    "section_title": sec.get("title"),
                    "ordinal": ordinal,
                    "text": sec["text"],
                    "embed_model": EMBED_MODEL,
                    "chunker": chunker_name,
                }
            )
    for i in range(0, len(rows), rag.EMBED_BATCH):
        batch = rows[i : i + rag.EMBED_BATCH]
        vecs = rag.embed_texts([r["text"] for r in batch])
        for r, vec in zip(batch, vecs):
            r["vector"] = vec.astype(np.float32).tobytes()
        conn.executemany(
            """
            INSERT OR REPLACE INTO chunks
            (chunk_id, document_id, source_path, section_title, ordinal,
             text, embed_model, chunker, vector)
            VALUES
            (:chunk_id, :document_id, :source_path, :section_title, :ordinal,
             :text, :embed_model, :chunker, :vector)
            """,
            batch,
        )
    conn.commit()
    conn.close()
    return len(rows)


def search(conn, query: str, chunker_name: str, k: int) -> list[dict]:
    rows = conn.execute(
        "SELECT chunk_id, text, section_title, source_path, vector "
        "FROM chunks WHERE embed_model = ? AND chunker = ?",
        (EMBED_MODEL, chunker_name),
    ).fetchall()
    if not rows:
        return []
    mat = np.vstack([np.frombuffer(r[4], dtype=np.float32) for r in rows])
    q = rag.embed_texts([query])[0]
    scores = mat @ q
    top = np.argsort(-scores)[:k]
    return [
        {
            "chunk_id": rows[i][0],
            "text": rows[i][1],
            "section_title": rows[i][2],
            "source_path": rows[i][3],
            "score": float(scores[i]),
        }
        for i in top
    ]


# --- Metrics (chapter 7) ----------------------------------------------------
def recall_at_k(retrieved_ids: list[str], correct_ids: set[str], k: int) -> float:
    if not correct_ids:
        return 1.0  # nothing to recall; treat as success
    hits = sum(1 for cid in retrieved_ids[:k] if cid in correct_ids)
    return hits / len(correct_ids)


def reciprocal_rank(retrieved_ids: list[str], correct_ids: set[str]) -> float:
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in correct_ids:
            return 1.0 / rank
    return 0.0


def answer_contains(answer: str, expected_snippet: str | None) -> bool:
    if not expected_snippet:
        return True
    return expected_snippet.lower() in answer.lower()


def refused(answer: str) -> bool:
    low = answer.lower()
    return any(p in low for p in REFUSAL_PHRASES)


def citations_valid(answer: str, retrieved_ids: set[str]) -> bool:
    _, valid = rag.validate_citations(answer, [{"chunk_id": c} for c in retrieved_ids])
    return valid


# --- Harness ----------------------------------------------------------------
def evaluate(conn, dataset: list[dict], *, chunker_name: str, k: int) -> dict:
    rows = []
    for row in dataset:
        chunks = search(conn, row["q"], chunker_name, k)
        retrieved_ids = [c["chunk_id"] for c in chunks]
        user = rag.build_user_message(chunks, row["q"])
        if rag.os.environ.get("ANTHROPIC_API_KEY"):
            ans, _ = rag._answer_anthropic(rag.SYSTEM, user)
        else:
            ans, _ = rag._answer_local(chunks, row["q"])
        correct = set(row.get("answer_chunk_ids", []))
        expect_refusal = bool(row.get("expected_refusal", False))
        did_refuse = refused(ans)
        rows.append(
            {
                "recall_at_k": recall_at_k(retrieved_ids, correct, k),
                "mrr": reciprocal_rank(retrieved_ids, correct),
                "answer_ok": answer_contains(ans, row.get("expected_answer_snippet")),
                "citations_ok": citations_valid(ans, set(retrieved_ids)),
                "expected_refusal": expect_refusal,
                "did_refuse": did_refuse,
            }
        )

    non_refusal = [r for r in rows if not r["expected_refusal"]]
    refusal = [r for r in rows if r["expected_refusal"]]
    summary = {
        "n": len(rows),
        "recall_at_k_mean": round(mean(r["recall_at_k"] for r in non_refusal), 3)
        if non_refusal
        else 0.0,
        "mrr_mean": round(mean(r["mrr"] for r in non_refusal), 3) if non_refusal else 0.0,
        "answer_ok_rate": round(mean(r["answer_ok"] for r in non_refusal), 3)
        if non_refusal
        else 0.0,
        "citations_ok_rate": round(mean(r["citations_ok"] for r in non_refusal), 3)
        if non_refusal
        else 0.0,
        "over_answer_rate": round(mean(not r["did_refuse"] for r in refusal), 3)
        if refusal
        else 0.0,
        "over_refuse_rate": round(mean(r["did_refuse"] for r in non_refusal), 3)
        if non_refusal
        else 0.0,
        "embed_model": EMBED_MODEL,
        "answer_model": rag.ANSWER_MODEL,
    }
    return {"rows": rows, "summary": summary}


def load_dataset(path: str = "eval/dataset.jsonl") -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
```

### The sweep (`sweep.py`)

```python
"""exercise-03: run the 3x3 chunker x K sweep and lock the winning baseline."""

import json
from pathlib import Path

from chunkers import CHUNKERS
import eval as ev

CORPUS, DB = "corpus", "eval.db"
KS = (3, 5, 10)


def main() -> None:
    dataset = ev.load_dataset()
    Path("eval_runs").mkdir(exist_ok=True)
    sweep_path = Path("eval_runs/sweep.jsonl")
    sweep_path.write_text("")  # fresh run

    for chunker_name, chunker in CHUNKERS.items():
        ev.ingest(CORPUS, DB, chunker, chunker_name)  # idempotent per chunker
        conn = ev.connect(DB)
        for k in KS:
            result = ev.evaluate(conn, dataset, chunker_name=chunker_name, k=k)
            record = {"chunker": chunker_name, "k": k, **result["summary"]}
            with sweep_path.open("a") as fh:
                fh.write(json.dumps(record) + "\n")
            print(
                f"{chunker_name:>18}  K={k:<2}  "
                f"recall@K={record['recall_at_k_mean']}  "
                f"mrr={record['mrr_mean']}  answer_ok={record['answer_ok_rate']}"
            )
        conn.close()


if __name__ == "__main__":
    main()
```

Locking the winner once the sweep is reviewed (e.g. `recursive_800_100`, K=5):

```bash
python - <<'PY'
import json
from pathlib import Path
rows = [json.loads(l) for l in Path("eval_runs/sweep.jsonl").read_text().splitlines()]
winner = next(r for r in rows if r["chunker"] == "recursive_800_100" and r["k"] == 5)
Path("eval_runs/baseline.json").write_text(json.dumps(winner, indent=2))
print("locked:", winner["chunker"], "K", winner["k"])
PY
```

### Eval set shape (`eval/dataset.jsonl`)

```jsonl
{"q": "How many days do I have to return an item?", "answer_doc": "returns", "answer_chunk_ids": ["returns#0000#a1b2c3d4e5"], "expected_answer_snippet": "30 days", "expected_refusal": false}
{"q": "Do you deliver to Hawaii?", "answer_doc": "shipping", "answer_chunk_ids": ["shipping#0001#0f9e8d7c6b"], "expected_answer_snippet": "5-7 business days", "expected_refusal": false}
{"q": "If I return a sale item, when will I see the money?", "answer_doc": "payments", "answer_chunk_ids": ["returns#0002#11aa22bb33", "payments#0001#44cc55dd66"], "expected_answer_snippet": "5-10 business days", "expected_refusal": false}
{"q": "What is your stance on cryptocurrency payments?", "answer_doc": "", "answer_chunk_ids": [], "expected_answer_snippet": "", "expected_refusal": true}
```

(The real file has 30+ rows: 5+ multi-doc, 5+ paraphrase, 5+ refusal, 5+
direct-hit. The `answer_chunk_ids` hashes come from listing chunks after a
baseline ingest — never hand-typed.)

## Meeting the acceptance criteria

- **`eval/dataset.jsonl` with 30+ labeled rows** across direct-hit, paraphrase,
  multi-doc (5+), and refusal (5+) classes, with `answer_chunk_ids` taken from a
  real ingested index.
- **Three chunkers + switchable ingest.** `CHUNKERS` registry plus
  `ingest(corpus_dir, db_path, chunker, chunker_name)`, each row tagged with a
  `chunker` column.
- **9-configuration sweep.** `sweep.py` runs `{heading, recursive_800_100,
  tokens_400_40} × {3, 5, 10}` and appends one record per run to
  `eval_runs/sweep.jsonl`.
- **Summary table + defensible pick** rendered in `NOTES.md` (template below).
- **Locked baseline.** `eval_runs/baseline.json` holds the winning config with
  all summary metrics, re-runnable via `python sweep.py`.
- **Stable ids.** `stable_chunk_id` content-hashes the text so labels survive
  re-ingest of the same chunker and break visibly when content changes.

A `NOTES.md` table template:

```markdown
| chunker | K | recall@K | MRR | answer_ok | citations_ok | over_refuse | over_answer |
|---|---|---|---|---|---|---|---|
| heading | 3 | 0.68 | 0.55 | 0.74 | 0.90 | 0.06 | 0.00 |
| heading | 5 | 0.74 | 0.58 | 0.81 | 0.91 | 0.06 | 0.00 |
| recursive_800_100 | 5 | 0.81 | 0.63 | 0.85 | 0.92 | 0.04 | 0.00 |
| tokens_400_40 | 5 | 0.79 | 0.60 | 0.83 | 0.91 | 0.05 | 0.00 |
```

## Common pitfalls

1. **Labeling against position-only ids.** If `answer_chunk_ids` use
   `doc#ordinal`, the first chunker swap shifts boundaries and every label now
   points at the wrong (or missing) chunk, so recall collapses for reasons that
   have nothing to do with the chunker. Use `stable_chunk_id` *before* labeling.
2. **Editing the eval set after seeing metrics.** Adjusting labels to match the
   system's output is circular and inflates every score. Freeze the dataset once
   labeling starts; only grow it with cases the system *struggles* on.
3. **Changing the embed or answer model mid-sweep.** A model swap is a confound:
   you can no longer attribute a recall change to the chunker. Pin both for the
   entire 9-run sweep.
4. **Reporting recall@K over refusal rows.** Refusal questions have no correct
   chunks; averaging recall over them is meaningless. Compute recall/MRR/answer
   rates over non-refusal rows and over-answer over refusal rows, as the harness
   does.
5. **Shipping a 0.02 recall bump that tanks `over_answer_rate`.** A higher K
   raises recall but can drag in irrelevant context that tempts the model to
   answer a refusal question. Defend the pick on the metric your failures
   actually live in, and name the tradeoff explicitly.

## Verification

```bash
# Offline sweep (deterministic backends, no keys):
python sweep.py

# Real providers:
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
python sweep.py
```

Expected:

- `eval_runs/sweep.jsonl` has exactly 9 lines (3 chunkers × 3 K values).
- Each line carries `recall_at_k_mean`, `mrr_mean`, `answer_ok_rate`,
  `citations_ok_rate`, `over_answer_rate`, `over_refuse_rate`, plus the pinned
  `embed_model` and `answer_model`.
- After locking, `eval_runs/baseline.json` exists with the winning config.

Automated check:

```bash
python - <<'PY'
import json
from pathlib import Path
import eval as ev
from chunkers import CHUNKERS, stable_chunk_id

# stable ids are content-addressed
a = stable_chunk_id("returns", 0, "30 day window")
b = stable_chunk_id("returns", 0, "30 day window")
c = stable_chunk_id("returns", 0, "60 day window")
assert a == b and a != c, "ids must depend on content"

# all three chunkers produce non-empty records
for name, fn in CHUNKERS.items():
    recs = fn("# H\n\npara one.\n\npara two with more words to split.", "doc")
    assert recs and all("text" in r for r in recs), name

# metric sanity
assert ev.recall_at_k(["x"], {"x"}, 5) == 1.0
assert ev.reciprocal_rank(["a", "x"], {"x"}) == 0.5
print("ok")
PY
```

A regression-gate target (`make test-retrieval`) compares a fresh
`evaluate` against `eval_runs/baseline.json` and fails if recall@K or
answer_ok_rate drops by more than 0.03.
