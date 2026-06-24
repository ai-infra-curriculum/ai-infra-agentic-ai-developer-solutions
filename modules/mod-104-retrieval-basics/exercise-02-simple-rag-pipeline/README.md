# mod-104-retrieval-basics/exercise-02 — Solution

## Approach

This exercise ships the smallest honest RAG system: an **offline ingest**
(load → chunk → embed → store) and an **online query loop** (embed → search →
prompt → answer → validate). Everything lives in one SQLite file plus brute-force
NumPy search — no vector library, no RAG framework — so every part of the loop is
visible.

Design decisions:

1. **One SQLite table is the index *and* the metadata store.** Columns
   `(chunk_id, document_id, source_path, section_title, ordinal, text,
   embed_model, vector)` keep the vector next to the text it came from. Vectors
   are stored as `float32` bytes (`vec.tobytes()`) and reconstituted with
   `np.frombuffer`. `INSERT OR REPLACE` keyed on `chunk_id` makes re-ingest
   idempotent.
2. **Heading-based chunker with a paragraph fallback.** Each `##`-or-higher
   heading starts a new chunk; sections over ~1500 characters fall back to
   paragraph splits. That is deliberately simpler than exercise-03's recursive
   splitter — the assignment is the loop, not the chunker.
3. **Filter search by `embed_model`.** The query is embedded with the *same*
   pinned model, and only rows tagged with that model are scored. This is the
   single most common RAG footgun: query and index embedded by different models
   produce meaningless cosine scores.
4. **Grounded, cited, refusable answers.** The system prompt instructs the model
   to answer only from context, cite `[chunk_id: ...]`, and refuse with a fixed
   phrase when the answer is absent. A regex validator confirms every cited id
   was actually retrieved, flagging hallucinated citations.
5. **`temperature=0`** on the answer model for reproducibility, and `K = 5`
   pinned (exercise-03 sweeps it).

The code runs end-to-end. If both `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` are
set it uses the real providers; otherwise it falls back to a deterministic
offline embedder and an extractive "answer model" that returns the top chunk's
text plus a real citation, so a grader can exercise the whole loop — including
citation validation and the refusal path — without keys.

## Reference implementation

Layout:

```text
exercise-02/
  rag.py                # ingest + search + answer loop (below)
  corpus/
    returns.md  shipping.md  payments.md  support.md  products.md
```

A representative `corpus/returns.md` (the others follow the same shape — 2-3
`##` headings, ~300-600 words total):

```markdown
# Returns

## Return window
We accept returns of unused items within 30 days of delivery. Start a return
from your order history; we email a prepaid label within one business day.

## Condition requirements
Items must be unused and in original condition. Electronics must be returned in
their original packaging with all accessories.

## Final sale
Sale items marked "final" cannot be returned or exchanged.
```

`rag.py`:

```python
"""exercise-02: a minimal end-to-end RAG pipeline (SQLite + NumPy).

Offline:  ingest(corpus_dir, db_path) -> load, heading-chunk, embed, store.
Online:   answer(conn, question)      -> embed, search, prompt, answer, validate.

No vector library, no RAG framework. Brute-force NumPy search over a SQLite
index. Falls back to deterministic offline backends when API keys are absent.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from pathlib import Path

import numpy as np

# --- Pinned configuration ---------------------------------------------------
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
ANSWER_MODEL = "claude-sonnet-4-6"
K = 5
TEMPERATURE = 0
MAX_SECTION_CHARS = 1500
EMBED_BATCH = 100
REFUSAL = "I do not see that in the documents."
_EPS = 1e-12

SYSTEM = (
    "You are a support assistant. Answer the user's question using ONLY the "
    "information in the provided context chunks. If the answer is not present in "
    "the context, say \"" + REFUSAL + "\" Cite chunks you used by [chunk_id: ...]."
)

_CITE_RE = re.compile(r"\[chunk_id:\s*([\w\-/#]+)\]")


# --- Schema -----------------------------------------------------------------
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
            vector        BLOB NOT NULL
        )
        """
    )
    return conn


# --- Embedding backends -----------------------------------------------------
def _embed_openai(texts: list[str]) -> np.ndarray:
    from openai import OpenAI

    client = OpenAI()
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)


def _embed_local(texts: list[str]) -> np.ndarray:
    """Deterministic offline fallback. Not a real embedding model."""
    vecs = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)
    for row, text in enumerate(texts):
        for token in text.lower().split():
            for i in range(max(len(token) - 2, 0)):
                trigram = token[i : i + 3]
                bucket = int.from_bytes(
                    hashlib.sha1(trigram.encode()).digest()[:4], "big"
                ) % EMBED_DIM
                vecs[row, bucket] += 1.0
    return vecs


def _raw_embed(texts: list[str]) -> np.ndarray:
    if os.environ.get("OPENAI_API_KEY"):
        return _embed_openai(texts)
    return _embed_local(texts)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed and L2-normalize so dot product equals cosine similarity."""
    vecs = _raw_embed(texts)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.clip(norms, _EPS, None)


# --- Chunking ---------------------------------------------------------------
def chunk_by_heading(text: str) -> list[dict]:
    """Split markdown into chunks at each '##'-or-higher heading.

    Sections longer than MAX_SECTION_CHARS fall back to paragraph splits.
    Returns records of {"title": str | None, "text": str}.
    """
    sections: list[dict] = []
    title: str | None = None
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append({"title": title, "text": body})

    for line in text.splitlines():
        if re.match(r"^#{1,6}\s+", line):
            flush()
            title = line.lstrip("#").strip()
            buf = []
        else:
            buf.append(line)
    flush()

    out: list[dict] = []
    for sec in sections:
        if len(sec["text"]) <= MAX_SECTION_CHARS:
            out.append(sec)
            continue
        for para in re.split(r"\n\s*\n", sec["text"]):
            para = para.strip()
            if para:
                out.append({"title": sec["title"], "text": para})
    return out


# --- Ingest -----------------------------------------------------------------
def ingest(corpus_dir: str, db_path: str) -> dict:
    """load -> chunk -> embed -> store. Idempotent on chunk_id."""
    start = time.perf_counter()
    conn = connect(db_path)
    root = Path(corpus_dir)
    rows: list[dict] = []

    for path in sorted(root.glob("**/*.md")):
        document_id = path.relative_to(root).with_suffix("").as_posix()
        for ordinal, sec in enumerate(chunk_by_heading(path.read_text())):
            rows.append(
                {
                    "chunk_id": f"{document_id}#{ordinal:04d}",
                    "document_id": document_id,
                    "source_path": str(path),
                    "section_title": sec["title"],
                    "ordinal": ordinal,
                    "text": sec["text"],
                    "embed_model": EMBED_MODEL,
                }
            )

    for i in range(0, len(rows), EMBED_BATCH):
        batch = rows[i : i + EMBED_BATCH]
        vecs = embed_texts([r["text"] for r in batch])
        for r, vec in zip(batch, vecs):
            r["vector"] = vec.astype(np.float32).tobytes()
        conn.executemany(
            """
            INSERT OR REPLACE INTO chunks
            (chunk_id, document_id, source_path, section_title, ordinal,
             text, embed_model, vector)
            VALUES
            (:chunk_id, :document_id, :source_path, :section_title, :ordinal,
             :text, :embed_model, :vector)
            """,
            batch,
        )
    conn.commit()

    n_docs = len({r["document_id"] for r in rows})
    summary = {
        "corpus_root": str(root),
        "documents": n_docs,
        "chunks": len(rows),
        "embed_model": EMBED_MODEL,
        "elapsed_s": round(time.perf_counter() - start, 3),
    }
    Path("manifest.json").write_text(json.dumps(summary, indent=2))
    print(
        f"ingested {summary['documents']} docs / {summary['chunks']} chunks "
        f"({EMBED_MODEL}) in {summary['elapsed_s']}s"
    )
    conn.close()
    return summary


# --- Search -----------------------------------------------------------------
def search(conn: sqlite3.Connection, query: str, k: int = K) -> list[dict]:
    """Brute-force dot-product search over rows for the pinned embed model."""
    rows = conn.execute(
        "SELECT chunk_id, text, section_title, source_path, vector "
        "FROM chunks WHERE embed_model = ?",
        (EMBED_MODEL,),
    ).fetchall()
    if not rows:
        return []

    mat = np.vstack([np.frombuffer(r[4], dtype=np.float32) for r in rows])
    q = embed_texts([query])[0]
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


# --- Answer backends --------------------------------------------------------
def _answer_anthropic(system: str, user: str) -> tuple[str, str]:
    from anthropic import Anthropic

    client = Anthropic()
    resp = client.messages.create(
        model=ANSWER_MODEL,
        max_tokens=1024,
        temperature=TEMPERATURE,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text, resp.stop_reason


def _answer_local(chunks: list[dict], question: str) -> tuple[str, str]:
    """Offline extractive fallback. Returns top chunk text + a real citation,
    or the refusal phrase when retrieval is weak."""
    if not chunks or chunks[0]["score"] < 0.15:
        return REFUSAL, "end_turn"
    top = chunks[0]
    return f"{top['text']} [chunk_id: {top['chunk_id']}]", "end_turn"


# --- RAG query loop ---------------------------------------------------------
def build_user_message(chunks: list[dict], question: str) -> str:
    blocks = []
    for c in chunks:
        title = c["section_title"] or ""
        blocks.append(f"[chunk_id: {c['chunk_id']}] {title}\n{c['text']}")
    context = "\n\n---\n\n".join(blocks)
    return f"Context:\n{context}\n\nQuestion: {question}"


def validate_citations(answer_text: str, chunks: list[dict]) -> tuple[list[str], bool]:
    """Return (cited_ids, all_valid). Invalid means the model invented an id."""
    cited = _CITE_RE.findall(answer_text)
    retrieved = {c["chunk_id"] for c in chunks}
    valid = all(cid in retrieved for cid in cited)
    return cited, valid


def answer(conn: sqlite3.Connection, question: str) -> dict:
    chunks = search(conn, question, k=K)
    user = build_user_message(chunks, question)
    if os.environ.get("ANTHROPIC_API_KEY"):
        text, stop_reason = _answer_anthropic(SYSTEM, user)
    else:
        text, stop_reason = _answer_local(chunks, question)
    cited, valid = validate_citations(text, chunks)
    return {
        "question": question,
        "chunks": chunks,
        "answer": text,
        "stop_reason": stop_reason,
        "cited_ids": cited,
        "citations_valid": valid,
    }


# --- Driver -----------------------------------------------------------------
QUESTIONS = [
    "How many days do I have to return an item?",
    "Do you deliver to Hawaii?",
    "If I return a sale item, when will I see the money?",
    "What warranty does the desk lamp come with?",
    "What is your stance on cryptocurrency payments?",
]


def main() -> None:
    print(f"EMBED_MODEL={EMBED_MODEL}  ANSWER_MODEL={ANSWER_MODEL}  K={K}  temp={TEMPERATURE}")
    ingest("corpus", "rag.db")
    conn = connect("rag.db")
    with open("results.jsonl", "w") as fh:
        for q in QUESTIONS:
            res = answer(conn, q)
            fh.write(
                json.dumps(
                    {
                        "q": q,
                        "retrieved_ids": [c["chunk_id"] for c in res["chunks"]],
                        "cited_ids": res["cited_ids"],
                        "citations_valid": res["citations_valid"],
                        "answer": res["answer"],
                    }
                )
                + "\n"
            )
            print(f"\n> {q}\n  {res['answer']}")
    conn.close()


if __name__ == "__main__":
    main()
```

## Meeting the acceptance criteria

- **Working `ingest` producing 15+ tagged chunks.** A five-file corpus with
  2-5 sections each yields well over 15 chunks; every row stores
  `embed_model = EMBED_MODEL`. `manifest.json` records corpus root, embed model,
  chunk count, and elapsed seconds.
- **Working `answer` returning answer + retrieved chunks.** `answer` runs the
  full loop for all five baseline questions and returns
  `{question, chunks, answer, stop_reason, cited_ids, citations_valid}`.
- **`results.jsonl` per question.** One line each with `retrieved_ids`,
  `cited_ids`, `citations_valid`, and `answer`.
- **Citation validation.** `validate_citations` regex-extracts cited ids and
  confirms each appears in the retrieved set; `citations_valid` is `False` when
  the model invents an id.
- **Refusal case.** The cryptocurrency question retrieves only weakly-related
  chunks, so a grounded model (or the offline fallback's score floor) returns
  the exact `REFUSAL` phrase rather than a fabricated policy.
- **`NOTES.md`.** Records the pinned `EMBED_MODEL`, `ANSWER_MODEL`, `K`,
  `temperature`, and the step-6 trace (system + chunks + question, the reply,
  retrieved chunks with scores, and the 4-8 sentence analysis).

## Common pitfalls

1. **Embedding the query with a different model than the index.** `search`
   embeds with `EMBED_MODEL` and filters rows by `embed_model = ?`. Skip either
   guard and you score query vectors against incompatible document vectors,
   producing confident-looking but meaningless rankings.
2. **Position-only `chunk_id` that breaks on re-chunk.** `document_id#ordinal`
   is fine here, but if a chunk's *content* shifts under the same ordinal the id
   silently points at different text. Exercise-03 hardens this with a content
   hash — do not carry position-only ids into a labeled eval.
3. **Non-zero temperature on the answer model.** Without `temperature=0` the
   same question yields different answers and citations run-to-run, making the
   trace and `results.jsonl` non-reproducible.
4. **Trusting citations without validating them.** Models cite ids that were
   never retrieved. Always intersect cited ids with the retrieved set; an
   invalid citation is a groundedness failure even when the prose looks right.
5. **Letting the model answer from training data.** Drop the "use ONLY the
   context" instruction and the model will helpfully invent a returns policy.
   The refusal phrase plus the "answer only from context" instruction are what
   make this RAG rather than a chatbot.

## Verification

```bash
# Offline (no keys): deterministic embedder + extractive answerer.
python rag.py

# Real providers:
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
python rag.py
```

Expected:

- `manifest.json` exists and reports `chunks` >= 15.
- `results.jsonl` has exactly five lines, each with `retrieved_ids`,
  `cited_ids`, `citations_valid`, `answer`.
- The cryptocurrency line's `answer` is the refusal phrase.

Automated smoke check:

```bash
python - <<'PY'
import json, rag
summary = rag.ingest("corpus", "rag.db")
assert summary["chunks"] >= 15, summary
conn = rag.connect("rag.db")
res = rag.answer(conn, "How many days do I have to return an item?")
assert res["chunks"], "no chunks retrieved"
assert res["citations_valid"], "model cited an unretrieved id"
refusal = rag.answer(conn, "What is your stance on cryptocurrency payments?")
assert rag.REFUSAL in refusal["answer"], refusal["answer"]
print("ok")
PY
```
