# mod-104-retrieval-basics: Retrieval Basics (Embeddings & Simple RAG) — Solutions

Reference solutions for the three `mod-104` exercises. Each walks through the
approach, gives annotated runnable Python (embeddings + simple RAG), maps the
code to the exercise acceptance criteria, lists common pitfalls, and shows how
to verify the result.

The three solutions build on each other: exercise-01 establishes the embedding
and cosine-similarity primitive, exercise-02 ships an end-to-end SQLite + NumPy
RAG loop on top of it, and exercise-03 wraps that loop in a labeled eval harness
and sweeps chunkers and K. Pinned constants are shared across all three
(`EMBED_MODEL = "text-embedding-3-small"`, `EMBED_DIM = 1536`,
`ANSWER_MODEL = "claude-sonnet-4-6"`).

## Exercises

- [exercise-01 — Embeddings and similarity](exercise-01-embeddings-similarity/README.md):
  embed a tiny corpus, rank by cosine similarity, and prove that normalized dot
  product equals explicit cosine.
- [exercise-02 — A simple RAG pipeline](exercise-02-simple-rag-pipeline/README.md):
  ingest a markdown corpus to SQLite + NumPy, run the full embed → search →
  prompt → answer loop, and return validated, cited answers with a clean
  refusal path.
- [exercise-03 — Chunking, indexing, and evaluation](exercise-03-chunking-and-indexing/README.md):
  add content-addressed ids, three swappable chunkers, a recall@K / MRR /
  groundedness harness, and a 3×3 sweep that locks a defensible baseline.

## Running the code

All reference scripts run offline with deterministic fallback backends (no API
keys required) or against real providers when `OPENAI_API_KEY` and
`ANTHROPIC_API_KEY` are set. Each exercise's `## Verification` section lists the
exact commands and an automated smoke check.
