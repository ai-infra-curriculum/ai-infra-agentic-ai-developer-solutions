# mod-104-retrieval-basics/exercise-01 — Solution

## Approach

The exercise asks for the smallest honest "semantic search": embed ~10 short
documents, embed a query, rank by cosine similarity, and print the top-K. The
whole point is to *see the primitive* before any index, chunker, or answer model
gets in the way.

Three decisions drive the implementation:

1. **Pin the model and dimension as constants.** `EMBED_MODEL` and `EMBED_DIM`
   live at module scope and are printed at startup. Exercises 2 and 3 reuse the
   same model name; an index embedded with one model and queried with another
   returns garbage scores, so making the name a single source of truth matters
   even at this scale.
2. **Normalize once, then use dot product.** After L2-normalizing every row to
   unit length, `cosine(a, b) == dot(a, b)`. That lets `search` reduce to one
   matrix-vector multiply (`doc_vecs @ q`) instead of recomputing norms per
   query. The required "probe the metric" step proves the equivalence by also
   computing cosine the long way on *un-normalized* vectors and confirming the
   ranking is identical.
3. **Stay brute-force.** At N = 10 a full scan is instant and removes recall as
   a confound. No FAISS, no answer model — those arrive in later exercises.

The script is fully runnable. If `OPENAI_API_KEY` is set it calls the real
embedding API; otherwise it falls back to a deterministic local hashing
"embedder" so a grader can run it offline and still observe correct ranking
behavior on the seed corpus. The fallback is clearly labeled and never silently
masks a misconfigured key — it only activates when no key is present.

## Reference implementation

Save as `embed_similarity.py`. Run with `python embed_similarity.py`.

```python
"""exercise-01: embeddings + cosine similarity over a tiny corpus.

Embeds a small document set, embeds queries, ranks by cosine similarity, and
prints the top-3 hits. Demonstrates that L2-normalized dot product equals
explicit cosine. No vector index, no chunker, no answer model.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np

# --- Pinned configuration (reused by exercise-02 and exercise-03) -----------
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
TOP_K = 3
_EPS = 1e-12  # guards against divide-by-zero on a zero vector

DOCS: list[str] = [
    "We accept returns of unused items within 30 days of delivery.",
    "All electronics must be returned in their original packaging.",
    "Sale items marked 'final' cannot be returned or exchanged.",
    "Standard shipping to the contiguous US takes 2-4 business days.",
    "Orders to Hawaii and Alaska usually arrive in 5-7 business days.",
    "International orders are sent via UPS Worldwide; tracking is included.",
    "Gift wrap is available at checkout for an additional $4.99 per item.",
    "Our customer service team is available Monday-Friday, 9am-6pm Pacific.",
    "We do not currently offer phone support; please use the contact form.",
    "Refunds appear on the original payment method within 5-10 business days.",
]

QUERIES: list[str] = [
    "how long do I have to return something?",  # direct hit
    "do you ship to non-mainland states?",      # paraphrase
    "can I call you?",                          # out-of-vocab
    "what is your favorite color?",             # mostly-unrelated
    "refund timeline",                          # specific phrase
]


# --- Embedding backends -----------------------------------------------------
def _embed_openai(texts: list[str]) -> np.ndarray:
    """Embed via the OpenAI API in a single batched call."""
    from openai import OpenAI  # imported lazily so offline runs do not need it

    client = OpenAI()
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)


def _embed_local(texts: list[str]) -> np.ndarray:
    """Deterministic offline fallback. NOT a real embedding model.

    Hashes character trigrams into a fixed-width vector so that texts sharing
    vocabulary land near each other. Good enough to demonstrate ranking
    mechanics without an API key; never use it in production.
    """
    vecs = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)
    for row, text in enumerate(texts):
        tokens = text.lower().split()
        for token in tokens:
            for i in range(len(token) - 2):
                trigram = token[i : i + 3]
                digest = hashlib.sha1(trigram.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:4], "big") % EMBED_DIM
                vecs[row, bucket] += 1.0
    return vecs


def _raw_embed(texts: list[str]) -> np.ndarray:
    """Pick a backend. Real API when a key is present, else offline fallback."""
    if os.environ.get("OPENAI_API_KEY"):
        return _embed_openai(texts)
    print("[warn] OPENAI_API_KEY not set; using offline hashing embedder.")
    return _embed_local(texts)


def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    """Scale each row to unit length so dot product equals cosine similarity."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.clip(norms, _EPS, None)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed and L2-normalize. Returns an (N, EMBED_DIM) float32 array."""
    return _l2_normalize(_raw_embed(texts))


# --- Search -----------------------------------------------------------------
def search(
    query: str,
    doc_vecs: np.ndarray,
    docs: list[str],
    k: int = TOP_K,
) -> list[tuple[float, str]]:
    """Return the top-k (score, document) pairs, highest score first."""
    q = embed_texts([query])[0]
    scores = doc_vecs @ q  # normalized rows -> this is cosine similarity
    top = np.argsort(-scores)[:k]
    return [(float(scores[i]), docs[i]) for i in top]


def _explicit_cosine(
    query: str,
    raw_doc_vecs: np.ndarray,
    docs: list[str],
    k: int = TOP_K,
) -> list[tuple[float, str]]:
    """Cosine the long way on UN-normalized vectors, to verify equivalence."""
    q = _raw_embed([query])[0]
    q_norm = np.linalg.norm(q)
    scores = np.array(
        [
            float(np.dot(raw_doc_vecs[i], q) / (np.linalg.norm(raw_doc_vecs[i]) * q_norm + _EPS))
            for i in range(len(docs))
        ]
    )
    top = np.argsort(-scores)[:k]
    return [(float(scores[i]), docs[i]) for i in top]


# --- Driver -----------------------------------------------------------------
def main() -> None:
    print(f"EMBED_MODEL = {EMBED_MODEL}  EMBED_DIM = {EMBED_DIM}")
    raw_doc_vecs = _raw_embed(DOCS)
    doc_vecs = _l2_normalize(raw_doc_vecs)
    print(f"Loaded {len(DOCS)} docs into shape {doc_vecs.shape}\n")

    for query in QUERIES:
        print(f"> {query}")
        for score, doc in search(query, doc_vecs, DOCS, k=TOP_K):
            print(f"  {score:.3f}  {doc}")
        print()

    # Required probe: normalized dot product must match explicit cosine ranking.
    print("=== metric check: normalized-dot vs explicit-cosine ===")
    for query in QUERIES[:3]:
        dot_hits = [doc for _, doc in search(query, doc_vecs, DOCS, k=TOP_K)]
        cos_hits = [doc for _, doc in _explicit_cosine(query, raw_doc_vecs, DOCS, k=TOP_K)]
        status = "MATCH" if dot_hits == cos_hits else "DIFFER"
        print(f"  [{status}] {query}")


if __name__ == "__main__":
    main()
```

`NOTES.md` is a short write-up, not code. A reference version:

```markdown
# exercise-01 notes

- EMBED_MODEL: text-embedding-3-small, dimension 1536.

## Semantic vs keyword
The paraphrase query "do you ship to non-mainland states?" returns the
Hawaii/Alaska doc as the top hit even though it shares no content words with
the query ("non-mainland" never appears). The literal-"ship" international
doc ranks below it, with a visible score gap, because the model encodes the
*meaning* "states that are not the mainland," not the surface tokens.

## Wrong-but-plausible
For "can I call you?" the "we do not currently offer phone support" doc
ranks above the generic "customer service hours" doc. That is the better
answer: it tells the user the thing they actually need (no phone line) rather
than office hours for a channel that does not exist.

## Off-topic
"what is your favorite color?" tops out around a low score, well under the
top score for a real query like "refund timeline." That gap is exactly what a
production score floor would threshold on to refuse instead of guessing.
```

## Meeting the acceptance criteria

- **Top-3 with scores for five queries.** `main` runs all five `QUERIES` and
  prints `score doc` lines for each.
- **`EMBED_MODEL` and `dim` pinned and printed.** Both are module constants and
  the first line of output reports them; the doc-vector shape is printed too.
- **Normalized vectors; dot and explicit-cosine rankings match.** `embed_texts`
  L2-normalizes; the metric-check block compares the normalized-dot ranking
  against an explicit cosine computed on un-normalized vectors for three
  queries and prints `MATCH`.
- **`NOTES.md` content.** The reference `NOTES.md` records the model and
  dimension, a paragraph each on three of the step-4 questions, and the
  off-topic-vs-real top-score comparison.

## Common pitfalls

1. **Forgetting to normalize, then using dot product anyway.** Raw dot product
   rewards long vectors, so a verbose doc can outrank a relevant one. Either
   normalize and use dot, or do not normalize and divide by both norms — never
   mix the two. The metric-check step exists to catch exactly this.
2. **Re-embedding the corpus inside `search`.** Embed the documents once, keep
   the matrix, and only embed the query per call. Embedding all docs on every
   query is slow and, against a paid API, needlessly expensive.
3. **Using a different model (or `dimensions` setting) for query vs docs.**
   Query and document vectors must come from the same model and dimension or the
   scores are meaningless. Pinning `EMBED_MODEL`/`EMBED_DIM` once prevents drift.
4. **`np.argsort(scores)` instead of `np.argsort(-scores)`.** `argsort` is
   ascending; without the negation you return the *least* similar documents.
5. **Tuning the corpus to make hits look good.** The assignment is to observe
   and explain surprises, not to engineer a clean demo. A "hard negative" that
   shares words but not meaning is a feature, not a bug to delete.

## Verification

```bash
# From the exercise directory, with numpy installed.
# Offline (deterministic hashing embedder, no key needed):
python embed_similarity.py

# Real embeddings:
export OPENAI_API_KEY=sk-...
python embed_similarity.py
```

Expected behavior:

- First line prints `EMBED_MODEL = text-embedding-3-small  EMBED_DIM = 1536`.
- Each of the five queries prints three `score  doc` lines, descending.
- The metric-check block prints `[MATCH]` for all three probed queries.
- With real embeddings, the paraphrase query surfaces the Hawaii/Alaska doc and
  the off-topic query's top score is visibly lower than a real query's.

A minimal automated check:

```bash
python - <<'PY'
import embed_similarity as m
import numpy as np
vecs = m.embed_texts(m.DOCS)
assert vecs.shape == (len(m.DOCS), m.EMBED_DIM)
# rows are unit length
assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-4)
hits = m.search("refund timeline", vecs, m.DOCS, k=3)
assert len(hits) == 3 and hits[0][0] >= hits[-1][0]
print("ok")
PY
```
