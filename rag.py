from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

DB_PATH = Path(__file__).parent / "database" / "chroma"
COLLECTION_NAME = "andrew_ng"


def retrieve_context(query, k=5):
    client = chromadb.PersistentClient(path=str(DB_PATH))

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    try:
        collection = client.get_collection(
            COLLECTION_NAME,
            embedding_function=embedding_fn
        )

        data = collection.get(
            include=["documents", "metadatas"]
        )

    except Exception:
        return "", [], []

    docs = data.get("documents", [])
    metas = data.get("metadatas", [])
    ids = data.get("ids", [])

    if not docs:
        return "", [], []

    bm25 = BM25Okapi(
        [doc.lower().split() for doc in docs]
    )

    bm25_scores = bm25.get_scores(
        query.lower().split()
    )

    bm25_results = sorted(
        zip(ids, bm25_scores),
        key=lambda x: x[1],
        reverse=True
    )[:50]

    try:
        vector_results = collection.query(
            query_texts=[query],
            n_results=min(50, len(docs))
        )

        vector_ids = vector_results["ids"][0]

    except Exception:
        vector_ids = []

    scores = {}

    for rank, doc_id in enumerate(vector_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (61 + rank)

    for rank, (doc_id, _) in enumerate(bm25_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (61 + rank)

    top_ids = sorted(
        scores,
        key=scores.get,
        reverse=True
    )[:k]

    id_to_index = {
        doc_id: i
        for i, doc_id in enumerate(ids)
    }

    context = []
    sources = []
    chunks = []

    for n, doc_id in enumerate(top_ids, start=1):
        idx = id_to_index[doc_id]

        doc = docs[idx]
        meta = metas[idx]

        context.append(f"[{n}] {doc}")

        source = {
            "index": n,
            "source": meta.get("source", ""),
            "title": meta.get("title", ""),
            "url": meta.get("url", "")
        }

        chunk = {
            "text": doc,
            **source
        }

        sources.append(source)
        chunks.append(chunk)

    return "\n\n".join(context), sources, chunks