
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from cloud_ingest import ensure_collection, get_qdrant_client


# -------------------- CONFIGURATION --------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "prepared_documents.jsonl"
QDRANT_PATH = BASE_DIR / "qdrant_storage"
COLLECTION_NAME = "indian_legal_sections_v2"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DENSE_TOP_K = 50
BM25_TOP_K = 50
RRF_K = 60
DEFAULT_TOP_K = 5

ACT_FILES = {
    "BNS": BASE_DIR / "BNS_sections.json",
    "BNSS": BASE_DIR / "BNSS_sections.json",
    "BSA": BASE_DIR / "BSA_sections.json",
}


# -------------------- HELPERS --------------------

def normalize_section(value):
    if value is None:
        return ""
    return str(value).strip()


def tokenize(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())


def document_text(doc):
    """Use the same text representation as the evaluation script."""
    return doc.get("embedding_text") or " ".join([
        str(doc.get("act_name", "")),
        str(doc.get("act_code", "")),
        str(doc.get("section_number", "")),
        str(doc.get("section_title", "")),
        str(doc.get("text", "")),
    ])


def section_key(doc):
    return (
        str(doc.get("act_code", "")).upper(),
        normalize_section(doc.get("section_number")),
    )


def load_json_sections(path, act_code):
    """Load original full sections, tolerating common JSON layouts."""
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        sections = data
    else:
        sections = data.get("sections", [])

    result = {}

    for section in sections:
        number = normalize_section(section.get("section_number"))
        if not number:
            continue

        section = dict(section)
        section.setdefault("act_code", act_code)
        section.setdefault("act_name", act_code)

        result[(act_code, number)] = section

    return result


# -------------------- INITIALIZE RESOURCES --------------------

_documents = None
_doc_by_id = None
_doc_by_section_and_chunk = None
_bm25 = None
_model = None
_client = None
_original_sections = None


def _initialize():
    global _documents, _doc_by_id
    global _doc_by_section_and_chunk, _bm25
    global _model, _client, _original_sections

    if _documents is not None:
        return

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing prepared documents: {DATA_FILE}")

    with DATA_FILE.open("r", encoding="utf-8") as f:
        documents = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    if not documents:
        raise ValueError(f"No documents found in {DATA_FILE}")

    doc_by_id = {
        str(doc["id"]): doc
        for doc in documents
        if doc.get("id") is not None
    }

    doc_by_section_and_chunk = {}

    for doc in documents:
        key = (
            *section_key(doc),
            str(doc.get("chunk_index", "")),
        )
        doc_by_section_and_chunk[key] = doc

    # Build BM25 using exactly the evaluation text/tokenization.
    tokenized_corpus = [
        tokenize(document_text(doc))
        for doc in documents
    ]
    bm25 = BM25Okapi(tokenized_corpus)

    client = get_qdrant_client()
    ensure_collection(client)

    available = {
        collection.name
        for collection in client.get_collections().collections
    }

    if COLLECTION_NAME not in available:
        raise ValueError(
            f"Qdrant collection '{COLLECTION_NAME}' was not initialized."
    )

    original_sections = {}
    for act_code, path in ACT_FILES.items():
        original_sections.update(
            load_json_sections(path, act_code)
        )

    _documents = documents
    _doc_by_id = doc_by_id
    _doc_by_section_and_chunk = doc_by_section_and_chunk
    _bm25 = bm25
    _model = model
    _client = client
    _original_sections = original_sections


# -------------------- HYBRID RETRIEVAL --------------------

def retrieve(question):
    """
    Return all ranked sections using the evaluation script's
    BM25 + dense retrieval + RRF algorithm.

    Each result contains:
      key: (act_code, section_number)
      title: section title
      score: RRF score
    """
    _initialize()

    if not question or not question.strip():
        return []

    # 1. BM25 candidates
    query_tokens = tokenize(question)
    bm25_scores = _bm25.get_scores(query_tokens)

    bm25_indices = np.argsort(bm25_scores)[::-1][:BM25_TOP_K]

    # 2. Dense candidates
    query_vector = _model.encode(
        question,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()

    dense_response = _client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=DENSE_TOP_K,
        with_payload=True,
    )

    # 3. Reciprocal Rank Fusion at chunk level
    chunk_scores = defaultdict(float)

    for rank, idx in enumerate(bm25_indices, start=1):
        if bm25_scores[idx] > 0:
            doc_id = str(_documents[int(idx)].get("id", idx))
            chunk_scores[doc_id] += 1.0 / (RRF_K + rank)

    for rank, point in enumerate(dense_response.points, start=1):
        payload = point.payload or {}

        chunk_id = payload.get("chunk_id", payload.get("id"))

        if chunk_id is not None:
            doc_id = str(chunk_id)

            if doc_id in _doc_by_id:
                chunk_scores[doc_id] += 1.0 / (RRF_K + rank)
                continue

        # Fallback for payloads with section/chunk metadata.
        key = (
            str(payload.get("act_code", "")).upper(),
            normalize_section(payload.get("section_number")),
            str(payload.get("chunk_index", "")),
        )

        matched_doc = _doc_by_section_and_chunk.get(key)

        if matched_doc:
            doc_id = str(matched_doc["id"])
            chunk_scores[doc_id] += 1.0 / (RRF_K + rank)

    # 4. Aggregate chunks into sections, keeping the best chunk score.
    section_scores = {}
    section_docs = {}

    for doc_id, score in chunk_scores.items():
        doc = _doc_by_id.get(doc_id)
        if not doc:
            continue

        key = section_key(doc)
        if not all(key):
            continue

        if key not in section_scores or score > section_scores[key]:
            section_scores[key] = score
            section_docs[key] = doc

    ranked_keys = sorted(
        section_scores,
        key=lambda key: section_scores[key],
        reverse=True,
    )

    results = []

    for key in ranked_keys:
        chunk_doc = section_docs[key]
        original = _original_sections.get(key, {})

        # Prefer the full original section, with chunk text as fallback.
        full_text = (
            original.get("text")
            or original.get("section_text")
            or original.get("full_text")
            or original.get("content")
            or chunk_doc.get("text", "")
        )

        result = {
            "key": key,
            "act_code": key[0],
            "act_name": (
                original.get("act_name")
                or chunk_doc.get("act_name")
                or key[0]
            ),
            "section_number": key[1],
            "section_title": (
                original.get("section_title")
                or chunk_doc.get("section_title")
                or "(title unavailable)"
            ),
            "text": full_text,
            "start_page": (
                original.get("start_page")
                or chunk_doc.get("start_page")
            ),
            "end_page": (
                original.get("end_page")
                or chunk_doc.get("end_page")
            ),
            "score": section_scores[key],
            "rrf_score": section_scores[key],
        }

        results.append(result)

    return results


def search_legal(question, top_k=DEFAULT_TOP_K, act_filter=None):
    """
    Streamlit-friendly search function.

    Parameters:
        question: Natural-language legal query.
        top_k: Number of results to return.
        act_filter: None, "All Acts", "BNS", "BNSS", or "BSA".

    Returns:
        List of result dictionaries with section text and metadata.
    """
    ranked = retrieve(question)

    if act_filter and act_filter != "All Acts":
        act_filter = act_filter.upper()
        ranked = [
            result for result in ranked
            if result["act_code"] == act_filter
        ]

    return ranked[:max(1, int(top_k))]


def close():
    """Close the persistent Qdrant client when shutting down."""
    global _client

    if _client is not None:
        _client.close()
        _client = None


if __name__ == "__main__":
    try:
        question = input("Enter your legal question: ").strip()

        for rank, result in enumerate(search_legal(question, top_k=5), 1):
            print(
                f"\n{rank}. {result['act_code']} "
                f"§{result['section_number']} — "
                f"{result['section_title']}"
            )
            print(f"RRF score: {result['score']:.6f}")
            print(result["text"][:500])

    finally:
        close()
