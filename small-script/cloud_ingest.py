
import json
import uuid
from pathlib import Path

import numpy as np
import streamlit as st
from qdrant_client import QdrantClient, models

BASE_DIR = Path(__file__).resolve().parent
COLLECTION_NAME = "indian_legal_sections_v2"
VECTOR_SIZE = 384
BATCH_SIZE = 100

ID_NAMESPACE = uuid.UUID(
    "12345678-1234-5678-1234-567812345678"
)


@st.cache_resource
def get_qdrant_client():
    """Use Qdrant Cloud if configured; otherwise use local storage."""
    url = st.secrets.get("QDRANT_URL", "")
    api_key = st.secrets.get("QDRANT_API_KEY", "")

    if url:
        return QdrantClient(url=url, api_key=api_key or None)

    return QdrantClient(path=str(BASE_DIR / "qdrant_storage"))


def ensure_collection(client):
    if client.collection_exists(COLLECTION_NAME):
        info = client.get_collection(COLLECTION_NAME)
        if info.points_count and info.points_count > 0:
            return

    vectors_file = BASE_DIR / "embeddings.npz"
    metadata_file = BASE_DIR / "embedding_metadata.jsonl"

    if not vectors_file.exists() or not metadata_file.exists():
        raise FileNotFoundError(
            "Missing embeddings.npz or embedding_metadata.jsonl."
        )

    with np.load(vectors_file, allow_pickle=False) as data:
        ids = data["ids"].astype(str)
        vectors = data["embeddings"].astype(np.float32)

    with metadata_file.open(encoding="utf-8") as f:
        documents = [json.loads(line) for line in f if line.strip()]

    if len(ids) != len(vectors) or len(ids) != len(documents):
        raise ValueError("Embedding and metadata counts do not match.")

    if vectors.shape != (len(documents), VECTOR_SIZE):
        raise ValueError(f"Unexpected embedding shape: {vectors.shape}")

    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings contain invalid numeric values.")

    for i, document in enumerate(documents):
        if str(document.get("id")) != ids[i]:
            raise ValueError(f"Document ID mismatch at row {i}.")

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )

    for start in range(0, len(documents), BATCH_SIZE):
        points = []

        for i in range(start, min(start + BATCH_SIZE, len(documents))):
            payload = {
                key: value
                for key, value in documents[i].items()
                if value is not None
            }
            payload["chunk_id"] = ids[i]

            points.append(
                models.PointStruct(
                    id=str(uuid.uuid5(ID_NAMESPACE, ids[i])),
                    vector=vectors[i].tolist(),
                    payload=payload,
                )
            )

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )

    count = client.count(
        collection_name=COLLECTION_NAME,
        exact=True,
    ).count

    if count < len(documents):
        raise RuntimeError(
            f"Expected {len(documents)} points but found {count}."
        )
