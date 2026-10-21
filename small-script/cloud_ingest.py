
import os
from pathlib import Path

import streamlit as st
from qdrant_client import QdrantClient, models

BASE_DIR = Path(__file__).resolve().parent
COLLECTION_NAME = "indian_legal_sections_v2"
VECTOR_SIZE = 384


def _get_setting(name):
    """Read configuration from environment variables or Streamlit Secrets."""
    value = os.getenv(name)

    if value:
        return value

    try:
        return st.secrets.get(name, "")
    except Exception:
        # Local development may not have a Streamlit secrets file.
        return ""


@st.cache_resource
def get_qdrant_client():
    """Use Qdrant Cloud when configured; otherwise use local storage."""
    url = _get_setting("QDRANT_URL")
    api_key = _get_setting("QDRANT_API_KEY")

    if url and api_key:
        print("QDRANT CONNECTION: CLOUD")
        return QdrantClient(
            url=url,
            api_key=api_key,
            timeout=30,
        )

    if url or api_key:
        raise RuntimeError(
            "Both QDRANT_URL and QDRANT_API_KEY must be configured."
        )

    print("QDRANT CONNECTION: LOCAL STORAGE")
    return QdrantClient(path=str(BASE_DIR / "qdrant_storage"))


def ensure_collection(client):
    """
    Validate the existing collection without creating or ingesting data.
    """
    if not client.collection_exists(COLLECTION_NAME):
        raise RuntimeError(
            f"Qdrant collection '{COLLECTION_NAME}' does not exist. "
            "Check your Qdrant connection and collection name."
        )

    info = client.get_collection(COLLECTION_NAME)
    config = info.config.params.vectors

    if not isinstance(config, models.VectorParams):
        raise ValueError(
            "Expected a collection with one unnamed vector."
        )

    if config.size != VECTOR_SIZE:
        raise ValueError(
            f"Expected {VECTOR_SIZE}-dimensional vectors, "
            f"found {config.size}."
        )

    if config.distance != models.Distance.COSINE:
        raise ValueError("Expected cosine distance for the collection.")

    count = client.count(
        collection_name=COLLECTION_NAME,
        exact=True,
    ).count

    if count == 0:
        raise RuntimeError(
            f"Qdrant collection '{COLLECTION_NAME}' is empty. "
            "Run ingestion separately if needed."
        )

    print(
        f"Qdrant collection verified: {COLLECTION_NAME} "
        f"({count} points)"
    )
