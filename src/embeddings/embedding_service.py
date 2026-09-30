"""Local embedding model wrapper.

We embed locally (sentence-transformers) rather than via an API so ingestion
and retrieval never require GROQ_API_KEY - only the final recommendation
LLM call needs a key. This also makes retrieval free and deterministic,
which matters for the demo and for tests.
"""
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from src.config import EMBEDDING_MODEL_NAME


@lru_cache(maxsize=1)
def get_embedding_function() -> HuggingFaceEmbeddings:
    """Return a cached HuggingFace embedding model instance.

    Cached because loading the model from disk/network takes a couple of
    seconds; callers (ingestion, retriever, tests) can call this freely.
    """
    # normalize_embeddings=True makes vector magnitude 1, so cosine
    # similarity and (squared) L2 distance rank documents identically -
    # without this, Chroma's default L2 space lets embedding magnitude
    # drown out semantic similarity.
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
    )
