from __future__ import annotations

import os
from functools import lru_cache
from typing import Iterable, List

import cohere

MODEL_NAME = "embed-multilingual-light-v3.0"


class CohereClient:
    def __init__(self) -> None:
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise RuntimeError("COHERE_API_KEY is not set")
        self._client = cohere.Client(api_key)

    def embed(self, texts: Iterable[str], input_type: str) -> List[List[float]]:
        response = self._client.embed(
            texts=list(texts),
            model=MODEL_NAME,
            input_type=input_type,
        )
        return response.embeddings


_client: CohereClient | None = None


def get_client() -> CohereClient:
    global _client
    if _client is None:
        _client = CohereClient()
    return _client


@lru_cache(maxsize=2048)
def embed_query_cached(text: str) -> List[float]:
    return get_client().embed([text], input_type="search_query")[0]


@lru_cache(maxsize=4096)
def embed_document_cached(text: str) -> List[float]:
    return get_client().embed([text], input_type="search_document")[0]


def embed_documents_batch(texts: Iterable[str], input_type: str) -> List[List[float]]:
    return get_client().embed(texts, input_type=input_type)
