"""Ollama embedding adapter for the QueryEngine provider boundary."""

from __future__ import annotations

import json
from typing import Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from proposal_ai_agent.embeddings.providers.base import EmbeddingProvider, Vector


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Generate query embeddings using Ollama's local ``/api/embed`` endpoint."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434/api", timeout: int = 60) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        normalized_base_url = base_url.rstrip("/")
        self._model = model
        self._base_url = (
            normalized_base_url
            if normalized_base_url.endswith("/api")
            else f"{normalized_base_url}/api"
        )
        self._timeout = timeout

    def embed(self, text: str) -> Vector:
        return self.embed_batch((text,))[0]

    def embed_batch(self, texts: Sequence[str]) -> tuple[Vector, ...]:
        if not texts:
            return ()
        request = Request(
            f"{self._base_url}/embed", data=json.dumps({"model": self._model, "input": list(texts)}).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urlopen(request, timeout=self._timeout) as response:
                raw_body = response.read().decode("utf-8")
        except HTTPError as error:
            raise RuntimeError(f"Ollama embeddings returned HTTP {error.code}") from error
        except URLError as error:
            raise RuntimeError("Ollama embeddings connection failed") from error
        except TimeoutError as error:
            raise RuntimeError("Ollama embeddings request timed out") from error
        except OSError as error:
            raise RuntimeError("Ollama embeddings request failed") from error
        try:
            embeddings = json.loads(raw_body)["embeddings"]
        except (TypeError, KeyError, ValueError) as error:
            raise RuntimeError("Ollama embeddings returned invalid output") from error
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise RuntimeError("Ollama embeddings returned an unexpected vector count")
        try:
            return tuple(tuple(float(value) for value in vector) for vector in embeddings)
        except (TypeError, ValueError) as error:
            raise RuntimeError("Ollama embeddings returned invalid vectors") from error
