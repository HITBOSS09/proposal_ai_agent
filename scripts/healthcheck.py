#!/usr/bin/env python3
"""Validate BDIL's required local-service and collection prerequisites."""
from __future__ import annotations
import os, socket, sys
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError
from urllib.parse import urlparse
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qdrant_client import QdrantClient

def main() -> int:
    load_dotenv(ROOT / ".env")
    ollama = os.getenv("BDIL_OLLAMA_URL", "http://localhost:11434").rstrip("/")
    if not ollama.endswith("/api"): ollama += "/api"
    qdrant_url = os.getenv("BDIL_QDRANT_URL", "http://localhost:6333")
    collection = os.getenv("BDIL_COLLECTION", "bdil_demo")
    embedding = os.getenv("BDIL_EMBEDDING_MODEL", "bge-m3")
    llm = os.getenv("BDIL_OLLAMA_MODEL", "qwen2.5:3b")
    expected_dimension = int(os.getenv("BDIL_EMBEDDING_DIMENSIONS", "1024"))
    ok = True
    try:
        import json
        with urlopen(f"{ollama}/tags", timeout=3) as response: tags = json.loads(response.read())
        names = {item.get("name") for item in tags.get("models", [])}
        for label, model in (("embedding model", embedding), ("LLM", llm)):
            passed = model in names or f"{model}:latest" in names
            print(("✓" if passed else "✗") + f" {label} installed: {model}"); ok &= passed
        print("✓ Ollama reachable")
    except (URLError, OSError, ValueError) as error:
        print(f"✗ Ollama reachable: {error}"); ok = False
    try:
        client = QdrantClient(url=qdrant_url, timeout=3)
        exists = client.collection_exists(collection); print(("✓" if exists else "✗") + f" Collection exists: {collection}"); ok &= exists
        if exists:
            info = client.get_collection(collection); dimension = info.config.params.vectors.size
            matched = dimension == expected_dimension
            print(("✓" if matched else "✗") + f" Vector dimension: {dimension} (expected {expected_dimension})"); ok &= matched
        print("✓ Qdrant reachable")
    except Exception as error:
        print(f"✗ Qdrant reachable: {error}"); ok = False
    pg = urlparse(os.getenv("BDIL_POSTGRES_URL", "postgresql://localhost:5432"))
    try:
        with socket.create_connection((pg.hostname or "localhost", pg.port or 5432), timeout=3): pass
        print("✓ PostgreSQL reachable")
    except OSError as error:
        print(f"✗ PostgreSQL reachable: {error}"); ok = False
    print(("✓" if ok else "✗") + " Environment variables valid")
    return 0 if ok else 1
if __name__ == "__main__": raise SystemExit(main())
