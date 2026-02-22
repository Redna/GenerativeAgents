"""
agents/memory/repository.py
Phase 6 (clean): Pure Qdrant store — no JSON, no numpy.
  - Vectors stored in Qdrant (persistent local path or :memory:)
  - Graph edges stored as payload lists inside each point
  - No re-indexing on startup needed; persistence handled by Qdrant itself

Payload schema per point:
  {
    "str_id":        str,            # original UUID string
    "content":       str,            # memory text (for retrieval without separate store)
    "memory_type":   str,
    "depth":         int,
    "created_ts":    float,          # Unix timestamp
    "importance":    float,
    "entity_id":     str,
    "related_events": [              # graph edges
      {"id": str, "relation": str},
      ...
    ]
  }
"""
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

import hashlib
import json
import os
import pickle

import httpx

# ---------------------------------------------------------------------------
# vLLM Embeddings client (replaces CachableSentenceTransformer)
# ---------------------------------------------------------------------------

EMBED_URL   = "http://pop-os:8001/v1/embeddings"
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
VECTOR_DIM  = 1024   # Qwen3-Embedding-0.6B output dimensionality
COLLECTION  = "memories"
_CACHE_DIR  = ".generation_cache/embed/"


class VLLMEmbedder:
    """Thin wrapper around the vLLM OpenAI-compatible /v1/embeddings endpoint.
    Caches results to disk to avoid redundant API calls (same interface as
    the old CachableSentenceTransformer).
    """

    def __init__(self):
        # In-memory dictionary to hold computed embeddings (avoids disk read overhead)
        self._memory_cache: dict[str, list[float]] = {}
        # Max size to prevent unbounded RAM usage over a long running simulation
        self._max_memory_cache_size = 5000 

    def encode(self, texts: list[str]) -> list[list[float]]:
        """
        Encodes a list of texts and returns their embedding vectors.
        Single-string input is also accepted (returns a 1-element list).
        """
        if isinstance(texts, str):
            texts = [texts]

        os.makedirs(_CACHE_DIR, exist_ok=True)
        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []

        # Check in-memory cache and then disk cache
        for i, text in enumerate(texts):
            cache_key = hashlib.md5(text.encode()).hexdigest()
            
            # 1. Fast path: Memory Cache
            if cache_key in self._memory_cache:
                results[i] = self._memory_cache[cache_key]
                continue
                
            # 2. Slower path: Disk Cache
            cache_path = os.path.join(_CACHE_DIR, f"{cache_key}.pkl")
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    vec = pickle.load(f)
                    results[i] = vec
                    self._add_to_memory_cache(cache_key, vec)
            else:
                uncached_indices.append(i)

        # 3. Slowest path: Batch-fetch uncached embeddings via API
        if uncached_indices:
            batch = [texts[i] for i in uncached_indices]
            response = httpx.post(
                EMBED_URL,
                headers={"Authorization": "Bearer EMPTY"},
                json={"model": EMBED_MODEL, "input": batch},
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()["data"]
            for idx, item in zip(uncached_indices, data):
                vec = item["embedding"]
                results[idx] = vec
                cache_key = hashlib.md5(texts[idx].encode()).hexdigest()
                
                # Write to disk cache
                cache_path = os.path.join(_CACHE_DIR, f"{cache_key}.pkl")
                with open(cache_path, "wb") as f:
                    pickle.dump(vec, f)
                    
                # Write to memory cache
                self._add_to_memory_cache(cache_key, vec)

        return results  # type: ignore[return-value]

    def _add_to_memory_cache(self, key: str, vec: list[float]):
        if len(self._memory_cache) >= self._max_memory_cache_size:
            # Naive pruning if limit reached (clears 20% of oldest/random entries)
            keys_to_delete = list(self._memory_cache.keys())[:int(self._max_memory_cache_size * 0.2)]
            for k in keys_to_delete:
                del self._memory_cache[k]
        self._memory_cache[key] = vec


# Lazy singleton
_embedder: VLLMEmbedder | None = None


def _get_model() -> VLLMEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = VLLMEmbedder()
    return _embedder


def _to_qdrant_id(str_id: str) -> int:
    """Convert a UUID string to a stable uint64 for Qdrant point IDs."""
    try:
        return uuid.UUID(str_id).int % (2**63)
    except (ValueError, AttributeError):
        return abs(hash(str_id)) % (2**63)


def _ts(value) -> float:
    if value is None:
        return 0.0
    if isinstance(value, datetime):
        return value.timestamp()
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except (ValueError, TypeError):
        return 0.0


class QdrantMemoryRepository:
    """
    Pure Qdrant memory store.
    Graph edges live in the 'related_events' payload field — no separate JSON file.
    Use path=':memory:' for tests / ephemeral runs, or a directory path for persistence.
    """

    def __init__(self, agent_name: str, path: str = ":memory:", data_dir: Optional[str] = None):
        self.agent_name = agent_name
        import os
        if data_dir is not None:
            # Persistent local store derived from the old JSON data_dir convention
            effective_path = os.path.join(data_dir, agent_name, "qdrant")
            os.makedirs(effective_path, exist_ok=True)
            self._client = QdrantClient(path=effective_path)
        elif path == ":memory:":
            self._client = QdrantClient(location=":memory:")
        else:
            os.makedirs(path, exist_ok=True)
            self._client = QdrantClient(path=path)
        if not self._client.collection_exists(COLLECTION):
            self._client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_memory(self, memory_object: Any) -> Dict[str, Any]:
        # Normalise to dict
        if hasattr(memory_object, "model_dump"):
            entry = memory_object.model_dump()
        elif hasattr(memory_object, "dict"):
            entry = memory_object.dict()
        else:
            entry = dict(memory_object)

        if not entry.get("id"):
            entry["id"] = str(uuid.uuid4())

        text = entry.get("content", "")
        vector = _get_model().encode([text])[0]

        payload = {
            "str_id":        entry["id"],
            "content":       text,
            "memory_type":   str(entry.get("memory_type", "")),
            "depth":         int(entry.get("depth", 1)),
            "created_ts":    _ts(entry.get("created_at")),
            "importance":    float(entry.get("importance", 0.5)),
            "entity_id":     str(entry.get("entity_id", "")),
            "related_events": entry.get("related_events", []),  # graph edges
        }

        self._client.upsert(
            collection_name=COLLECTION,
            points=[PointStruct(id=_to_qdrant_id(entry["id"]), vector=vector, payload=payload)],
        )
        return entry

    # ------------------------------------------------------------------
    # Vector search
    # ------------------------------------------------------------------

    def retrieve(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        collection_info = self._client.get_collection(COLLECTION)
        if collection_info.points_count == 0:
            return []

        query_vec = _get_model().encode([query])[0]
        result = self._client.query_points(
            collection_name=COLLECTION,
            query=query_vec,
            limit=limit,
            with_payload=True,
        )
        return [self._payload_to_dict(hit.payload, hit.score) for hit in result.points]

    # ------------------------------------------------------------------
    # Point lookup
    # ------------------------------------------------------------------

    def get_by_id(self, str_id: str) -> Optional[Dict[str, Any]]:
        results = self._client.retrieve(
            collection_name=COLLECTION,
            ids=[_to_qdrant_id(str_id)],
            with_payload=True,
        )
        if results:
            return self._payload_to_dict(results[0].payload)
        return None

    # ------------------------------------------------------------------
    # Graph edges — stored in payload["related_events"]
    # ------------------------------------------------------------------

    def add_relation(self, source_id: str, target_id: str, relation: str):
        point = self.get_by_id(source_id)
        if not point:
            return
        edges: List[Dict] = point.get("related_events", [])
        # Deduplicate
        for e in edges:
            if e.get("id") == target_id and e.get("relation") == relation:
                return
        edges.append({"id": target_id, "relation": relation})
        # Patch only the payload field
        self._client.set_payload(
            collection_name=COLLECTION,
            payload={"related_events": edges},
            points=[_to_qdrant_id(source_id)],
        )

    def get_related(self, source_id: str, relation: Optional[str] = None) -> List[Tuple[str, str]]:
        point = self.get_by_id(source_id)
        if not point:
            return []
        edges = point.get("related_events", [])
        return [
            (e["id"], e["relation"])
            for e in edges
            if relation is None or e.get("relation") == relation
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _payload_to_dict(payload: dict, score: float = 0.0) -> Dict[str, Any]:
        """Convert a Qdrant payload back to the MemoryEntry-compatible dict."""
        d = dict(payload)
        # Restore the 'id' field expected by MemoryEntry
        d["id"] = d.pop("str_id", "")
        if score:
            d["_score"] = score
        # MemoryEntry expects created_at / last_accessed_at as datetime strings
        ts = d.pop("created_ts", 0.0)
        if ts and "created_at" not in d:
            d["created_at"] = datetime.fromtimestamp(ts).isoformat()
        if "last_accessed_at" not in d:
            d["last_accessed_at"] = d.get("created_at", datetime.now().isoformat())
        return d


# ---------------------------------------------------------------------------
# Backward-compat alias so `database.py` and `system.py` still import by the
# old class name without code changes elsewhere.
# ---------------------------------------------------------------------------
JSONMemoryRepository = QdrantMemoryRepository
