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
    "subject":       str,
    "predicate":     str,
    "object_":       str,
    "related_events": [              # graph edges
      {"id": str, "relation": str},
      ...
    ]
  }
"""
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from generative_agents.persistence.cachable_sentence_transformer import CachableSentenceTransformer

VECTOR_DIM = 768       # sentence-transformers/all-mpnet-base-v2
COLLECTION = "memories"

# Lazy singleton embedding model
_model: "CachableSentenceTransformer | None" = None


def _get_model() -> CachableSentenceTransformer:
    global _model
    if _model is None:
        _model = CachableSentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    return _model


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
        vector = _get_model().encode([text])[0].tolist()

        payload = {
            "str_id":        entry["id"],
            "content":       text,
            "memory_type":   str(entry.get("memory_type", "")),
            "depth":         int(entry.get("depth", 1)),
            "created_ts":    _ts(entry.get("created_at")),
            "importance":    float(entry.get("importance", 0.5)),
            "subject":       str(entry.get("subject", "")),
            "predicate":     str(entry.get("predicate", "")),
            "object_":       str(entry.get("object_", "")),
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

        query_vec = _get_model().encode([query])[0].tolist()
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
