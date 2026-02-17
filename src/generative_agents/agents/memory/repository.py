import json
import os
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import asdict

from generative_agents.common import global_state
from generative_agents.persistence.cachable_sentence_transformer import (
    CachableSentenceTransformer,
)

# Reuse existing embedding model wrapper
_model = CachableSentenceTransformer("sentence-transformers/all-mpnet-base-v2")


class JSONMemoryRepository:
    def __init__(self, agent_name: str, data_dir: str = "data/agents"):
        self.agent_name = agent_name
        self.base_dir = os.path.join(data_dir, agent_name)
        self.memory_file = os.path.join(self.base_dir, "memory.json")
        
        self.memories: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        
        # Load immediately on init
        self._load()

    def _load(self):
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir, exist_ok=True)
            
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r') as f:
                try:
                    data = json.load(f)
                    self.memories = data.get("memories", [])
                except json.JSONDecodeError:
                    self.memories = []
        else:
            self.memories = []
            
        # Rebuild embeddings cache (fast enough for small scale)
        self._rebuild_embeddings()

    def _rebuild_embeddings(self):
        if not self.memories:
            self.embeddings = None
            return

        texts = [m["content"] for m in self.memories]
        # This uses the cached model
        # Check if texts is empty list
        if not texts:
            self.embeddings = None
            return

        self.embeddings = _model.encode(texts)

    def save(self):
        data = {
            "agent_name": self.agent_name,
            "last_updated": datetime.now().isoformat(),
            "memories": self.memories
        }
        with open(self.memory_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def add_memory(self, memory_object: Any) -> Dict[str, Any]:
        """
        Accepts a Pydantic model or dict.
        Returns the stored dict.
        """
        # Convert Pydantic to dict if needed
        if hasattr(memory_object, 'model_dump'):
            entry = memory_object.model_dump()
        elif hasattr(memory_object, 'dict'):
             entry = memory_object.dict()
        else:
            entry = dict(memory_object)

        # Ensure ID and timestamps
        if not entry.get("id"):
            import uuid
            entry["id"] = str(uuid.uuid4())
            
        # Ensure embedding is computed (for internal use, not necessarily stored in JSON if we rebuild)
        # But we need it for instant search
        text = entry.get("content", "")
        embedding = _model.encode([text])[0]
        
        # We don't store the embedding in JSON to keep it readable, 
        # but we append it to our in-memory numpy array
        if self.embeddings is None:
            self.embeddings = np.array([embedding])
        else:
            self.embeddings = np.vstack([self.embeddings, embedding])
            
        self.memories.append(entry)
        self.save()
        return entry

    def retrieve(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        if not self.memories or self.embeddings is None:
            return []

        query_embedding = _model.encode([query])[0]
        
        # Cosine similarity: (A . B) / (||A|| * ||B||)
        # _model.encode produces normalized vectors usually, but let's be safe
        embedding_norms = np.linalg.norm(self.embeddings, axis=1)
        query_norm = np.linalg.norm(query_embedding)
        
        # Avoid division by zero
        if query_norm == 0:
            return []
            
        # Ensure embedding_norms are not zero
        nonzero_indices = embedding_norms > 1e-9
        
        if not np.any(nonzero_indices):
             return []

        filtered_embeddings = self.embeddings[nonzero_indices]
        filtered_memories = [self.memories[i] for i in range(len(self.memories)) if nonzero_indices[i]]
        
        if not filtered_memories:
            return []
            
        scores = np.dot(filtered_embeddings, query_embedding) / (
            embedding_norms[nonzero_indices] * query_norm
        )
        
        # Get top K indices
        # If limit is larger than population, take all
        k = min(limit, len(scores))
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            # Add score to result for transparency
            item = filtered_memories[idx].copy()
            item["_score"] = float(scores[idx])
            results.append(item)
            
        return results

    def get_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        for m in self.memories:
            if m["id"] == memory_id:
                return m
        return None

    def add_relation(self, source_id: str, target_id: str, relation: str):
        for m in self.memories:
            if m["id"] == source_id:
                if "related_events" not in m:
                    m["related_events"] = []
                # Check for duplicates
                for link in m["related_events"]:
                    if link["id"] == target_id and link["relation"] == relation:
                        return
                m["related_events"].append({"id": target_id, "relation": relation})
                self.save()
                return

    def get_related(self, source_id: str, relation: Optional[str] = None) -> List[Tuple[str, str]]:
        related = []
        for m in self.memories:
            if m["id"] == source_id:
                links = m.get("related_events", [])
                for link in links:
                    if relation is None or link["relation"] == relation:
                        related.append((link["id"], link["relation"]))
                return related
        return []
