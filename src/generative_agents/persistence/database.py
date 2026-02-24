from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
from pydantic import BaseModel
import logging

from generative_agents.agents.memory.repository import QdrantMemoryRepository

# ---------------------------------------------------------------------------
# Singleton registry — one repo per agent
# ---------------------------------------------------------------------------
_repositories: Dict[str, QdrantMemoryRepository] = {}


class MemoryType(Enum):
    CHAT = "chat"
    OBSERVATION = "observation"
    EVENT = "event"
    THOUGHT = "thought"


class ConversationFilling(BaseModel):
    name: str = ""
    utterance: str = ""
    end: bool = False


class MemoryEntry(BaseModel):
    """Canonical in-memory representation of a memory."""
    # Core
    id: str
    content: str
    created_at: datetime
    last_accessed_at: datetime
    importance: float = 0.5
    # Metadata
    memory_type: str
    depth: int = 1
    entity_id: str = ""
    # Optional
    poignancy: float = 0.5
    keywords: List[str] = []
    filling: List[Union[ConversationFilling, dict]] = []
    hash_key: Optional[str] = None
    expiration_date: Optional[datetime] = None
    # Graph links (edges stored in Qdrant payload)
    related_events: List[Dict[str, str]] = []


# ---------------------------------------------------------------------------
# Repository access
# ---------------------------------------------------------------------------

def get_repository(agent_name: str) -> QdrantMemoryRepository:
    if agent_name not in _repositories:
        from generative_agents.common.utils import get_project_root
        import os
        
        # Save Qdrant persistent collections to physical disk to survive server restarts
        storage_path = os.path.join(get_project_root(), "storage", "qdrant", agent_name)
        os.makedirs(storage_path, exist_ok=True)
        
        _repositories[agent_name] = QdrantMemoryRepository(agent_name, path=storage_path)
    return _repositories[agent_name]


def initialize_database(recreate: bool = False):
    pass  # No-op — Qdrant manages its own schema


def initialize_agent(agent_name: str):
    get_repository(agent_name)


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def add(agent_name: str, memory_entry: Any) -> MemoryEntry:
    repo = get_repository(agent_name)
    stored = repo.add_memory(memory_entry)
    return _to_entry(stored)


def get(agent_name: str, context: str, limit: int = 5) -> List[MemoryEntry]:
    repo = get_repository(agent_name)
    results = repo.retrieve(context, limit=limit)
    return [_to_entry(d) for d in results]


def get_by_hash(agent_name: str, hash_key: str) -> List[MemoryEntry]:
    """Linear scan via retrieve — hash_key is stored in payload."""
    repo = get_repository(agent_name)
    # Approximate: search by hash_key text (not ideal, but keeps interface)
    results = repo.retrieve(hash_key, limit=20)
    return [_to_entry(d) for d in results if d.get("hash_key") == hash_key]


# ---------------------------------------------------------------------------
# Chat helpers — search by memory_type filter
# ---------------------------------------------------------------------------

def get_active_chat(agent_name: str, with_agent: str) -> Optional[MemoryEntry]:
    repo = get_repository(agent_name)
    results = repo.retrieve(f"chatting with {with_agent}", limit=20)
    for d in results:
        if d.get("memory_type") == MemoryType.CHAT.value and with_agent in d.get("content", ""):
            filling = d.get("filling", [])
            if filling:
                last = filling[-1]
                is_end = last.get("end") if isinstance(last, dict) else getattr(last, "end", False)
                if not is_end:
                    return _to_entry(d)
            else:
                return _to_entry(d)
    return None


def get_last_chat(agent_name: str, with_agent: str) -> Optional[MemoryEntry]:
    repo = get_repository(agent_name)
    results = repo.retrieve(f"chatting with {with_agent}", limit=20)
    for d in results:
        if d.get("memory_type") == MemoryType.CHAT.value and with_agent in d.get("content", ""):
            return _to_entry(d)
    return None


# ---------------------------------------------------------------------------
# Graph operations — delegate to per-agent repo
# ---------------------------------------------------------------------------

def add_event_link(source_id: str, target_id: str, relation_type: str):
    """Adds a directed graph edge. Scans all repos to find the owner."""
    for repo in _repositories.values():
        if repo.get_by_id(source_id):
            repo.add_relation(source_id, target_id, relation_type)
            return


def get_related_events(source_id: str, relation_type: Optional[str] = None) -> List[Tuple[str, str]]:
    """Returns graph edges for source_id across all repos."""
    for repo in _repositories.values():
        result = repo.get_related(source_id, relation_type)
        if result:
            return result
    return []


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _to_entry(d: Dict) -> MemoryEntry:
    try:
        return MemoryEntry(**d)
    except Exception:
        # Ensure required fields have fallbacks
        now = datetime.now().isoformat()
        d.setdefault("id", "unknown")
        d.setdefault("content", "")
        d.setdefault("created_at", now)
        d.setdefault("last_accessed_at", now)
        d.setdefault("memory_type", MemoryType.OBSERVATION.value)
        return MemoryEntry(**{k: v for k, v in d.items() if k in MemoryEntry.model_fields})
