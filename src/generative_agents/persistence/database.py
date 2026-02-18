from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any, Union
from enum import Enum
from pydantic import BaseModel
import logging

from generative_agents.agents.memory.repository import JSONMemoryRepository

# Singleton registry for repositories
_repositories: Dict[str, JSONMemoryRepository] = {}

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
    # Core fields
    id: str
    content: str
    created_at: datetime
    last_accessed_at: datetime
    importance: float = 0.5
    
    # Metadata
    memory_type: str
    depth: int = 1
    subject: str = ""
    predicate: str = ""
    object_: str = ""
    
    # Optional
    poignancy: float = 0.5
    keywords: List[str] = []
    filling: List[Union[ConversationFilling, dict]] = []
    hash_key: Optional[str] = None
    expiration_date: Optional[datetime] = None
    
    # Graph Links (New JSON structure)
    related_events: List[Dict[str, str]] = []  # {"id": "uuid", "relation": "caused"}

def get_repository(agent_name: str) -> JSONMemoryRepository:
    if agent_name not in _repositories:
        _repositories[agent_name] = JSONMemoryRepository(agent_name)
    return _repositories[agent_name]

def initialize_database(recreate: bool = False):
    # No-op for file based system
    pass

def initialize_agent(agent_name: str):
    get_repository(agent_name)

def add(agent_name: str, memory_entry: Any) -> MemoryEntry:
    repo = get_repository(agent_name)
    
    # If it's a Pydantic model, convert to dict for storage but keep object for return
    stored_data = repo.add_memory(memory_entry)
    
    return MemoryEntry(**stored_data)

def get(agent_name: str, context: str, limit=5) -> List[MemoryEntry]:
    repo = get_repository(agent_name)
    results = repo.retrieve(context, limit=limit)
    return [MemoryEntry(**data) for data in results]

def get_by_hash(agent_name: str, hash_key: str):
    # Linear scan 
    repo = get_repository(agent_name)
    found = []
    for m in repo.memories:
        if m.get("hash_key") == hash_key:
            found.append(MemoryEntry(**m))
    return found

def get_active_chat(agent_name: str, with_agent: str) -> Optional[MemoryEntry]:
    repo = get_repository(agent_name)
    sorted_memories = sorted(repo.memories, key=lambda x: x["created_at"], reverse=True)
    
    for m in sorted_memories:
        if m["memory_type"] == MemoryType.CHAT.value and m.get("object_") == with_agent:
            fillings = m.get("filling", [])
            if fillings:
                last_fill = fillings[-1]
                is_end = last_fill.get("end") if isinstance(last_fill, dict) else last_fill.end
                if not is_end:
                    return MemoryEntry(**m)
            else:
                return MemoryEntry(**m)
    return None

def get_last_chat(agent_name: str, with_agent: str) -> Optional[MemoryEntry]:
    repo = get_repository(agent_name)
    sorted_memories = sorted(repo.memories, key=lambda x: x["created_at"], reverse=True)
    for m in sorted_memories:
        if m["memory_type"] == MemoryType.CHAT.value and m.get("object_") == with_agent:
            return MemoryEntry(**m)
    return None

# Graph operations
def add_event_link(source_id: str, target_id: str, relation_type: str):
    # We must scan repositories because agent_name is lost in this global function signature
    for name, repo in _repositories.items():
        repo.add_relation(source_id, target_id, relation_type)

def get_related_events(source_id: str, relation_type: Optional[str] = None) -> List[Tuple[str, str]]:
    related = []
    for name, repo in _repositories.items():
        res = repo.get_related(source_id, relation_type)
        if res:
            related.extend(res)
            # Typically a memory ID is unique globally (UUID), but checking all is safer if ID reused
    return related
