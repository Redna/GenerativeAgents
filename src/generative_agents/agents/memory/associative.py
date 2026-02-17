from datetime import datetime, timedelta
from typing import List, Optional, Union

from generative_agents.common import global_state
from generative_agents.persistence import database
from generative_agents.persistence.database import ConversationFilling, MemoryEntry, MemoryType


class LastEntryStore:
    def __init__(self, key: str, period: timedelta):
        self.key = key
        self.period = period
        self.last_entry: Optional[datetime] = None

    def check(self) -> bool:
        if self.last_entry is None:
            self.last_entry = global_state.time.time
            return True
        if global_state.time.time - self.last_entry > self.period:
            self.last_entry = global_state.time.time
            return True
        return False


class AssociativeMemory:
    def __init__(self, agent_name: str, retention: float = 1.0):
        self.agent_name = agent_name
        self.retention = retention

    def add(self, memory_entry: MemoryEntry) -> MemoryEntry:
        # Pass through to new persistence layer
        return database.add(self.agent_name, memory_entry)

    def retrieve_relevant_entries(self, context: str, limit: int = 50) -> List[MemoryEntry]:
        return database.get(self.agent_name, context, limit)

    def retrieve_relevant_memories(
        self, context: str, memory_type: MemoryType, limit: int = 50
    ) -> List[MemoryEntry]:
        # Filter done in database/repository or post-processing
        # Our repository retrieve doesn't filter by type yet, so we filter here for now
        entries = database.get(self.agent_name, context, limit=limit * 2)
        filtered = [e for e in entries if e.memory_type == memory_type.value]
        return filtered[:limit]

    def add_relation(self, source_id: str, target_id: str, relation: str):
        database.add_event_link(source_id, target_id, relation)

    def get_related_events(self, source_id: str, relation: str):
        return database.get_related_events(source_id, relation)

    def get_last_chat(self, with_agent_name: str) -> Optional[MemoryEntry]:
        return database.get_last_chat(self.agent_name, with_agent_name)

    def get_active_chat(self, with_agent_name: str) -> Optional[MemoryEntry]:
        return database.get_active_chat(self.agent_name, with_agent_name)
