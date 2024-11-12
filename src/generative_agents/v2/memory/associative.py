from dataclasses import dataclass, field
from queue import LifoQueue
from typing import Dict, List

from pydantic import BaseModel

from generative_agents.persistence import database
from generative_agents.core.events import PerceivedEvent

@dataclass
class LastEntryStore:

    max_size: int
    entries: List[PerceivedEvent] = field(default_factory=list)
    entry_hashmap: Dict[str, int] = field(default_factory=dict)

    def put(self, event: PerceivedEvent):
        if len(self.entries) == self.max_size:
           self._pop()
        
        self._put(event)
        
    def get(self, most_recent=0):
        if most_recent == 0:
            return self.entries
        else:
            return self.entries[-most_recent:] 

    def _pop(self):
        entry = self.entries.pop(0)
        del self.entry_hashmap[entry.id]
        self.entry_hashmap = {key: value - 1 for key, value in self.entry_hashmap.items()}

    def _put(self, event: PerceivedEvent):
        if event.id in self.entry_hashmap:
            index = self.entry_hashmap[event.id]
            self.entries[index] = event
        else:
            self.entries.append(event)
            self.entry_hashmap[event.id] = len(self.entries) - 1

class AssociativeMemory:

    def __init__(self, agent_name, retention):
        self.agent_name = agent_name
        self.retention = retention
        self.last_entries = LastEntryStore(max_size=retention)

    def add(self, event: PerceivedEvent) -> PerceivedEvent:
        db_event = database.get_by_hash(self.agent_name, event.hash_key)

        if not db_event:
            memory_entry = database.add(self.agent_name, event.to_db_entry())
            db_event = PerceivedEvent.from_db_entry(memory_entry)
        else:
            db_event = PerceivedEvent.from_db_entry(db_event[-1])

        self.last_entries.put(db_event)
        return db_event

    @property
    def latest_events_summary(self):
        return [event.spo_summary for event in self.last_entries.entries]

    def retrieve_relevant_entries(self, context: List[str], limit=50) -> List[PerceivedEvent]:
        memories = []

        for context_element in context:
            memories += database.get(self.agent_name, context_element, limit=limit // len(context))
        
        memories = database.get(self.agent_name, context, limit=50)
        return [PerceivedEvent.from_db_entry(memory) for memory in memories]
    
    def last_conversation_with(self, agent_name: str) -> PerceivedEvent:
        last_chat = database.get_last_chat(self.agent_name, agent_name)
        return PerceivedEvent.from_db_entry(last_chat) if last_chat else None

    def active_conversation_with(self, agent_name: str) -> PerceivedEvent:
        active_chat = database.get_active_chat(self.agent_name, agent_name)
        return PerceivedEvent.from_db_entry(active_chat) if active_chat else None
    
    def get_most_recent_memories(self, most_recent=0):
        memories = self.last_entries.get(most_recent=most_recent)
        return [PerceivedEvent.from_db_entry(memory) for memory in memories]
