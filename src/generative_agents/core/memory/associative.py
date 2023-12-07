from queue import LifoQueue
from typing import Dict, List

from pydantic import BaseModel

from generative_agents.persistence import database
from generative_agents.core.events import PerceivedEvent

class LastEntryStore(BaseModel):

    max_size: int
    entries: List[PerceivedEvent] = []
    entry_hashmap: Dict[str, int] = {}

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
        database.add(self.agent_name, event.to_db_entry())
        self.last_entries.put(event)
        return event

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
