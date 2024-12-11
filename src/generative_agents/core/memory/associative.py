from typing import List

from generative_agents.persistence import database
from generative_agents.core.events import EventType, PerceivedEvent

class AssociativeMemory:

    def __init__(self, agent_name, retention):
        self.agent_name = agent_name
        self.retention = retention
        self.database = database.AgentCollection(agent_name)

    def add(self, event: PerceivedEvent) -> PerceivedEvent:
        db_event = self.database.get_by_hash(event.hash_key)

        if not db_event:
            memory_entry = self.database.add(self.agent_name, event.to_db_entry())
            db_event = PerceivedEvent.from_db_entry(memory_entry)
        else:
            db_event = PerceivedEvent.from_db_entry(db_event[-1])

        return db_event

    @property
    def latest_events_summary(self):
        last_entries = self.database.get_last_entries(limit=50)
        events = [PerceivedEvent.from_db_entry(memory) for memory in last_entries]
        return [event.spo_summary for event in events]

    def retrieve_relevant_entries(self, context: list[str] | str, limit=50) -> List[PerceivedEvent]:
        return self.retrieve_relevant_entries_by_type(context, limit=limit)

    def retrieve_relevant_entries_by_type(self, context: List[str], event_type: EventType = None, limit=50) -> List[PerceivedEvent]:
        memories = []
        if isinstance(context, str):
            context = [context]

        for context_element in context:
            if event_type:
                memories += self.database.get_by_type(self.agent_name, context_element, event_type.value, limit=limit // len(context))
            else:
                memories += self.database.get(self.agent_name, context_element, limit=limit // len(context))

        return [PerceivedEvent.from_db_entry(memory) for memory in memories]

    def last_conversation_with(self, agent_name: str) -> PerceivedEvent:
        last_chat = self.database.get_last_chat(self.agent_name, agent_name)
        return PerceivedEvent.from_db_entry(last_chat) if last_chat else None

    def get_most_recent_memories(self, most_recent=0):
        memories = self.database.get_last_entries(limit=most_recent)
        return [PerceivedEvent.from_db_entry(memory) for memory in memories]