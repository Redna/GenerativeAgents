from queue import LifoQueue

from persistence import database
from core.events import PerceivedEvent

class AssociativeMemory:

    def __init__(self, agent_name, retention):
        self.agent_name = agent_name
        self.retention = retention
        self.last_entries = LifoQueue(maxsize=retention)

    def add(self, event: PerceivedEvent) -> PerceivedEvent:
        database.add(self.agent_name, event.to_db_entry())
        self.last_entries.put(event)
        return event

    @property
    def latest_events_summary(self):
        return [event.spo_summary for event in self.last_entries.queue]

    def retrieve_relevant_events(self, subject: str, predicate: str, object_: str):
        return database.get(self.agent_name, subject, predicate, object_)
