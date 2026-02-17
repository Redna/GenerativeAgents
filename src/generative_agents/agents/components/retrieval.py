from generative_agents.common.events import EventType, PerceivedEvent
from generative_agents.persistence import database


class Retrieval:
    def __init__(self, agent):
        self.agent = agent

    def run(self, perceived_events: list[PerceivedEvent]) -> dict:
        retrieved = {}
        for event in perceived_events:
            retrieved[event.description] = {}
            retrieved[event.description]["curr_event"] = event
            retrieved[event.description]["events"] = self._get_related_events(
                event, EventType.EVENT
            )
            retrieved[event.description]["thoughts"] = self._get_related_events(
                event, EventType.THOUGHT
            )

        return {"retrieved": retrieved}

    def _get_related_to_text(self, text: str, event_type: EventType = None):
        if event_type:
            memories = database.get_by_type(self.agent.name, text, event_type)
        else:
            memories = database.get(self.agent.name, text)

        return [PerceivedEvent.from_db_entry(memory) for memory in memories]

    def _get_related_events(self, event: PerceivedEvent, event_type: EventType = None):
        return self._get_related_to_text(event.description, event_type)
