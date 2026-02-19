import dspy
from typing import List, Tuple
from generative_agents.common.events import PerceivedEvent, EventType
from generative_agents.common.percept import Percept
from generative_agents.common.neural_types import AgentState
from generative_agents.intelligence.modules.perception import PoignanceRater
from generative_agents.common.logging import log_agent

class SensoryProcessingLayer(dspy.Module):
    """
    Layer 1: Perception & Filtering.
    Processes raw percepts, rates their poignancy, and converts them to PerceivedEvents.
    """
    def __init__(self):
        super().__init__()
        self.poignance_rater = PoignanceRater()

    def forward(self, percept: Percept, state: AgentState) -> List[PerceivedEvent]:
        processed_events = []
        
        for event in percept.events:
            if not event.predicate:
                event.predicate = "is"

            type_ = EventType.EVENT
            
            # Identify Chat events
            if event.subject == state.name and event.predicate == "chat with":
                type_ = EventType.CHAT

            # Format description if address is subject
            if type_ == EventType.EVENT and ":" in event.subject:
                event.description = f"{event.subject.split(':')[-1]} is {event.description}"

            # Calculate Poignancy
            poignancy = self._rate_perception_poignancy(state.name, state.identity_description, type_, event.description)
            
            log_agent(state.name, f"Event '{event.description}' poignancy rated as {poignancy}", "DEBUG")
            
            processed_event = PerceivedEvent(
                event_type=type_,
                poignancy=poignancy,
                depth=1,
                description=event.description,
                subject=event.subject,
                predicate=event.predicate,
                object_=event.object_,
                created=event.created,
                expiration=event.expiration,
                tile=event.tile
            )
            processed_events.append(processed_event)
            
        return processed_events

    def _rate_perception_poignancy(self, agent_name: str, identity: str, event_type: EventType, description: str) -> float:
        if "idle" in description:
            return 0.1

        score = self.poignance_rater(
            agent_name, identity, event_type.value, description
        )
        return int(score) / 10.0
