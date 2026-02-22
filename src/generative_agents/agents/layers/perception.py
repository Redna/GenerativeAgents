import dspy
from typing import List, Tuple
from generative_agents.common.events import PerceivedEvent, EventType
from generative_agents.common.percept import Percept
from generative_agents.common.neural_types import AgentState
from generative_agents.intelligence.modules.perception import heuristic_poignance
from generative_agents.common.logging import log_agent
from generative_agents.common import global_state

class SensoryProcessingLayer(dspy.Module):
    """
    Layer 1: Perception & Filtering.
    Processes raw percepts, rates their poignancy, and converts them to PerceivedEvents.
    """
    def __init__(self):
        super().__init__()

    def forward(self, percept: Percept, state: AgentState) -> List[PerceivedEvent]:
        processed_events = []
        
        for event in percept.events:
            type_ = EventType.EVENT
            
            # Identify Chat events
            if event.entity_id == state.name and "chat with" in event.description:
                type_ = EventType.CHAT

            # Format description if address is entity_id
            if type_ == EventType.EVENT and ":" in event.entity_id:
                event.description = f"{event.entity_id.split(':')[-1]} is {event.description}"

            # Calculate Poignancy
            poignancy = self._rate_perception_poignancy(state.name, state.identity_description, type_, event.description)
            
            log_agent(state.name, f"Event '{event.description}' poignancy rated as {poignancy}", "DEBUG")
            
            processed_event = PerceivedEvent(
                event_type=type_,
                poignancy=poignancy,
                depth=1,
                description=event.description,
                entity_id=event.entity_id,
                created=getattr(event, 'created', None) or global_state.time.time,
                expiration=getattr(event, 'expiration', None),
                tile=event.tile
            )
            processed_events.append(processed_event)
            
        return processed_events

    def _rate_perception_poignancy(self, agent_name: str, identity: str, event_type: EventType, description: str) -> float:
        """Heuristic scoring — zero LLM calls."""
        if "idle" in description:
            return 0.1
        return heuristic_poignance(event_type.value, description)
