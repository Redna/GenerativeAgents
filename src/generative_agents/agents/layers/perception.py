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
        current_observations = set()
        
        for event in percept.events:
            type_ = EventType.EVENT
            
            # Identify Chat events
            if event.entity_id == state.name and "chatting with" in event.description:
                type_ = EventType.CHAT

            # ---> NEW: Identify Overheard Conversations (Someone else is talking) <---
            elif ", saying: '" in event.description:
                speaker_name = event.entity_id
                # Extract listener name from the string "chatting with [Name], saying:"
                try:
                    listener_name = event.description.split("chatting with ")[1].split(",")[0]
                    # If the perceiving agent is NOT the speaker and NOT the listener, they are a bystander
                    if state.name != speaker_name and state.name != listener_name:
                        type_ = EventType.OBSERVATION
                        # Restructure the description so the bystander remembers it in the 3rd person
                        utterance = event.description.split(", saying: '")[1].rstrip("'")
                        event.description = f"overheard {speaker_name} say to {listener_name}: '{utterance}'"
                        log_agent(state.name, f"Overheard snippet: {utterance}", "DEBUG")
                except IndexError:
                     pass

            # Format description if address is entity_id
            elif type_ == EventType.EVENT and ":" in event.entity_id:
                event.description = f"{event.entity_id.split(':')[-1]} is {event.description}"

            current_observations.add(event.description)

            # Deduplicate generic world events to prevent hyper-frequent reflection triggers
            if type_ == EventType.EVENT and event.description in state.working_memory.last_observations_cache:
                continue

            # Calculate Poignancy
            poignancy = self._rate_perception_poignancy(state.name, state.identity_description, type_, event.description)
            
            if poignancy > 0.1: # Only log non-trivial poignancy scores above idle to reduce noise
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
            
        state.working_memory.last_observations_cache = current_observations
        return processed_events

    def _rate_perception_poignancy(self, agent_name: str, identity: str, event_type: EventType, description: str) -> float:
        """Heuristic scoring — zero LLM calls."""
        if "idle" in description:
            return 0.1
        return heuristic_poignance(event_type.value, description)
