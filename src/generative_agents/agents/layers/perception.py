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
        new_fade_tracker = {}
        
        for event in percept.events:
            type_ = EventType.EVENT
            # Use a local variable to avoid mutating the original Event object in-place on the map
            desc = event.description
            
            # Identify Chat events
            if event.entity_id == state.name and "chatting with" in desc:
                type_ = EventType.CHAT

            # Identify Overheard Conversations
            elif ", saying: '" in desc:
                speaker_name = event.entity_id
                try:
                    listener_name = desc.split("chatting with ")[1].split(",")[0]
                    if state.name != speaker_name and state.name != listener_name:
                        type_ = EventType.OBSERVATION
                        utterance = desc.split(", saying: '")[1].rstrip("'")
                        desc = f"overheard {speaker_name} say to {listener_name}: '{utterance}'"
                        log_agent(state.name, f"Overheard snippet: {utterance}", "DEBUG")
                except IndexError:
                     pass

            # Format description if address is entity_id
            elif type_ == EventType.EVENT and ":" in event.entity_id:
                desc = f"{event.entity_id.split(':')[-1]} is {desc}"

            current_observations.add(desc)

            # Calculate Base Poignancy
            base_poignancy = self._rate_perception_poignancy(state.name, state.identity_description, type_, desc)
            
            # --- Memory Fade Filter ---
            # If we've seen this exact description recently, decay its impact
            fade_factor = state.working_memory.repeated_event_fade.get(desc, 1.0)
            final_poignancy = base_poignancy * fade_factor
            
            # We carry over the fade to the next tick, and decay it further (halving its impact each tick)
            new_fade_tracker[desc] = fade_factor * 0.5
            
            # If the decayed poignancy falls below 0.1, it's considered background "noise" and ignored
            if final_poignancy < 0.1:
                continue

            if final_poignancy >= 0.15: # Only log non-trivial poignancy scores above idle to reduce console noise
                 log_agent(state.name, f"Event '{desc}' poignancy rated as {final_poignancy:.2f} (Base: {base_poignancy:.2f})", "DEBUG")
            
            processed_event = PerceivedEvent(
                event_type=type_,
                poignancy=final_poignancy,
                depth=1,
                description=desc,
                entity_id=event.entity_id,
                created=getattr(event, 'created', None) or global_state.time.time,
                expiration=getattr(event, 'expiration', None),
                tile=event.tile
            )
            processed_events.append(processed_event)
            
        # Update caches (events that weren't seen this tick will be naturally dropped from the tracker)
        state.working_memory.repeated_event_fade = new_fade_tracker
        state.working_memory.last_observations_cache = current_observations
        return processed_events

    def _rate_perception_poignancy(self, agent_name: str, identity: str, event_type: EventType, description: str) -> float:
        """Heuristic scoring — zero LLM calls."""
        if "idle" in description:
            return 0.1
        return heuristic_poignance(event_type.value, description)
