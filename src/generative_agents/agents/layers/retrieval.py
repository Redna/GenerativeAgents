import dspy
from typing import List, Callable
from generative_agents.common.neural_types import AgentState
from generative_agents.common.events import PerceivedEvent
from generative_agents.persistence.database import MemoryEntry

class AssociativeMemoryLayer(dspy.Module):
    """
    Layer 2: Retrieval & Attention.
    Fetches relevant memories based on the agent's current state and incoming percepts.
    """
    def __init__(self):
        super().__init__()

    def forward(self, state: AgentState, recent_events: List[PerceivedEvent], retrieve_fn: Callable[[str, int], List[MemoryEntry]]) -> List[MemoryEntry]:
        """
        Retrieves context using the provided retrieval function (typically from MemorySystem).
        """
        # Build a query based on the most salient recent events, or default to current action/identity
        query_parts = []
        
        # Add current action if any
        if state.current_action:
            query_parts.append(state.current_action.event.description)
            
        # Add high-poignancy recent events
        salient_events = [e for e in recent_events if e.poignancy > 0.4]
        for e in salient_events[:2]: # Take top 2
            query_parts.append(e.description)
            
        # Fallback to general identity if nothing specific is happening
        if not query_parts:
            query_parts.append(state.identity_description)
            
        query = " ".join(query_parts)
        
        # Retrieve context (limit 10)
        retrieved_nodes = retrieve_fn(query, limit=10)
        return retrieved_nodes
