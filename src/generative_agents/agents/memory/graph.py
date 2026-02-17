from typing import List, Tuple, Optional
from generative_agents.persistence import database

class GraphMemory:
    def __init__(self, agent_name: str):
        self.agent_name = agent_name

    def link_events(self, source_event_id: str, target_event_id: str, relation: str = "related"):
        """
        Creates a directed edge between two events.
        """
        database.add_event_link(source_event_id, target_event_id, relation)

    def get_related(self, event_id: str, relation: Optional[str] = None) -> List[str]:
        """
        Returns a list of event IDs related to the given event.
        """
        relations = database.get_related_events(event_id, relation)
        return [target_id for target_id, _ in relations]

    def get_causal_chain(self, event_id: str, depth: int = 1) -> List[str]:
        """
        Retrieves events that were caused by this event (using 'caused_by' or 'caused' relation).
        This is a simplified example.
        """
        # Assuming 'caused' relation: source caused target
        chain = []
        current_layer = [event_id]
        
        for _ in range(depth):
            next_layer = []
            for current_id in current_layer:
                children = self.get_related(current_id, "caused")
                next_layer.extend(children)
            
            if not next_layer:
                break
            
            chain.extend(next_layer)
            current_layer = next_layer
            
        return chain
