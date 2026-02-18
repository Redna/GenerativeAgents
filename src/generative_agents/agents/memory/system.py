from typing import List, Optional, Union, Tuple
from generative_agents.persistence import database
from generative_agents.persistence.database import MemoryEntry, MemoryType, ConversationFilling

class MemorySystem:
    """
    The Unified Memory System ("The Connectome").
    Combines Associative Memory (Vector Search) and Knowledge Graph (Relationships).
    """
    def __init__(self, agent_name: str, retention: float = 1.0):
        self.agent_name = agent_name
        self.retention = retention
        
    def add(self, memory_entry: Union[MemoryEntry, "PerceivedEvent"]) -> MemoryEntry:
        """
        Adds a memory entry to the system.
        Automatically handles vector embedding (via database layer) and node creation.
        """
        # Handle conversion if PerceivedEvent passed
        if hasattr(memory_entry, "to_db_entry"):
            memory_entry = memory_entry.to_db_entry()
            
        return database.add(self.agent_name, memory_entry)

    def retrieve(self, query: str, limit: int = 5) -> List[MemoryEntry]:
        """
        Semantic search for memories relevant to the query.
        Wrapper around Associative Memory retrieval.
        """
        return database.get(self.agent_name, query, limit)

    def get_context(self, node_id: str, depth: int = 1) -> List[MemoryEntry]:
        """
        Retrieves the context around a specific memory node (Graph Traversal).
        Returns the node itself and its neighbors up to 'depth'.
        """
        # Get the central node
        repo = database.get_repository(self.agent_name)
        root = repo.get_by_id(node_id)
        if not root:
             return []
             
        context_ids = {node_id}
        frontier = [node_id]
        
        for _ in range(depth):
            next_frontier = []
            for current_id in frontier:
                 # Get related events (outgoing edges)
                 related = database.get_related_events(current_id)
                 for target_id, relation in related:
                     if target_id not in context_ids:
                         context_ids.add(target_id)
                         next_frontier.append(target_id)
            frontier = next_frontier
            
        # Fetch actual memory objects
        results = []
        for mid in context_ids:
            mem = repo.get_by_id(mid)
            if mem:
                results.append(MemoryEntry(**mem))
                
        return results

    def add_relation(self, source_id: str, target_id: str, relation: str):
        """
        Creates a directed edge between two memory nodes.
        """
        database.add_event_link(source_id, target_id, relation)

    # Legacy wrappers to maintain compatibility or specific queries
    def get_last_chat(self, with_agent_name: str) -> Optional[MemoryEntry]:
        return database.get_last_chat(self.agent_name, with_agent_name)

    def get_active_chat(self, with_agent_name: str) -> Optional[MemoryEntry]:
        return database.get_active_chat(self.agent_name, with_agent_name)
