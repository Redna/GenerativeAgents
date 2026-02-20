from typing import List, Optional, Union
from generative_agents.persistence import database
from generative_agents.persistence.database import MemoryEntry, MemoryType, ConversationFilling


class MemorySystem:
    """
    The Unified Memory System ("The Connectome").
    Pure Qdrant backend — vectors AND graph edges live in the same store.
    """

    def __init__(self, agent_name: str, retention: float = 1.0):
        self.agent_name = agent_name
        self.retention = retention

    def add(self, memory_entry: Union[MemoryEntry, "PerceivedEvent"]) -> MemoryEntry:
        """Adds a memory to Qdrant (vector + payload with graph edges)."""
        if hasattr(memory_entry, "to_db_entry"):
            memory_entry = memory_entry.to_db_entry()
        return database.add(self.agent_name, memory_entry)

    def retrieve(self, query: str, limit: int = 5) -> List[MemoryEntry]:
        """ANN vector search via Qdrant."""
        return database.get(self.agent_name, query, limit)

    def get_context(self, node_id: str, depth: int = 1) -> List[MemoryEntry]:
        """
        BFS graph traversal starting from node_id.
        Edges are read from the 'related_events' payload field in Qdrant.
        Returns the expanded neighborhood as MemoryEntry objects.
        """
        repo = database.get_repository(self.agent_name)

        context_ids: set[str] = {node_id}
        frontier: list[str] = [node_id]

        for _ in range(depth):
            next_frontier: list[str] = []
            for current_id in frontier:
                for target_id, _ in repo.get_related(current_id):
                    if target_id not in context_ids:
                        context_ids.add(target_id)
                        next_frontier.append(target_id)
            frontier = next_frontier

        results: List[MemoryEntry] = []
        for mid in context_ids:
            raw = repo.get_by_id(mid)
            if raw:
                try:
                    results.append(MemoryEntry(**raw))
                except Exception:
                    pass
        return results

    def add_relation(self, source_id: str, target_id: str, relation: str):
        """Creates a directed edge (stored in Qdrant payload)."""
        database.add_event_link(source_id, target_id, relation)

    # ------------------------------------------------------------------
    # Legacy helpers
    # ------------------------------------------------------------------

    def get_last_chat(self, with_agent_name: str) -> Optional[MemoryEntry]:
        return database.get_last_chat(self.agent_name, with_agent_name)

    def get_active_chat(self, with_agent_name: str) -> Optional[MemoryEntry]:
        return database.get_active_chat(self.agent_name, with_agent_name)
