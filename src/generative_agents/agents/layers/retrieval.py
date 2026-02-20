"""
agents/layers/retrieval.py
Phase 6: AssociativeMemoryLayer — Qdrant broad retrieval → graph expansion → ContextualReranker.
Total LLM calls: 1 (batched rerank) instead of N per-event poignancy calls.
"""
import dspy
from typing import List, Callable, Optional, Any

from generative_agents.common.neural_types import AgentState
from generative_agents.common.events import PerceivedEvent
from generative_agents.persistence.database import MemoryEntry

from generative_agents.intelligence.modules.retrieval import ContextualReranker


class AssociativeMemoryLayer(dspy.Module):
    """
    Phase 6: Three-stage retrieval pipeline:
      1. Qdrant semantic search (broad, 20 candidates)  — 0 LLM calls
      2. Graph neighbor expansion (depth=1)              — 0 LLM calls
      3. ContextualReranker (scores candidates jointly)  — 1 LLM call
    """

    BROAD_LIMIT = 20   # candidates from Qdrant before reranking
    TOP_N = 10         # candidates returned to Brain after reranking

    def __init__(self):
        super().__init__()
        self.reranker = ContextualReranker()

    def forward(
        self,
        state: AgentState,
        recent_events: List[PerceivedEvent],
        retrieve_fn: Callable[[str, int], List[MemoryEntry]],
        expand_fn: Optional[Callable[[str, int], List[MemoryEntry]]] = None,
    ) -> List[MemoryEntry]:
        """
        retrieve_fn: (query, limit) -> List[MemoryEntry]   (Qdrant search via MemorySystem)
        expand_fn:   (node_id, depth) -> List[MemoryEntry] (graph neighbors via MemorySystem)
        """
        # --- Stage 1: Build a rich query from current state ---
        query_parts: List[str] = []
        if state.current_action:
            query_parts.append(state.current_action.event.description)
        salient = [e for e in recent_events if e.poignancy > 0.4]
        for e in salient[:2]:
            query_parts.append(e.description)
        if not query_parts:
            query_parts.append(state.identity_description)
        query = " ".join(query_parts)

        # --- Stage 2: Broad Qdrant retrieval (fast, no LLM) ---
        candidates: List[MemoryEntry] = retrieve_fn(query, self.BROAD_LIMIT)

        # --- Stage 3: Graph expansion (walk related_events edges depth=1) ---
        if expand_fn:
            expanded_ids = {m.id for m in candidates}
            graph_neighbors: List[MemoryEntry] = []
            for mem in list(candidates):  # iterate over snapshot to avoid mutation
                neighbors = expand_fn(mem.id, depth=1)
                for n in neighbors:
                    if n.id not in expanded_ids:
                        expanded_ids.add(n.id)
                        graph_neighbors.append(n)
            candidates = candidates + graph_neighbors

        if not candidates:
            return []

        # --- Stage 4: ContextualReranker (1 batched LLM call) ---
        current_situation = (
            state.current_action.event.description if state.current_action else "idle"
        )
        ranked = self.reranker(
            agent_name=state.name,
            current_situation=current_situation,
            current_plan=state.daily_plan_requirements or "no plan",
            candidates=candidates,
            top_n=self.TOP_N,
        )
        return ranked
