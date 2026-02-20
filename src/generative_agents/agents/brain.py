import dspy
from typing import List, Optional, Callable

from generative_agents.common.neural_types import AgentState, ActionSignal
from generative_agents.common.percept import Percept
from generative_agents.common.logging import log_agent
from generative_agents.persistence.database import MemoryEntry

from generative_agents.agents.layers.perception import SensoryProcessingLayer
from generative_agents.agents.layers.retrieval import AssociativeMemoryLayer
from generative_agents.agents.layers.planning import PlanningLayer
from generative_agents.agents.layers.actor import ActorLayer


class AgentBrain(dspy.Module):
    """
    Phase 6: The forward pass now uses:
      - UnifiedDayPlanner  (1 call on new-day)
      - ReActActor         (1 call every tick)
      - ContextualReranker (1 batched call every tick, inside AssociativeMemoryLayer)
    Total: 2 LLM calls on new-day ticks, 1 LLM call on normal ticks (per agent).
    """

    def __init__(self):
        super().__init__()
        self.perception = SensoryProcessingLayer()
        self.retrieval = AssociativeMemoryLayer()
        self.planning = PlanningLayer()
        self.actor = ActorLayer()

    def forward(
        self,
        percept: Percept,
        state: AgentState,
        retrieve_fn: Callable[[str, int], List[MemoryEntry]] = None,
        expand_fn: Callable[[str, int], List[MemoryEntry]] = None,
    ) -> ActionSignal:
        """
        Forward pass of the Agent Brain.

        Args:
            percept:     Sensory input from the SimulationEngine.
            state:       Snapshot of agent memory & internal status.
            retrieve_fn: (query, limit) -> List[MemoryEntry]   — Qdrant search.
            expand_fn:   (node_id, depth) -> List[MemoryEntry] — Graph neighbor walk.
        """
        log_agent(state.name, "Brain Forward Pass Started", "DEBUG")

        # 1. Perception — filter & process raw percept (heuristic, no LLM)
        filtered_events = self.perception(percept, state)
        state.recent_events = filtered_events

        # 2. Retrieval — Qdrant broad search → graph expansion → ContextualReranker (1 LLM call)
        if retrieve_fn:
            retrieved_memories = self.retrieval(
                state, filtered_events, retrieve_fn, expand_fn
            )
            from generative_agents.common.events import PerceivedEvent
            context_events = [PerceivedEvent.from_db_entry(m) for m in retrieved_memories]
            state.recent_events.extend(context_events)

        # 3. Planning — unified day plan if it's a new day (1 LLM call, else 0)
        plan_signal = self.planning(state)

        # 4. Actor — single ReActActor tool-choice call (1 LLM call)
        action_signal = self.actor(state, plan_signal)

        # 5. Merge signals
        final_signal = self._merge_signals(plan_signal, action_signal)
        final_signal.new_memories.extend(filtered_events)

        return final_signal

    def _merge_signals(self, plan_signal: ActionSignal, action_signal: ActionSignal) -> ActionSignal:
        merged = ActionSignal()
        merged.updated_daily_plan = plan_signal.updated_daily_plan
        merged.updated_daily_schedule = plan_signal.updated_daily_schedule
        merged.next_action = action_signal.next_action
        merged.update_chat_buffer = action_signal.update_chat_buffer
        merged.stop_chatting = action_signal.stop_chatting
        merged.new_memories = plan_signal.new_memories + action_signal.new_memories
        merged.thought_trace = f"Planning: {plan_signal.thought_trace} | Actor: {action_signal.thought_trace}"
        return merged
