import dspy
from typing import List, Optional

from generative_agents.common.neural_types import AgentState, ActionSignal
from generative_agents.common.percept import Percept
from generative_agents.common.logging import log_agent

from generative_agents.agents.layers.planning import PlanningLayer
from generative_agents.agents.layers.actor import ActorLayer

class AgentBrain(dspy.Module):
    def __init__(self):
        super().__init__()
        # self.perception = SensoryProcessingLayer() # TODO
        # self.retrieval = AssociativeMemoryLayer() # TODO
        self.planning = PlanningLayer()
        self.actor = ActorLayer()

    def forward(self, percept: Percept, state: AgentState) -> ActionSignal:
        """
        The Forward Pass of the Agent's Brain.
        Takes in Sensory Inputs (Percept) and Internal State (Memory/Context).
        Returns an Action Signal (What to do next).
        """
        log_agent(state.name, "Brain Forward Pass Started", "DEBUG")
        
        # 1. Perception Layer (Filter & Process)
        # filtered_percepts = self.perception(percept)
        
        # 2. Retrieval Layer (Attention)
        # retrieved_memories = self.retrieval(state, filtered_percepts)
        
        # 3. Planning Layer (Reasoning)
        # Generates long-term plan updates (e.g. daily schedule)
        plan_signal = self.planning(state)
        
        # 4. Actor Layer (Decision)
        # Decides immediate action, considering the plan updates
        action_signal = self.actor(state, plan_signal)
        
        # 5. Signal Combination
        # We need to merge the signals. Action signal takes precedence for immediate actions,
        # but we need to preserve plan updates and memories from the planning layer.
        final_signal = self._merge_signals(plan_signal, action_signal)
        
        return final_signal

    def _merge_signals(self, plan_signal: ActionSignal, action_signal: ActionSignal) -> ActionSignal:
        """
        Merges two action signals.
        """
        merged = ActionSignal()
        
        # Planning updates
        merged.updated_daily_plan = plan_signal.updated_daily_plan
        merged.updated_daily_schedule = plan_signal.updated_daily_schedule
        
        # Action updates
        merged.next_action = action_signal.next_action
        merged.update_chat_buffer = action_signal.update_chat_buffer
        merged.stop_chatting = action_signal.stop_chatting
        
        # Combine memories
        merged.new_memories = plan_signal.new_memories + action_signal.new_memories
        
        # Combine traces
        merged.thought_trace = f"Planning: {plan_signal.thought_trace} | Actor: {action_signal.thought_trace}"
        
        return merged
