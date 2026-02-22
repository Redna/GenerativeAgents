import datetime
import dspy
from typing import List, Optional, Callable

from generative_agents.common.neural_types import AgentState, ActionSignal
from generative_agents.common.events import EventType, PerceivedEvent
from generative_agents.common.logging import log_agent
from generative_agents.intelligence.modules.perception import heuristic_poignance
from generative_agents.intelligence.modules.reflection import (
    ReflectionPointGenerator, InsightGenerator, IdentityFormulator
)

class MemoryConsolidator(dspy.Module):
    def __init__(self):
        super().__init__()

        self.reflection_generator = ReflectionPointGenerator()
        self.insight_generator = InsightGenerator()
        self.identity_formulator = IdentityFormulator()
        
    def forward(self, state: AgentState, retrieve_fn: Optional[Callable] = None) -> ActionSignal:
        """
        Refactoring of Reflection._run_reflect.
        Analyze recent memories and generate high-level insights.
        """
        signal = ActionSignal()
        
        # 1. Check if reflection is needed (Trigger)
        # Note: In legacy code, this was state based (counter). 
        # Here we assume the caller checks the trigger, or we check it here if passed in state.
        # For simplicity, we assume this is called when reflection is DESIRED.
        
        log_agent(state.name, "System 2: Starting Memory Consolidation (Reflection)", "INFO")
        
        # 2. Generate Focal Points (What to think about?)
        # We use recent memories provided in the state for focal point generation.
        
        if not state.recent_events or len(state.recent_events) < 5:
            return signal

        # Convert events to string for reflection
        memory_str = "\n".join([e.description for e in state.recent_events])
        focal_points = self.reflection_generator(memory_str, 3)
        log_agent(state.name, f"Generated focal points: {focal_points}", "DEBUG")
        
        # 3. Retrieve Nodes for Focal Points
        relevant_nodes = []
        if retrieve_fn is not None:
            seen_desc = set()
            for point in focal_points:
                retrieved = retrieve_fn(point, limit=10)
                for entry in retrieved:
                    event = PerceivedEvent.from_db_entry(entry)
                    if event.description not in seen_desc:
                        seen_desc.add(event.description)
                        relevant_nodes.append(event)
        else:
            # Fallback for isolated testing without memory access
            relevant_nodes = state.recent_events
        
        # 4. Generate Insights
        statements = [e.description for e in relevant_nodes]
        insights_data = self.insight_generator(statements, 3)
        
        if not isinstance(insights_data, list):
            insights_data = [insights_data] if insights_data else []
        
        # 5. Create Thoughts from Insights
        for thought in insights_data:
            if not isinstance(thought, str) or not thought.strip():
                continue
            # Create a Thought Event
            
            expiration = state.time.time + datetime.timedelta(days=30)
            
            # Rate poignance
            poignancy = heuristic_poignance(EventType.THOUGHT.value, thought)
            
            thought_event = PerceivedEvent(
                event_type=EventType.THOUGHT,
                poignancy=poignancy, 
                depth=1,
                description=thought,
                entity_id=state.name,
                created=state.time.time,
                expiration=expiration,
            )
            
            signal.new_memories.append(thought_event)
            log_agent(state.name, f"Consolidated Insight: {thought}", "INFO")

        # 6. Update Identity (System 2 Self-Concept Modification)
        # We re-evaluate who we are based on recent thoughts and actions
        
        commonset = ""
        commonset += f"Name: {state.name}\n"
        commonset += f"Age: {state.working_memory.age}\n"
        commonset += f"Innate traits: {state.innate_traits}\n"
        commonset += f"Current Role/Lifestyle: {state.identity_description}\n" # approximate
        commonset += f"Daily Requirement: {state.daily_plan_requirements}\n"
        
        if state.current_action:
             commonset += f"Currently: {state.current_action.event.description}\n"
        commonset += f"Current Date: {state.time.today}\n"
        
        # Add generated insights to the identity context
        if isinstance(insights_data, dict):
            commonset += f"Recent Insights: {list(insights_data.keys())}\n"

        new_identity = self.identity_formulator(state.name, commonset)
        signal.updated_identity = new_identity
        log_agent(state.name, f"Identity Updated: {new_identity[:50]}...", "INFO")

        return signal
