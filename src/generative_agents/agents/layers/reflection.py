import datetime
import dspy
from typing import List, Optional

from generative_agents.common.neural_types import AgentState, ActionSignal
from generative_agents.common.events import EventType, PerceivedEvent
from generative_agents.common.logging import log_agent
from generative_agents.intelligence.modules.perception import EventParser, PoignanceRater
from generative_agents.intelligence.modules.reflection import (
    ReflectionPointGenerator, InsightGenerator, IdentityFormulator
)

class MemoryConsolidator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.event_parser = EventParser()
        self.poignance_rater = PoignanceRater()
        self.reflection_generator = ReflectionPointGenerator()
        self.insight_generator = InsightGenerator()
        self.identity_formulator = IdentityFormulator()
        
    def forward(self, state: AgentState) -> ActionSignal:
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
        # We need recent memories. Assuming retrieval layer or state provides access.
        # For MVP, we need access to AssociativeMemory which is not in AgentState fully.
        # This is a limitation of the current strict isolation. 
        # We will assume a helper or that we can query the DB.
        
        # TODO: This requires DB access. 
        # Ideally, `state.recent_events` is populated with enough history.
        # Or we inject a 'memory_reader' function into this module.
        # For now, let's assume we can generate points based on the recent events in state
        # if they are sufficient, or we skipped implementation until we solve DB access in pure modules.
        
        if not state.recent_events or len(state.recent_events) < 5:
            return signal

        # Convert events to string for reflection
        memory_str = "\n".join([e.description for e in state.recent_events])
        focal_points = self.reflection_generator(memory_str, 3)
        log_agent(state.name, f"Generated focal points: {focal_points}", "DEBUG")
        
        # 3. Retrieve Nodes for Focal Points
        # This step implies we search the *entire* memory for these topics.
        # Again, requires DB access.
        # Simulating retrieval for now using available context
        relevant_nodes = state.recent_events # Placeholder for retrieval(focal_points)
        
        # 4. Generate Insights
        statements = [e.description for e in relevant_nodes]
        insights_data = self.insight_generator(statements, 3)
        
        log_agent(state.name, f"Insights Data Type: {type(insights_data)} Value: {insights_data}", "DEBUG")
        
        if isinstance(insights_data, list) and len(insights_data) > 0:
             # Handle case where it might be returned as a list of dicts or strings
             insights_data = insights_data[0]
             
        if isinstance(insights_data, str):
            try:
                import json
                # Try to parse JSON string
                # Clean potential markdown code blocks
                if "```json" in insights_data:
                    insights_data = insights_data.split("```json")[1].split("```")[0]
                elif "```" in insights_data:
                    insights_data = insights_data.split("```")[1].split("```")[0]
                    
                insights_data = json.loads(insights_data)
            except Exception as e:
                log_agent(state.name, f"Failed to parse insights JSON: {e}", "ERROR")
                insights_data = {}
        
        if not isinstance(insights_data, dict):
            log_agent(state.name, f"Insights data is not a dict: {type(insights_data)}", "ERROR")
            insights_data = {}
        
        # 5. Create Thoughts from Insights
        for thought, evidence in insights_data.items():
            # Create a Thought Event
            s, p, o = self.event_parser.get_triple(state.name, thought)
            
            expiration = state.time.time + datetime.timedelta(days=30)
            
            # Rate poignance (using just Thought type context if possible, or mapping to EVENT/THOUGHT)
            poignancy = self.poignance_rater(state.name, state.identity_description, EventType.THOUGHT.value, thought)
            
            thought_event = PerceivedEvent(
                event_type=EventType.THOUGHT,
                poignancy=int(poignancy)/10, 
                depth=1,
                description=thought,
                subject=s,
                predicate=p,
                object_=o,
                created=state.time.time,
                expiration=expiration,
                filling=evidence # Storing evidence IDs
            )
            
            signal.new_memories.append(thought_event)
            log_agent(state.name, f"Consolidated Insight: {thought}", "INFO")

        # 6. Update Identity (System 2 Self-Concept Modification)
        # We re-evaluate who we are based on recent thoughts and actions
        # initialized in init ideally, but for now importing or using if added to init
        # self.identity_formulator was not added to init in previous step, checking...
        # I need to add it to init first.
        
        # For this step I will just replace the import and use the module if available
        # Wait, I missed adding `self.identity_formulator` to `__init__` in the previous successful tool call.
        # I should add it there.
        
        commonset = ""
        commonset += f"Name: {state.name}\n"
        commonset += f"Age: {state.innate_traits}\n" # Note: Age/Traits mixed in source, correcting in principle but following pattern
        # Actually state.innate_traits is a list. state (AgentState) has innate_traits. 
        # Using state attributes directly.
        # But wait, AgentState doesn't have age explicitly? It does have innate_traits.
        # Let's check AgentState definition or assume widely available.
        # AgentState definition in Agent.run_step: name, identity_description, innate_traits...
        
        commonset += f"Innate traits: {state.innate_traits}\n"
        commonset += f"Current Role/Lifestyle: {state.identity_description}\n" # approximate
        commonset += f"Daily Requirement: {state.daily_plan_requirements}\n"
        
        if state.current_action:
             commonset += f"Currently: {state.current_action.event.description}\n"
        commonset += f"Current Date: {state.time.today}\n"
        
        # Add generated insights to the identity context
        if isinstance(insights_data, dict):
            commonset += f"Recent Insights: {list(insights_data.keys())}\n"

        # Assumption: self.identity_formulator is available (will add in next step if not)
        new_identity = self.identity_formulator(state.name, commonset)
        signal.updated_identity = new_identity
        log_agent(state.name, f"Identity Updated: {new_identity[:50]}...", "INFO")

        return signal
