import datetime
import dspy
from typing import Optional

from generative_agents.common.neural_types import AgentState, ActionSignal
from generative_agents.common.events import EventType, PerceivedEvent
from generative_agents.common.logging import log_agent
from generative_agents.simulation.time import DayType

# Intelligence Functions (Existing DSPy chains)
from generative_agents.intelligence.modules.planning import (
    WakeUpHourPredictor, DailyPlanGenerator, HourlyScheduler
)

class PlanningLayer(dspy.Module):
    def __init__(self):
        super().__init__()
        # Register sub-modules for optimization
        self.wake_up_predictor = WakeUpHourPredictor()
        self.daily_planner = DailyPlanGenerator()
        self.hourly_scheduler = HourlyScheduler()
    
    def forward(self, state: AgentState) -> ActionSignal:
        """
        Checks if it's a new day and generates the daily schedule.
        """
        signal = ActionSignal()
        
        if state.daytype not in [DayType.NEW_DAY, DayType.FIRST_DAY]:
            return signal

        log_agent(state.name, f"Planning Layer: Starting planning for {state.daytype}", "INFO")

        # 1. Estimate Wake Up Hour
        wake_up_hour = self.wake_up_predictor(
            state.name,
            state.identity_description,
            "generative_agents.intelligence.lifestyle_placeholder" 
        )

        # 2. Create Daily Requirements
        daily_reqs = state.daily_plan_requirements
        
        # Determine strictness of planning based on day type
        is_first_day = (state.daytype == DayType.FIRST_DAY)
        
        # We reuse the DailyPlanGenerator which handles both first day (simple) and new day (contextual)
        # Note: We need additional context for New Day (yesterday, statements etc) which isn't fully in AgentState yet.
        # Passing what we have.
        
        generated_plan, _ = self.daily_planner(
            name=state.name,
            identity=state.identity_description,
            today=str(state.time.today),
            wake_up_hour=wake_up_hour,
            is_first_day=is_first_day
        )
        
        if generated_plan:
            daily_reqs = generated_plan

        # 3. Create Hourly Schedule
        daily_schedule = self.hourly_scheduler(
            state.name,
            state.identity_description,
            daily_reqs,
            wake_up_hour,
        )

        # 4. Create Plan Event (Memory)
        plan_description = f"This is {state.name}'s plan for {state.time.today}: {self._summarize_schedule(daily_schedule)}."
        
        plan_event = PerceivedEvent(
            event_type=EventType.PLAN,
            poignancy=0.5,
            depth=1,
            description=plan_description,
            subject=state.name,
            predicate="plan",
            object_=str(state.time.today),
            created=state.time.time,
            expiration=state.time.time + datetime.timedelta(days=30),
            tile=state.current_tile,
        )

        # Populate Signal
        signal.updated_daily_plan = daily_reqs
        signal.updated_daily_schedule = [(item["activity"], 60) for item in daily_schedule] # basic conv
        signal.new_memories.append(plan_event)
        
        return signal

    def _summarize_schedule(self, schedule):
        return ",".join([entry["activity"] for entry in schedule])
