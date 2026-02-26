import datetime
import dspy
from typing import Optional

from generative_agents.common.neural_types import AgentState, ActionSignal
from generative_agents.common.events import EventType, PerceivedEvent
from generative_agents.common.logging import log_agent
from generative_agents.simulation.time import DayType

from generative_agents.intelligence.modules.planning import UnifiedDayPlanner, ReplanEvaluator
from generative_agents.persistence.database import get_repository, MemoryEntry


class PlanningLayer(dspy.Module):
    """
    Phase 6: Replaces the 3–5 sequential planning calls (WakeUp → Remember →
    Feelings → DailyPlan → Hourly) with a single UnifiedDayPlanner call.
    """

    def __init__(self):
        super().__init__()
        self.day_planner = UnifiedDayPlanner()

    def forward(self, state: AgentState, yesterday_summary: str = "", relevant_memories: str = "") -> ActionSignal:
        """
        Checks if it's a new day and generates the full daily schedule in one LLM call.
        """
        signal = ActionSignal()

        if state.daytype not in [DayType.NEW_DAY, DayType.FIRST_DAY]:
            return signal

        # --- Prevent Restart Duplication ---
        # If this is the FIRST_DAY (empty schedule on boot), check if we already generated a plan for this specific calendar day
        # querying the persistent Qdrant memory.
        if state.daytype == DayType.FIRST_DAY:
            repo = get_repository(state.name)
            # Find recent plans
            recent_plans = repo.retrieve(f"{state.name}'s plan for {state.time.today}", limit=5)
            # Filter to exact exact match for today to avoid picking up yesterday's plan if DB is stale
            for p in recent_plans:
                if p.get("memory_type") == EventType.PLAN.value and str(state.time.today) in p.get("content", ""):
                    log_agent(state.name, f"PlanningLayer: Found existing day plan for {state.time.today} in persistent memory. Skipping generation.", "INFO")
                    # We could parse the hourly schedule back out of `p.content`, but for simplicity 
                    # we will just provide a generic filler schedule for this resumed session since the true 
                    # state is meant to be handled by checkpoints, not raw ReAct logs.
                    # Or alternatively, the LLM will just continue from whatever activity it sees fit based on resumed time.
                    
                    # For a robust fix without complex regex parsing, we trigger generation but ONLY if we haven't today.
                    # Since we HAVE a plan today, we let the agent organically figure out what to do using ActorLayer.
                    
                    signal.updated_daily_plan = p.get("content", "")
                    # Generate a lightweight fallback schedule so the engine doesn't break
                    signal.updated_daily_schedule = [("Routine Activities", 60)] * 24
                    return signal

        log_agent(state.name, f"PlanningLayer: Generating unified day plan for {state.daytype}", "INFO")

        # Single LLM call → full day plan
        plan_narrative, hourly_schedule = self.day_planner(
            state=state,
            yesterday_summary=yesterday_summary,
            relevant_memories=relevant_memories,
        )

        # Build plan event for memory
        plan_description = (
            f"This is {state.name}'s plan for {state.time.today}: {plan_narrative or self._summarize(hourly_schedule)}."
        )

        plan_event = PerceivedEvent(
            event_type=EventType.PLAN,
            poignancy=0.5,
            depth=1,
            description=plan_description,
            entity_id=state.name,
            created=state.time.time,
            expiration=state.time.time + datetime.timedelta(days=30),
            tile=state.current_tile,
        )

        signal.updated_daily_plan = plan_narrative or plan_description
        signal.updated_daily_schedule = hourly_schedule
        signal.new_memories.append(plan_event)

        log_agent(
            state.name,
            f"PlanningLayer: Schedule generated ({len(hourly_schedule)} slots).",
            "INFO",
        )
        return signal

    def _summarize(self, schedule):
        activities = [act for act, _ in schedule if act not in ("Sleeping", "Idle")]
        return ", ".join(activities[:5]) if activities else "routine day"


class ReplanningLayer(dspy.Module):
    """
    Evaluates high-poignancy events mid-day to determine if the macro schedule must change.
    Outputs an ActionSignal with the updated daily schedule if a change occurred.
    """
    def __init__(self):
        super().__init__()
        self.evaluator = ReplanEvaluator()

    def forward(self, state: AgentState, interrupting_event: PerceivedEvent) -> ActionSignal:
        signal = ActionSignal()
        
        if not state.daily_schedule:
            return signal

        current_time_str = state.time.as_string()
        current_schedule_str = ", ".join([f"{act}" for act, _ in state.daily_schedule])

        log_agent(state.name, f"ReplanningLayer: Evaluating disruption from '{interrupting_event.description}'", "INFO")

        did_replan, new_schedule = self.evaluator(
            name=state.name,
            identity=state.identity_description,
            current_time=current_time_str,
            current_schedule=current_schedule_str,
            interrupting_event=interrupting_event.description
        )

        if did_replan and new_schedule:
            log_agent(state.name, "ReplanningLayer: Schedule dynamically adjusted!", "INFO")
            
            # Create a perceived memory of the replanning
            plan_description = f"{state.name} changed their plans for the rest of the day due to: {interrupting_event.description}"
            plan_event = PerceivedEvent(
                event_type=EventType.PLAN,
                poignancy=0.6,
                depth=1,
                description=plan_description,
                entity_id=state.name,
                created=state.time.time,
                expiration=state.time.time + datetime.timedelta(days=7),
                tile=state.current_tile,
            )

            signal.updated_daily_schedule = new_schedule
            signal.new_memories.append(plan_event)
            
        return signal
