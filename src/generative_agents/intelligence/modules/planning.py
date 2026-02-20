import datetime
import dspy
from pydantic import BaseModel, Field
from typing import List, Dict, Tuple, Union, Optional

from generative_agents.common.utils import time_string_to_time
from generative_agents.common.neural_types import AgentState, ActionSignal

# --- Signatures ---

class WakeUpHourSignature(dspy.Signature):
    """Estimate the wake up hour of an agent based on their identity and lifestyle."""
    agent_name: str = dspy.InputField(desc="Name of the agent.")
    agent_identity: str = dspy.InputField(desc="Identity and backstory of the agent.")
    agent_lifestyle: str = dspy.InputField(desc="Lifestyle details of the agent.")
    rationale: str = dspy.OutputField(desc="Maximum two sentences reason for the wake up hour.")
    wake_up_hour: int = dspy.OutputField(desc="Wake up hour in 24-hour format (0-23).")

class RememberSignature(dspy.Signature):
    """Identify the most important things the agent should remember to create their daily plan."""
    name: str = dspy.InputField(desc="Name of the agent.")
    identity: str = dspy.InputField(desc="Identity of the agent.")
    statements: str = dspy.InputField(desc="Relevant statements/memories.")
    today: str = dspy.InputField(desc="Current date.")
    things_i_should_remember: str = dspy.OutputField(desc="Brief overview of things to remember for planning.")

class FeelingsSignature(dspy.Signature):
    """Summarize the agent's feelings about their days up to now."""
    name: str = dspy.InputField(desc="Name of the agent.")
    identity: str = dspy.InputField(desc="Identity of the agent.")
    statements: str = dspy.InputField(desc="Relevant statements/memories.")
    feelings: str = dspy.OutputField(desc="Brief summary of feelings.")

class ReflectionsSignature(dspy.Signature):
    """Reflect on yesterday to determine status for today."""
    name: str = dspy.InputField(desc="Name of the agent.")
    identity: str = dspy.InputField(desc="Identity of the agent.")
    yesterday: str = dspy.InputField(desc="Yesterday's date.")
    today: str = dspy.InputField(desc="Today's date.")
    current_activity: str = dspy.InputField(desc="Status from yesterday.")
    thought_note: str = dspy.InputField(desc="Thoughts at end of yesterday.")
    plan_note: str = dspy.InputField(desc="Plan note from yesterday.")
    reflection_of_yestereday: str = dspy.OutputField(desc="Brief reflection of yesterday's thoughts in third-person.")

class PlanOutline(BaseModel):
    hour: int = Field(description="The hour of the day (0-23).", ge=0, le=23)
    description: str = Field(description="Brief description of activity.")

class DailyPlanSignature(dspy.Signature):
    """Create a broad-strokes daily plan based on feelings and identity."""
    name: str = dspy.InputField(desc="Name of the agent.")
    identity: str = dspy.InputField(desc="Identity of the agent.")
    today: str = dspy.InputField(desc="Today's date.")
    feelings_for_today: str = dspy.InputField(desc="Reflected feelings/status for today.")
    wake_up_hour: Optional[str] = dspy.InputField(desc="Wake up time.", default=None) # Included from FirstDailyPlan logic
    plan_in_broad_strokes: list[PlanOutline] = dspy.OutputField(desc="List of broad daily activities with hours.")

class HourlyScheduleItem(BaseModel):
    time: str = Field(description="The time of the activity (e.g. 09:00 AM).")
    activity: str = Field(description="Brief activity description.")

class HourlyScheduleSignature(dspy.Signature):
    """Create an hourly schedule for the agent based on their daily plan."""
    name: str = dspy.InputField(desc="Name of the agent.")
    identity: str = dspy.InputField(desc="Identity of the agent.")
    daily_plan_summary: str = dspy.InputField(desc="Summary of the daily plan.")
    wake_up_hour: str = dspy.InputField(desc="The time the agent wakes up.")
    schedule: list[HourlyScheduleItem] = dspy.OutputField(desc="List of hourly activities starting from wake up time.")

class SubtaskItem(BaseModel):
    activity_name: str = Field(description="Description of the 5-minute subtask.")
    duration_minutes: int = Field(description="Duration in minutes.", default=5)

class DecompositionSignature(dspy.Signature):
    """Decompose a task into subtasks for the given duration."""
    name: str = dspy.InputField(desc="Name of the agent.")
    identity: str = dspy.InputField(desc="Identity of the agent.")
    today: str = dspy.InputField(desc="Today's date.")
    task_description: str = dspy.InputField(desc="The task to decompose.")
    task_context: str = dspy.InputField(desc="Context for the task.")
    time_range: str = dspy.InputField(desc="Start and end time.")
    total_duration: int = dspy.InputField(desc="Total duration in minutes.")
    subtasks: list[SubtaskItem] = dspy.OutputField(desc="List of subtasks. The sum of their durations should equal total_duration.")

# --- Unified Day Plan (Phase 6: single call) ---

class UnifiedDayScheduleItem(BaseModel):
    hour: int = Field(description="Hour of day (0-23).", ge=0, le=23)
    activity: str = Field(description="Brief activity description.")

class UnifiedDayPlan(BaseModel):
    wake_up_hour: int = Field(description="The hour the agent wakes up (0-23).", ge=0, le=23)
    plan_narrative: str = Field(description="One-sentence summary of the day's goals and mood.")
    hourly_slots: list[UnifiedDayScheduleItem] = Field(
        description="Hourly activities for the full day, starting from wake_up_hour."
    )

class UnifiedDayPlanSignature(dspy.Signature):
    """Given the agent's identity, relevant memories, and today's date, produce the complete day plan in one structured response. Include wake‑up hour, a mood narrative, and an hourly schedule from wake-up to sleep."""
    name: str = dspy.InputField(desc="Agent name.")
    identity: str = dspy.InputField(desc="Agent identity and backstory.")
    today: str = dspy.InputField(desc="Today's date.")
    yesterday_summary: str = dspy.InputField(desc="Brief summary of yesterday's events and status.", default="")
    relevant_memories: str = dspy.InputField(desc="Key memories relevant for today's planning.", default="")
    day_plan: UnifiedDayPlan = dspy.OutputField(desc="Complete structured day plan.")


class UnifiedDayPlanner(dspy.Module):
    """
    Phase 6: Replaces WakeUpHourPredictor + DailyPlanGenerator + HourlyScheduler.
    One ChainOfThought call → full typed day plan.
    """
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(UnifiedDayPlanSignature)

    def forward(
        self,
        state: AgentState,
        yesterday_summary: str = "",
        relevant_memories: str = "",
    ) -> Tuple[str, List[Tuple[str, int]]]:
        """
        Returns (plan_narrative, hourly_schedule).
        hourly_schedule: list of (activity, duration_minutes) tuples.
        """
        try:
            result = self.predict(
                name=state.name,
                identity=state.identity_description,
                today=str(state.time.today),
                yesterday_summary=yesterday_summary,
                relevant_memories=relevant_memories,
            )
            plan = result.day_plan

            # Build 24-slot schedule; fill pre-wakeup with "Sleeping"
            hourly_schedule: List[Tuple[str, int]] = []
            slot_map: Dict[int, str] = {item.hour: item.activity for item in plan.hourly_slots}
            for h in range(24):
                if h < plan.wake_up_hour:
                    hourly_schedule.append(("Sleeping", 60))
                else:
                    hourly_schedule.append((slot_map.get(h, "Idle"), 60))

            return plan.plan_narrative, hourly_schedule
        except Exception:
            # Graceful fallback: basic sleeping → working day
            fallback = [("Sleeping", 60)] * 7 + [("Morning routine", 60), ("Work", 60) * 8] + [("Evening routine", 60)] * 3 + [("Sleeping", 60)] * 5  # noqa: E501
            return "", [("Sleeping", 60)] * 7 + [("Morning routine", 60)] + [("Work", 60)] * 8 + [("Evening", 60)] * 3 + [("Sleeping", 60)] * 5


# --- Legacy Modules (kept for backward compatibility / fallback) ---

class WakeUpHourPredictor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(WakeUpHourSignature)

    def forward(self, agent_name: str, agent_identity: str, agent_lifestyle: str) -> str:
        try:
            response = self.predict(
                agent_name=agent_name,
                agent_identity=agent_identity,
                agent_lifestyle=agent_lifestyle
            )
            hour = response.wake_up_hour
            # Format output
            formatted_hour = str(hour).zfill(2) + ":00 " + ("AM" if hour < 12 else "PM")
            return formatted_hour
        except Exception as e:
            # print(f"Error in WakeUpHourPredictor: {e}")
            return "07:00 AM"

class DailyPlanGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.remember = dspy.Predict(RememberSignature)
        self.feelings = dspy.Predict(FeelingsSignature)
        self.reflection = dspy.Predict(ReflectionsSignature)
        self.plan = dspy.Predict(DailyPlanSignature) # Used for both First and New Day

    def forward(self, 
                name: str, 
                identity: str, 
                today: str, 
                wake_up_hour: str = None,
                yesterday: str = None,
                statements: str = None,
                current_activity: str = None,
                thought_note: str = None,
                plan_note: str = None,
                is_first_day: bool = False) -> Tuple[str, str]:
        """
        Generates daily plan. 
        Returns (plan_string_summary, status_update_string).
        """
        feelings_for_today = ""
        
        if not is_first_day and yesterday:
            # Full context generation
            remember = self.remember(name=name, identity=identity, statements=statements, today=today).things_i_should_remember
            feelings_summary = self.feelings(name=name, identity=identity, statements=statements).feelings
            feelings_for_today = self.reflection(
                name=name, identity=identity, yesterday=yesterday, today=today,
                current_activity=current_activity, thought_note=thought_note, plan_note=plan_note
            ).reflection_of_yestereday
        
        # Generate Plans
        try:
            response = self.plan(
                name=name, identity=identity, today=today, 
                feelings_for_today=feelings_for_today,
                wake_up_hour=wake_up_hour
            )
            
            converted_plan = {}
            if hasattr(response, "plan_in_broad_strokes"):
                 for activity in response.plan_in_broad_strokes:
                    if isinstance(activity, dict):
                        h = activity.get("hour", 12)
                        d = activity.get("description", "")
                    else:
                        h = activity.hour
                        d = activity.description
                    
                    orig_h = h
                    h = h % 12
                    meridien = "AM" if orig_h < 12 and orig_h != 24 else "PM"
                    if h == 0: h = 12
                    time_str = f"{str(h).zfill(2)}:00 {meridien}"
                    converted_plan[time_str] = d

            # Format Plan Output String
            plan_str = " ".join([f"{i+1}) {act} at {time}" for i, (time, act) in enumerate(converted_plan.items())])
            return plan_str, feelings_for_today

        except Exception as e:
            # print(f"Error in DailyPlanGenerator: {e}")
            return "", ""

class HourlyScheduler(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(HourlyScheduleSignature)

    def forward(self, name: str, identity: str, daily_plan_summary: str, wake_up_hour: str) -> List[Dict[str, str]]:
        try:
            response = self.predict(
                name=name,
                identity=identity,
                daily_plan_summary=daily_plan_summary,
                wake_up_hour=wake_up_hour
            )
            
            # Simplified output parsing (assuming robustness of response or using raw if possible)
            # Recreating the logic from hourly_breakdown.py would go here
            # For brevity in this refactor, returning the structured list directly if possible
            # But the consumer expects a full 24h list.
            
            hours = ["12:00 AM", "01:00 AM", "02:00 AM", "03:00 AM", "04:00 AM", "05:00 AM", "06:00 AM", "07:00 AM", 
                     "08:00 AM", "09:00 AM", "10:00 AM", "11:00 AM", "12:00 PM", "01:00 PM", "02:00 PM", "03:00 PM", 
                     "04:00 PM", "05:00 PM", "06:00 PM", "07:00 PM", "08:00 PM", "09:00 PM", "10:00 PM", "11:00 PM"]
            
            generated_map = {}
            if hasattr(response, "schedule"):
                for item in response.schedule:
                     if isinstance(item, dict):
                        generated_map[item.get("time")] = item.get("activity")
                     else:
                        generated_map[item.time] = item.activity
            
            full_schedule = []
            wake_up_dt = time_string_to_time(wake_up_hour)
            woke_up = False
            
            for hour in hours:
                hour_dt = time_string_to_time(hour)
                if hour == wake_up_hour.zfill(8) or (not woke_up and hour_dt.hour == wake_up_dt.hour):
                    woke_up = True
                    activity = generated_map.get(hour, "Wake up and get ready")
                elif not woke_up:
                    activity = "Sleeping"
                else:
                    activity = generated_map.get(hour, "Idle")
                
                full_schedule.append({"time": hour, "activity": activity})
            
            return full_schedule
        except Exception as e:
            # print(f"Error in HourlyScheduler: {e}")
             return [{"time": "08:00 AM", "activity": "Wake up"}]

class TaskDecomposer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(DecompositionSignature)

    def forward(self, name: str, identity: str, task_desc: str, start: str, end: str, duration: int, today: str, context: str) -> List[Tuple[str, int]]:
        try:
            response = self.predict(
                name=name, identity=identity, today=today,
                task_description=task_desc, task_context=context,
                time_range=f"{start} ~ {end}", total_duration=duration
            )
            
            result = []
            current_dur = 0
            if hasattr(response, "subtasks"):
                 for item in response.subtasks:
                    if isinstance(item, dict):
                        act = item.get("activity_name", "")
                        dur = item.get("duration_minutes", 5)
                    else:
                        act = item.activity_name
                        dur = item.duration_minutes
                    
                    result.append((act, dur))
                    current_dur += dur
            
            if current_dur < duration:
                result.append((task_desc, duration - current_dur))
            
            return result
        except Exception:
            return [(task_desc, duration)]
