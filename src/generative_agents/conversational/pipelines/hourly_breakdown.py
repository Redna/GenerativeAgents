import dspy
from pydantic import BaseModel, Field
from generative_agents.utils import time_string_to_time

class HourlyScheduleItem(BaseModel):
    time: str = Field(description="The time of the activity (e.g. 09:00 AM).")
    activity: str = Field(description="Brief activity description.")

class HourlyScheduleSignature(dspy.Signature):
    """
    Create an hourly schedule for the agent based on their daily plan.
    """
    name: str = dspy.InputField(desc="Name of the agent.")
    identity: str = dspy.InputField(desc="Identity of the agent.")
    daily_plan_summary: str = dspy.InputField(desc="Summary of the daily plan.")
    wake_up_hour: str = dspy.InputField(desc="The time the agent wakes up.")
    
    schedule: list[HourlyScheduleItem] = dspy.OutputField(desc="List of hourly activities starting from wake up time.")

def create_hourly_schedule(name: str, identity: str, daily_plan: list[dict[str, str]], wake_up_hour: str) -> list[dict[str, str]]:
    hours = ["12:00 AM", "01:00 AM", "02:00 AM", "03:00 AM", "04:00 AM", "05:00 AM", "06:00 AM", "07:00 AM", "08:00 AM", "09:00 AM", "10:00 AM", "11:00 AM",
             "12:00 PM", "01:00 PM", "02:00 PM", "03:00 PM", "04:00 PM", "05:00 PM", "06:00 PM", "07:00 PM", "08:00 PM", "09:00 PM", "10:00 PM", "11:00 PM"]
    
    # Pre-format daily plan for the prompt
    plan_str = "\n".join([f"{i+1}.) {item['activity']} at {item['time']}" for i, item in enumerate(daily_plan)])
    
    try:
        # Using ChainOfThought instead of TypedPredictor
        predict = dspy.ChainOfThought(HourlyScheduleSignature)
        response = predict(
            name=name, 
            identity=identity, 
            daily_plan_summary=plan_str, 
            wake_up_hour=wake_up_hour
        )
        
        # Post-process to ensure full coverage and correct format
        # We start with the full list of hours
        full_schedule = []
        wake_up_dt = time_string_to_time(wake_up_hour)
        
        # Convert generated schedule to dict for easy lookup
        generated_map = {}
        if hasattr(response, 'schedule'):
            for item in response.schedule:
                if isinstance(item, dict):
                    generated_map[item.get('time')] = item.get('activity')
                else:
                    generated_map[item.time] = item.activity
        
        woke_up = False
        for hour in hours:
            hour_dt = time_string_to_time(hour)
            
            if hour == wake_up_hour.zfill(8) or (not woke_up and hour_dt.hour == wake_up_dt.hour): # simple check
                 woke_up = True
                 full_schedule.append({"time": hour, "activity": generated_map.get(hour, "Wake up and get ready for the day")})
            elif not woke_up:
                full_schedule.append({"time": hour, "activity": "Sleeping"})
            else:
                 # fill in from generated or default to previous activity or idle
                 activity = generated_map.get(hour, "Idle")
                 full_schedule.append({"time": hour, "activity": activity})
                 
        return full_schedule

    except Exception as e:
        print(f"Error in create_hourly_schedule: {e}")
        # Fallback: simple sleep/wake breakdown
        return [{"time": h, "activity": "Sleeping" if i < 7 else "Idle"} for i, h in enumerate(hours)]

if __name__ == "__main__":
    # Setup dummy for testing
    if not dspy.settings.lm:
         dspy.settings.configure(lm=dspy.DummyLM([{
             "schedule": [HourlyScheduleItem(time="08:00 AM", activity="Wake up")]
         }]))
         
    hourly_schedule = create_hourly_schedule(name="Emily Johnson", 
                                             identity="Emily Johnson...",
                                             daily_plan=[{"time": "08:00 AM", "activity": "breakfast"}],
                                             wake_up_hour="08:00 AM")
    print(hourly_schedule)
