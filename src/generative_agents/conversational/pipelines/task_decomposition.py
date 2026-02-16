import dspy
from pydantic import BaseModel, Field
import datetime
from generative_agents.utils import time_string_to_time

class SubtaskItem(BaseModel):
    activity_name: str = Field(description="Description of the 5-minute subtask.")
    duration_minutes: int = Field(description="Duration in minutes (usually 5 or 10).", default=5)

class DecompositionSignature(dspy.Signature):
    """
    Decompose a task into subtasks for the given duration.
    """
    name: str = dspy.InputField(desc="Name of the agent.")
    identity: str = dspy.InputField(desc="Identity of the agent.")
    today: str = dspy.InputField(desc="Today's date.")
    task_description: str = dspy.InputField(desc="The task to decompose.")
    task_context: str = dspy.InputField(desc="Context for the task.")
    time_range: str = dspy.InputField(desc="Start and end time.")
    total_duration: int = dspy.InputField(desc="Total duration in minutes.")
    
    subtasks: list[SubtaskItem] = dspy.OutputField(desc="List of subtasks. The sum of their durations should equal total_duration.", )


def create_decomposition_schedule(name: str, identity: str, task_description: str, task_start_time: str, task_end_time: str, task_duration: int, today: str, task_context: str) -> list[tuple[str, int]]:
    time_range = f"{task_start_time} ~ {task_end_time}"
    
    try:
        # Using ChainOfThought instead of TypedPredictor
        predict = dspy.ChainOfThought(DecompositionSignature)
        response = predict(
            name=name,
            identity=identity,
            today=today,
            task_description=task_description,
            task_context=task_context,
            time_range=time_range,
            total_duration=task_duration
        )
        
        # Format output as expected: list of (activity, duration)
        # Note: original code generated fixed slots. We'll trust LLM to generate enough 5-10 min chunks or normalize.
        result = []
        current_dur = 0
        if hasattr(response, 'subtasks'):
            for item in response.subtasks:
                if isinstance(item, dict):
                     act_name = item.get('activity_name', '')
                     dur_min = item.get('duration_minutes', 5)
                else:
                     act_name = item.activity_name
                     dur_min = item.duration_minutes
                
                result.append((act_name, dur_min))
                current_dur += dur_min
            
        # Fill remaining time if any
        if current_dur < task_duration:
            result.append((task_description, task_duration - current_dur))
            
        return result
    except Exception as e:
        print(f"Error in create_decomposition_schedule: {e}")
        return [(task_description, task_duration)]

if __name__ == "__main__":
    if not dspy.settings.lm:
         dspy.settings.configure(lm=dspy.DummyLM([{
             "subtasks": [SubtaskItem(activity_name="Start coding", duration_minutes=10)]
         }]))
         
    print(create_decomposition_schedule(name="James Peterson",
                                        identity="James...",
                                        task_description="Coding",
                                        task_start_time="9:00 AM",
                                        task_end_time="10:00 AM",
                                        task_duration=60,
                                        today="2023-10-01", 
                                        task_context="Working"))