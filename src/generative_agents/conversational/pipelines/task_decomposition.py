
import datetime
from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field, create_model

from generative_agents.conversational.pipelines.grammar_llm_pipeline import grammar_pipeline
from generative_agents.utils import get_time_string, hour_string_to_time, time_string_to_time


template = """You are in a roleplay and act as an agent. You will be asked to decompose a task into subtasks.
Break down the task in subtasks 5 minute increments. At the end no time should be left. Include a hint on main task in all subtasks.

{{identity}}

Today is {{today}}. {{task_context}}
In minimum 5 minutes increments, what are the subtasks that {{name}} does when {{name}} is "{{task_description}}" from {{task_start_time}} ~ {{task_end_time}}? (total duration in minutes: {{task_duration}})"""

def create_decomposition_schedule(name: str, identity: str, task_description: str, task_start_time: str, task_end_time: str, task_duration: int, today: str, task_context: str) -> list[dict[str, str]]:

    # iterate all 5 minutes increments
    subtasks = {}

    # startime in datetime format from xx:xx AM/PM
    start_time = time_string_to_time(task_start_time)

    for minutes in range(0, task_duration, 5):
        start_time = start_time + datetime.timedelta(minutes=minutes)

        Subtask = create_model(f"Subtask_{minutes//5}", time=(Enum("Time", [get_time_string(start_time)]), Field(description="The start time of the activity in 12-hour clock format (hh:mm AM/PM)")),
                               duration=(Enum("Duration", ["5"]), Field(description="5 minutes duration")),
                               activity=(str, Field(..., description="The activity planned for the time.")))

        subtasks[f"Subtask {minutes//5}/{task_duration // 5}"] = (Subtask, Field(..., description=f"Activity for subtask {minutes//5} of {task_description}"))

    DecompositionSchedule = create_model("DecompositionSchedule", **subtasks)

    schedule = grammar_pipeline.run(model=DecompositionSchedule, prompt_template=template, template_variables={
        "name": name,
        "identity": identity,
        "task_description": task_description,
        "task_start_time": task_start_time,
        "task_end_time": task_end_time,
        "task_duration": task_duration,
        "today": today,
        "task_context": task_context
    })

    subtasks = [task for task in schedule.model_dump().values()]
    return subtasks

if __name__ == "__main__":
    print(create_decomposition_schedule(name="James Peterson",
                                        identity="James Peterson is a 45 year old software developer. He enjoys coding and is a coffee enthusiast. James loves reading sci-fi novels and playing chess in his free time. He is an early bird and enjoys the quiet mornings.",
                                        task_description="Coding a new feature for the project",
                                        task_start_time="9:00 AM",
                                        task_end_time="10:00 AM",
                                        task_duration=60))
    print(create_decomposition_schedule(name="Emily Clark",
                                        identity="Emily Clark is a 25 year old freelance graphic designer. She is passionate about digital art and loves to travel. Emily is also a foodie and enjoys exploring new cuisines.",
                                        task_description="Visiting a new art museum",
                                        task_start_time="1:00 PM",
                                        task_end_time="2:00 PM",
                                        task_duration=60))
    print(create_decomposition_schedule(name="Alex Johnson",
                                        identity="Alex Johnson is a 35 year old personal trainer. He is dedicated to fitness and well-being. Alex enjoys outdoor activities and often goes hiking on weekends. He is motivated by helping others achieve their fitness goals.",
                                        task_description="Leading a morning fitness class",
                                        task_start_time="8:00 AM",
                                        task_end_time="9:30 AM",
                                        task_duration=90))
    print(create_decomposition_schedule(name="Maria Gonzales",
                                        identity="Maria Gonzales is a 32 year old architect. She is innovative and enjoys drawing sketches of her designs. Maria loves gardening and spends her evenings taking care of her plants.",
                                        task_description="Drafting a new building design",
                                        task_start_time="5:00 PM",
                                        task_end_time="6:00 PM",
                                        task_duration=60))
    
    
        