
import datetime
from typing import Annotated, TypedDict
from pydantic import BaseModel, Field, create_model

from generative_agents.utils import get_time_string, time_string_to_time

from langchain_groq.chat_models import ChatGroq
from langchain_core.messages import HumanMessage

llm = ChatGroq(model="llama3-8b-8192", name="task_decomposition")


template = """You act as {name}. You will decompose a task into subtasks.

{identity}

Today is {today}. {task_context}
In 5 minutes increments, what are the subtasks that {name} does when {name} is "{task_description}" from {task_start_time} ~ {task_end_time}? (total duration in minutes: {task_duration})"""

def create_decomposition_schedule(name: str, identity: str, task_description: str, task_start_time: str, task_end_time: str, task_duration: int, today: str, task_context: str) -> list[dict[str, str]]:

    class TaskDuration(TypedDict):
        duration: Annotated[int, 5, "The duration of the task in minutes."]
        task: Annotated[str, ..., "The acitivty within the timeframe"]

    class TaskDecompositionSchedule(TypedDict):
        tasks: Annotated[list[TaskDuration], ..., f"A list wich must consist of {task_duration // 5} tasks in 5 minutes increments during the {task_duration}"]

    structured_llm = llm.with_structured_output(TaskDecompositionSchedule)

    content = template.format(name=name, identity=identity, task_description=task_description, task_start_time=task_start_time, task_end_time=task_end_time, task_duration=task_duration, today=today, task_context=task_context)

    schedule = structured_llm.invoke([HumanMessage(content=content)])

    subtasks = []

    total_duration = task_duration
    for task in schedule["tasks"]:
        subtasks.append(([task["task"]], task["duration"]))
        total_duration -= task["duration"]
        if total_duration <= 0:
            break

    if total_duration > 0:
        subtasks[-1] = (subtasks[-1][0], subtasks[-1][1] + total_duration)

    return subtasks

if __name__ == "__main__":
    print(create_decomposition_schedule(name="James Peterson",
                                        identity="James Peterson is a 45 year old software developer. He enjoys coding and is a coffee enthusiast. James loves reading sci-fi novels and playing chess in his free time. He is an early bird and enjoys the quiet mornings.",
                                        task_description="Coding a new feature for the project",
                                        task_start_time="9:00 AM",
                                        task_end_time="10:00 AM",
                                        task_duration=60,
                                        today="Monday",
                                        task_context="James is working from home today."))
    print(create_decomposition_schedule(name="Emily Clark",
                                        identity="Emily Clark is a 25 year old freelance graphic designer. She is passionate about digital art and loves to travel. Emily is also a foodie and enjoys exploring new cuisines.",
                                        task_description="Visiting a new art museum",
                                        task_start_time="1:00 PM",
                                        task_end_time="2:00 PM",
                                        task_duration=60,
                                        today="Saturday",
                                        task_context="Emily is going to the museum with her friends."))
    print(create_decomposition_schedule(name="Alex Johnson",
                                        identity="Alex Johnson is a 35 year old personal trainer. He is dedicated to fitness and well-being. Alex enjoys outdoor activities and often goes hiking on weekends. He is motivated by helping others achieve their fitness goals.",
                                        task_description="Leading a morning fitness class",
                                        task_start_time="8:00 AM",
                                        task_end_time="9:30 AM",
                                        task_duration=90,
                                        today="Wednesday",
                                        task_context="Alex is conducting a group fitness session at the local park."))
    print(create_decomposition_schedule(name="Maria Gonzales",
                                        identity="Maria Gonzales is a 32 year old architect. She is innovative and enjoys drawing sketches of her designs. Maria loves gardening and spends her evenings taking care of her plants.",
                                        task_description="Drafting a new building design",
                                        task_start_time="5:00 PM",
                                        task_end_time="6:00 PM",
                                        task_duration=60,
                                        today="Friday",
                                        task_context="Maria is working on a new project for a client."))


