import asyncio
import json
import re
from pydantic import BaseModel
import yaml
from generative_agents import global_state
from generative_agents.conversational.chain.json_expert import JsonExpert
from generative_agents.conversational.llm import llm
from langchain.chains import LLMChain
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, ChatPromptTemplate


system = """You are in a roleplay and act as an agent. You will be asked to decompose a task into subtasks.
Break down the task in subtasks 5 minute increments. At the end no time should be left. Include a hint on main task in all subtasks.

Output format: Output a valid json of the following format:
```yaml
task_duration: "<total duration in minutes>"
subtasks:
    - duration: "<duration in minutes>"
      remaining_minutes: "<remaining minutes after completing this subtask>"
      activity: "<activity in one brief sentence>"
    - duration: "<duration in minutes>"
      remaining_minutes: "<remaining minutes after completing this subtask>"
      activity: "<activity in one brief sentence>"
    # ...
    - duration: "<duration in minutes>"
      remaining_minutes: "<remaining minutes after completing this subtask>"
      activity: "<activity in one brief sentence>"
```"""

user_shot_1 = """Sarah Lee is a 30 year old artist. She is deeply passionate about painting and spends most of her day in her studio. She loves nature and often takes long walks to find inspiration. Sarah is a night owl and finds herself most creative during the late hours.

In minimum 5 minutes increments, what are the subtasks that Sarah Lee does when Sarah Lee is "Waking up and finishing morning routine" from 7:00 AM ~ 8:00 PM? (total duration in minutes: 60)"""

ai_shot_1 = """```yaml
task_duration: 60
subtasks:
    - duration: 5
      remaining_minutes: 55
      activity: Turning off the alarm and stretching in bed, hinting at the start of Sarah's day.
    - duration: 10
      remaining_minutes: 45
      activity: Getting out of bed and heading to the bathroom for a morning wash, reflecting Sarah's morning refresh routine.
    - duration: 10
      remaining_minutes: 35
      activity: Preparing and enjoying a light breakfast, incorporating Sarah's love for starting her day peacefully.
    - duration: 5
      remaining_minutes: 30
      activity: Cleaning up the breakfast dishes, keeping her space tidy and conducive to creativity.
    - duration: 15
      remaining_minutes: 15
      activity: Taking a short morning walk outside, seeking inspiration from nature for her art.
    - duration: 10
      remaining_minutes: 5
      activity: Setting up her studio for the day's work, organizing materials and tools needed for painting.
    - duration: 5
      remaining_minutes: 0
      activity: Sitting down to sketch initial ideas, transitioning into her creative workflow.
```"""

user = """
{{identity}}

In minimum 5 minutes increments, what are the subtasks that {{name}} does when {{name}} is "{{task_description}}" from {{task_start_time}} ~ {{task_end_time}}? (total duration in minutes: {{task_duration}})"""

class TaskDecomposition(BaseModel):

    name: str
    identity: str
    task_description: str
    task_start_time: str
    task_end_time: str
    task_duration: str

    async def run(self):
        chat_template = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(
                    system, template_format="jinja2"),
                HumanMessagePromptTemplate.from_template(user_shot_1),
                AIMessagePromptTemplate.from_template(ai_shot_1),
                HumanMessagePromptTemplate.from_template(user, template_format="jinja2")])

        _task_decomposition_chain = LLMChain(prompt=chat_template, llm=llm, llm_kwargs={
            "max_tokens": 1000,
            "top_p": 0.95,
            "temperature": 0.3}, verbose=True)
        
        i = 0
        while True:
            completion = await _task_decomposition_chain.ainvoke(input={"name": self.name,
                                                                          "identity": self.identity,
                                                                          "task_description": self.task_description,
                                                                          "task_start_time": self.task_start_time,
                                                                          "task_end_time": self.task_end_time,
                                                                          "task_duration": int(self.task_duration)})
        
            pattern = r'```yaml(.*)```'
            match = re.search(pattern, completion["text"], re.DOTALL)
            if match:
                try:
                    output = yaml.safe_load(match.group(1))
                except Exception as error:
                    try:
                        output = await JsonExpert(wrong_yaml=match.group(1),
                                                    error_message=str(error)).run()
                    except Exception as e:
                        continue

                subtasks = output["subtasks"]
                
                if subtasks[-1]["remaining_minutes"] != 0:
                    subtasks[-1]["duration"] = subtasks[-2]["remaining_minutes"]
                    subtasks[-1]["remaining_minutes"] = 0
                
                return [(task["activity"], task["duration"]) for task in subtasks]
            
            i += 1
            print(f"Retry {i}, task decomposition failed")



async def __tests():
    t = [
        TaskDecomposition(name="James Peterson",
                  identity="James Peterson is a 45 year old software developer. He enjoys coding and is a coffee enthusiast. James loves reading sci-fi novels and playing chess in his free time. He is an early bird and enjoys the quiet mornings.",
                  today="Wednesday 12th February 2024",
                  task_description="Coding a new feature for the project",
                  task_start_time="9:00 AM",
                  task_end_time="10:00 AM",
                  task_duration="60").run(),
        TaskDecomposition(name="Emily Clark",
                  identity="Emily Clark is a 25 year old freelance graphic designer. She is passionate about digital art and loves to travel. Emily is also a foodie and enjoys exploring new cuisines.",
                  today="Friday 14th February 2024",
                  task_description="Visiting a new art museum",
                  task_start_time="1:00 PM",
                  task_end_time="2:00 PM",
                  task_duration="60").run(),
        TaskDecomposition(name="Alex Johnson",
                  identity="Alex Johnson is a 35 year old personal trainer. He is dedicated to fitness and well-being. Alex enjoys outdoor activities and often goes hiking on weekends. He is motivated by helping others achieve their fitness goals.",
                  today="Sunday 16th February 2024",
                  task_description="Leading a morning fitness class",
                  task_start_time="8:00 AM",
                  task_end_time="9:30 AM",
                  task_duration="90").run(),
        TaskDecomposition(name="Maria Gonzales",
                  identity="Maria Gonzales is a 32 year old architect. She is innovative and enjoys drawing sketches of her designs. Maria loves gardening and spends her evenings taking care of her plants.",
                  today="Tuesday 18th February 2024",
                  task_description="Drafting a new building design",
                  task_start_time="5:00 PM",
                  task_end_time="6:00 PM",
                  task_duration="60").run(),

    ]

    return await asyncio.gather(*t)

if __name__ == "__main__":
    from pprint import pprint
    t = asyncio.run(__tests())
    pprint(t)