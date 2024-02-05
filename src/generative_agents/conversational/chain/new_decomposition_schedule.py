import asyncio
import datetime
import json
from random import random
import re
from typing import Dict, List, Tuple

from pydantic import BaseModel
import yaml
from generative_agents import global_state
from generative_agents.conversational.chain.json_expert import JsonExpert
from generative_agents.conversational.llm import llm
from langchain.chains import LLMChain
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, ChatPromptTemplate


from generative_agents.core.whisper.whisper import whisper
from generative_agents.utils import hour_string_to_time


system = """You are {{agent}}. There was change in your schedule. Hence you will need to adjust it. 
Note: In this villiage neither cars, nor bikes exist. The only way to get around is by walking.

Output format: Output a valid yaml of the following format:
    ```yaml
    daily_schedule: {% for hour in range(3) %}
        - time: <time in 12-hour format>
          activity: <activity during that hour in one brief sentence>
          duration: <duration in minutes>
    {%- endfor %}
        # The last activity MUST end by {{end_hour}}
    ```"""

user_shot_1 = """Here was John Doe's originally planned schedule from 10:00 AM to 02:00 PM.
```yaml
daily_schedule: 
    - time: 10:00 AM
      activity: serving customers
      duration: 60
    - time: 11:00 AM
      activity: preparing lunch break
      duration: 15
    - time: 11:15 AM
      activity: getting all coworkers together
      duration: 5
    - time: 11:20 AM
      activity: lunch break
      duration: 40
    - time: 12:00 PM
      activity: meeting with the boss
      duration: 60
    - time: 1:00 PM
      activity: serving customers
      duration: 60
```
John Doe unexpectedly ended up at "10:00 AM" with "Conversation with Jon Snow on the white walkers." for "15" minutes. 
How does the John Doe's revised schedule from 10:00 AM to 02:00 PM look like? (the last activity MUST end by 02:00 PM)"""

ai_shot_1 = """Here's John Doe's revised schedule from 10:00 AM to 02:00 PM:
```yaml
daily_schedule: 
    - time: 10:00 AM
      activity: Conversation with Jon Snow on the white walkers
      duration: 15
    - time: 10:15 AM
      activity: serving customers
      duration: 45
    - time: 11:00 AM
      activity: preparing lunch break
      duration: 15
    - time: 11:15 AM
      activity: getting all coworkers together
      duration: 5
    - time: 11:20 AM
      activity: lunch break
      duration: 40
    - time: 12:00 PM
      activity: meeting with the boss
      duration: 60
    - time: 1:00 PM
      activity: serving customers
      duration: 60
```"""


user = """Here was {{agent}}'s originally planned schedule from {{start_hour}} to {{end_hour}}.
```yaml
daily_schedule: {% for hour, activity, duration in schedule %}
    - time: {{hour}}
      activity: {{activity}}
      duration: {{duration}}
{%- endfor %}
```
{{agent}} unexpectedly ended up at "{{start_hour}}" with "{{new_event}}" for "{{new_event_duration}}" minutes. 
How does the {{agent}}'s revised schedule from {{start_hour}} to {{end_hour}} look like? (the last activity MUST end by {{end_hour}})"""


class NewDecompositionSchedule(BaseModel):
    
    agent: str
    start_hour: int
    end_hour: int
    new_event: str
    new_event_duration: int
    schedule_slice: List[Tuple[str, int]]

    async def run(self): 
        chat_template = ChatPromptTemplate(messages=[
            SystemMessagePromptTemplate.from_template(system, template_format="jinja2"),
            HumanMessagePromptTemplate.from_template(user_shot_1),
            AIMessagePromptTemplate.from_template(ai_shot_1),
            HumanMessagePromptTemplate.from_template(user, template_format="jinja2")])
                
        _new_decomposition_schedule = LLMChain(prompt=chat_template, llm=llm, llm_kwargs={
                                                        "max_tokens":650,
                                                        "top_p": 0.94,
                                                        "temperature": 0.6}
                                                        , verbose=True)
                
            
        schedule = self._generate_schedule()
        start_hour_str = hour_string_to_time(self.start_hour).strftime("%I:%M %p")
        end_hour_str = hour_string_to_time(self.end_hour).strftime("%I:%M %p")

        i = 0
        while True:   

            completion = await _new_decomposition_schedule.ainvoke(input={"agent": self.agent,
                                                                        "start_hour": start_hour_str,
                                                                        "end_hour": end_hour_str,
                                                                        "schedule": schedule,
                                                                        "new_event": self.new_event,
                                                                        "new_event_duration": self.new_event_duration})

            pattern = r'```yaml(.*)```'
            match = re.search(pattern, completion["text"], re.DOTALL)
            if match:
                try:
                    output = yaml.safe_load(match.group(1))
                except Exception as error:
                    try:
                        output = await JsonExpert(wrong_json=match.group(1),
                                                    error_message=str(error)).run()
                    except Exception as e:
                        continue

                try: 
                    schedule = output[f"daily_schedule"]
                    sched = self._fix_schedule(schedule)
                    return [(entry["activity"], entry["duration"]) for entry in sched]
                except:
                    pass

            i += 1
            print("Retry {i}, new decomposition failed")
    
    def _generate_schedule(self) -> str:
        start_hour = hour_string_to_time(self.start_hour)
        for_time = start_hour

        schedule_with_hour = []
        for action, duration in self.schedule_slice:
            schedule_with_hour.append((for_time.strftime("%I:%M %p"), action, duration))
            for_time += datetime.timedelta(minutes=int(duration))

        return schedule_with_hour
    
    def _fix_schedule(self, schedule: List[Dict[str, str]]) -> List[Dict[str, str]]:
        end_hour = hour_string_to_time(self.end_hour)

        end_time = datetime.datetime.strptime(schedule[-1]["time"], "%I:%M %p")
        end_time += datetime.timedelta(minutes=int(schedule[-1]["duration"]))

        if end_time == end_hour:
            return schedule
        
        if end_time > end_hour:
            schedule[-1]["duration"] = schedule[-1]["duration"] - int((end_time - end_hour).total_seconds() / 60)
        else:
            schedule[-1]["duration"] = schedule[-1]["duration"] + int((end_time - end_hour).total_seconds() / 60)

        return schedule


async def __tests():
    t = [
        NewDecompositionSchedule(agent="Jane Doe", 
                         start_hour=9, end_hour=13, 
                         new_event="Team meeting about quarterly goals.", 
                         new_event_duration=30,
                         schedule_slice=[("email correspondence", 45), 
                                         ("project work", 60), 
                                         ("coffee break", 15),
                                         ("project work", 60),
                                         ("quick team catch-up", 20),
                                         ("administrative tasks", 40)
                         ]).run(),
        NewDecompositionSchedule(agent="Alex Smith", 
                         start_hour=8, end_hour=11, 
                         new_event="Training session on new software tools.", 
                         new_event_duration=45,
                         schedule_slice=[("checking emails", 30), 
                                         ("team briefing", 15), 
                                         ("work on current project", 60),
                                         ("coffee break", 15),
                                         ("preparation for training", 60)
                         ]).run(),
        NewDecompositionSchedule(agent="Michelle Lee", 
                         start_hour=13, end_hour=17, 
                         new_event="Client meeting to discuss project updates.", 
                         new_event_duration=25,
                         schedule_slice=[("project development", 60), 
                                         ("lunch break", 45), 
                                         ("internal review meeting", 30),
                                         ("preparing meeting materials", 60),
                                         ("end-of-day wrap-up", 45)
                         ]).run(),
        NewDecompositionSchedule(agent="Carlos Rodriguez", 
                         start_hour=12, end_hour=16, 
                         new_event="Workshop on effective communication skills.", 
                         new_event_duration=120,
                         schedule_slice=[("team stand-up meeting", 15), 
                                         ("project work", 60), 
                                         ("lunch break", 45),
                                         ("feedback session", 60),
                                         ("preparing for workshop", 60)
                         ]).run()
    ]

    return await asyncio.gather(*t)

if __name__ == "__main__":
    from pprint import pprint
    t = asyncio.run(__tests())
    pprint(t)