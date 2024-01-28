import asyncio
import datetime
import json
from random import random
import re
from typing import Dict, List, Tuple

from pydantic import BaseModel
from generative_agents import global_state
from generative_agents.conversational.chain.json_expert import JsonExpert
from generative_agents.conversational.llm import llm
from langchain import LLMChain, PromptTemplate

from generative_agents.core.whisper.whisper import whisper

hours = hours = ["12:00 AM", "01:00 AM", "02:00 AM", "03:00 AM", "04:00 AM", "05:00 AM", "06:00 AM", "07:00 AM", "08:00 AM", "09:00 AM", "10:00 AM", "11:00 AM", "12:00 PM", "01:00 PM", "02:00 PM", "03:00 PM", "04:00 PM", "05:00 PM", "06:00 PM", "07:00 PM", "08:00 PM", "09:00 PM", "10:00 PM", "11:00 PM"]

def _create_schedule_template(activities, name=""):
    key = f"{name}'s schedule" if name else "schedule"
    schedule = {key: []}
    
    for hour, activity in zip(hours, activities):
        schedule_entry = {
            "time": hour,
            "activity": activity
        }
        schedule[key].append(schedule_entry)

    return json.dumps(schedule, indent=4)

_template = """<|system|>You follow the tasks given by the user as close as possible. You will only generate 1 valid JSON object as mentioned below.
You will act as {name}.

Your identity is: 
{identity}

Note: In this villiage neither cars, nor bikes exist. The only way to get around is by walking.
<|user|>
Output format: Output a valid json of the following format:
```
{schedule_format}
````

Here is today's plan in broad-strokes:
{daily_plan}

What is {name}'s hourly schedule for today? You must follow the schedule format above. Do not forget any time slots.
<|assistant|>
```
{prior_schedule}"""

class HourlyBreakdown(BaseModel):
    
    identity: str
    wake_up_hour: str
    name: str
    hourly_organized_activities: List[Dict[str, str]]

    async def run(self): 
        hourly_schedule_format_template = _create_schedule_template(["<fill in one brief sentence>" for i in range(24)], name=self.name)
        
        _prompt = PromptTemplate(input_variables=["schedule_format", "identity", "name", "daily_plan", "prior_schedule"],
                                template=_template)
        _hourly_breakdown_chain = LLMChain(prompt=_prompt, llm=llm, llm_kwargs={
                                                "max_new_tokens":650,
                                                "do_sample": True,
                                                "top_p": 0.96,
                                                "top_k": 20,
                                                "temperature": 0.4,
                                                "repetition_penalty": 1.03}
                                                , verbose=global_state.verbose)
        i = 0
        while True:   
            _hourly_breakdown_chain.llm_kwargs["cache_key"] = f"7hourly_schedule_{self.name}_{global_state.tick}_{i}"

            completion = await _hourly_breakdown_chain.arun(schedule_format=hourly_schedule_format_template, 
                                                        identity=self.identity, 
                                                        name=self.name,
                                                        daily_plan=self._build_daily_plan(), 
                                                        prior_schedule=self._build_hourly_schedule_opening())

            if completion.strip().endswith("}"):
                completion += "\n```"
            pattern = r'<\|assistant\|\>\n*```\n*(\{.*?\})\n?```'
            match = re.search(pattern, completion, re.DOTALL)
                
            if match:
                try:
                    json_object = json.loads(match.group(1))
                except Exception as error:
                    try:
                        json_object = await JsonExpert(wrong_json=match.group(1),
                                                       error_message=str(error)).run()
                    except:
                        continue

                try:  
                    schedule = json_object[f"{self.name}'s schedule"]
                    return self._fix_schedule(schedule)
                except:
                    pass
            
            i += 1
            print(f"Retry {i}: Failed to generate hourly schedule for {self.name}. Trying again...")      
    
    def _build_daily_plan(self) -> str: 
        statements = [f"{i}) {hourly_organized_activity['activity']} at {hourly_organized_activity['time']})" for i, hourly_organized_activity in enumerate(self.hourly_organized_activities)]
        return ", ".join(statements)

    def _build_hourly_schedule_opening(self) -> str: 
        wake_up_index = hours.index(self.wake_up_hour.zfill(8))
        hourly_schedule = _create_schedule_template(["sleeping" for i in range(wake_up_index)] + ["waking up and finishing morning routine", " "], self.name)
        hourly_schedule = hourly_schedule[:-20]
        return hourly_schedule

    def __sort_schedule(self, schedule: List[Dict[str, str]]) -> List[Dict[str, str]]:
        def convert_to_24h(time_str: str) -> str:
            in_time = datetime.datetime.strptime(time_str, "%I:%M %p")
            out_time = datetime.datetime.strftime(in_time, "%H:%M")
            return out_time

        return sorted(schedule, key=lambda x: convert_to_24h(x["time"]))

    def _fix_schedule(self, schedule: List[Dict[str, str]]) -> List[Dict[str, str]]:
        full_schedule = []
        
        # order schedule by hour
        schedule = self.__sort_schedule(schedule)
        
        if schedule[-1]["time"] != "11:00 PM":
            schedule.append({
                "time": "11:00 PM",
                "activity": "sleeping"
            })

        for schedule_entry in schedule:
            i = len(full_schedule)
            if i >= len(hours):
                break

            current_hour = hours[i]

            if schedule_entry["time"] == current_hour:
                full_schedule.append(schedule_entry)
                continue

            # filter not full hours
            if ":00" not in schedule_entry["time"]:
                continue
            
            for hour in hours[i:]:
                if hour == schedule_entry["time"]:
                    break

                full_schedule.append({"time": hour, "activity": full_schedule[-1]["activity"]})
            
            full_schedule.append(schedule_entry)

        return schedule
