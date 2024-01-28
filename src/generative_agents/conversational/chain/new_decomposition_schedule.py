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
from generative_agents.utils import hour_string_to_time

class NewDecompositionSchedule(BaseModel):
    
    agent: str
    start_hour: int
    end_hour: int
    new_event: str
    new_event_duration: int
    new_event_index: int
    schedule_slice: List[Tuple[str, int]]

    async def run(self): 
        _template = """<|system|>You follow the tasks given by the user as close as possible. You will only generate 1 valid JSON object as mentioned below.
You will act as {agent}.

Output format: Output a valid json of the following format:
```
{{
    "{agent}'s schedule": [
        {{ 
            "time": "<time in 12-hour clock format>",
            "activity": "<activity in one brief sentence>"
            "duration": "<duration in minutes>"
        }},
        {{ 
            "time": "<time in 12-hour clock format>",
            "activity": "<activity in one brief sentence>"
            "duration": "<duration in minutes>"
        }},
        {{ 
            "time": "<time in 12-hour clock format>",
            "activity": "<activity in one brief sentence>"
            "duration": "<duration in minutes>"
        }},
        ...,
        {{  
            "time": "<time in 12-hour clock format>",
            "activity": "<activity in one brief sentence>"
            "duration": "<duration in minutes>"
        }}
    ]
}}
````

Here was {agent}'s originally planned schedule from {start_hour} to {end_hour}.
{{
    "{agent}'s schedule": {schedule}
}}

Task: {agent} unexpectedly ended up with "{new_event}" for "{new_event_duration}" minutes. How does the {agent}'s revised schedule from {start_hour} to {end_hour} look like? (the last activity MUST end by {end_hour})
<|assistant|>
```
{new_schedule_init}"""

        _prompt = PromptTemplate(input_variables=["agent",
                                                            "start_hour",
                                                            "end_hour",
                                                            "schedule",
                                                            "new_event",
                                                            "new_event_duration",
                                                            "new_schedule_init"
                                                            ],
                                        template=_template)
                
        _new_decomposition_schedule = LLMChain(prompt=_prompt, llm=llm, llm_kwargs={
                                                        "max_new_tokens":650,
                                                        "do_sample": True,
                                                        "top_p": 0.94,
                                                        "top_k": 40,
                                                        "temperature": 0.6}
                                                        , verbose=global_state.verbose)
                
        schedule, new_schedule_init = self._generate_new_schedule_init()

        start_hour_str = hour_string_to_time(self.start_hour).strftime("%I:%M %p")
        end_hour_str = hour_string_to_time(self.end_hour).strftime("%I:%M %p")

        if not new_schedule_init:
            new_schedule_init = f"""
{{
    "{self.agent}'s schedule": [
    {{
        "time": "{start_hour_str}",
        "activity": "{self.new_event}",
        "duration": {self.new_event_duration}
    }},"""      
        else:
            new_schedule_init = f"""
{{
    "{self.agent}'s schedule": {new_schedule_init},"""

        i = 0
        while True:   
            _new_decomposition_schedule.llm_kwargs["cache_key"] = f"16new_decomp_schedule_{self.agent}_{global_state.tick}_{i}"

            completion = await _new_decomposition_schedule.arun(agent=self.agent,
                                                        start_hour=start_hour_str,
                                                        end_hour=end_hour_str,
                                                        schedule=schedule,
                                                        new_event=self.new_event,
                                                        new_event_duration=self.new_event_duration,
                                                        new_schedule_init=new_schedule_init)

            pattern = r'<\|assistant\|\>\n*```\n*(\{.*?\})\n```'
            match = re.search(pattern, completion, re.DOTALL)
            if match:
                try:
                    json_object = json.loads(match.group(1))
                except Exception as error:
                    try:
                        json_object = await JsonExpert(wrong_json=match.group(1),
                                                    error_message=str(error)).run()
                    except Exception as e:
                        continue

                try: 
                    schedule = json_object[f"{self.agent}'s schedule"]
                    sched = self._fix_schedule(schedule)
                    return [(entry["activity"], entry["duration"]) for entry in sched]
                except:
                    pass

            i += 1
            print("Retry {i}, new decomposition failed")
    
    def _generate_new_schedule_init(self) -> str:
        start_hour = hour_string_to_time(self.start_hour)
        for_time = start_hour

        schedule_with_hour = []
        for action, duration in self.schedule_slice:
            schedule_with_hour.append({"time": for_time.strftime("%H:%M %p"), "activity": action, "duration": duration})
            for_time += datetime.timedelta(minutes=int(duration))

        return json.dumps(schedule_with_hour, indent=4), json.dumps(schedule_with_hour[:self.new_event_index], indent=4)[:-2]
           
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
