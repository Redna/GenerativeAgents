import asyncio
import datetime
import yaml
from random import random
import re
from typing import Dict, List, Tuple

from pydantic import BaseModel
from generative_agents.conversational.chain.json_expert import JsonExpert
from generative_agents.conversational.llm import llm

from langchain.chains import LLMChain
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, ChatPromptTemplate


from generative_agents.core.whisper.whisper import whisper

hours = ["12:00 AM", "01:00 AM", "02:00 AM", "03:00 AM", "04:00 AM", "05:00 AM", "06:00 AM", "07:00 AM", "08:00 AM", "09:00 AM", "10:00 AM", "11:00 AM",
         "12:00 PM", "01:00 PM", "02:00 PM", "03:00 PM", "04:00 PM", "05:00 PM", "06:00 PM", "07:00 PM", "08:00 PM", "09:00 PM", "10:00 PM", "11:00 PM"]

system = """You are a character in a role play game. You are thinking about your day and create an hourly schedule. 
Note: In this villiage neither cars, nor bikes exist. The only way to get around is by walking.

Output format: Output a valid yaml of the following format:
    ```yaml
    daily_schedule: {% for hour in hours %}
        - time: {{hour}}
          activity: <activity during that hour in one brief sentence>
          duration: <duration in minutes>
    {%- endfor %}
    ```
"""

user_shot_1 = """You will act as Alex Rivera.
Your identity is: 
Alex Rivera is a 40 year old freelance writer. He is a storyteller at heart and enjoys crafting narratives. He lives by the sea and finds the sound of waves calming. Alex is flexible with his schedule but tries to maintain a balance between work and leisure.

Here is today's plan in broad-strokes:
1.) morning coffee and reading at 09:00 AM
2.) writing session at 10:00 AM
3.) lunch and beach walk at 01:00 PM
4.) writing session or client meetings at 02:00 PM
5.) evening relaxation, possibly yoga at 06:00 PM
6.) dinner at 08:00 PM
7.) leisure reading or watching a movie at 09:00 PM
8.) go to sleep at 11:00 PM

How does Alex Rivera's complete hourly schedule look for today? You must follow the schedule format above. Do not forget any time slots (Starting from 12:00 AM to 11:00 PM)."""

ai_shot_1 = """Here is Emily Johnson's complete hourly schedule for today:

```yaml
daily_schedule: 
    - time: 12:00 AM
      activity: sleeping
      duration: 60
    - time: 01:00 AM
      activity: sleeping
      duration: 60
    - time: 02:00 AM
      activity: sleeping
      duration: 60
    - time: 03:00 AM
      activity: sleeping
      duration: 60
    - time: 04:00 AM
      activity: sleeping
      duration: 60
    - time: 05:00 AM
      activity: sleeping
      duration: 60
    - time: 06:00 AM
      activity: feeding her cats and starting morning routine
      duration: 60
    - time: 07:00 AM
      activity: waking up and finishing morning routine
      duration: 60
    - time: 08:00 AM
      activity: having breakfast
      duration: 60
    - time: 09:00 AM
      activity: starting work as a graphic designer
      duration: 60
    - time: 10:00 AM
      activity: working on design projects
      duration: 60
    - time: 11:00 AM
      activity: preparing for lunch break
      duration: 60
    - time: 12:00 PM
      activity: having lunch
      duration: 60
    - time: 01:00 PM
      activity: resuming work as a graphic designer
      duration: 60
    - time: 02:00 PM
      activity: working on design projects
      duration: 60
    - time: 03:00 PM
      activity: finishing work for the day and packing up
      duration: 60
    - time: 04:00 PM
      activity: walking to the town square
      duration: 60
    - time: 05:00 PM
      activity: socializing, meeting new people at the town square
      duration: 60
    - time: 06:00 PM
      activity: preparing for an evening beer at a local pub
      duration: 60
    - time: 07:00 PM
      activity: visiting a local pub for an evening beer
      duration: 60
    - time: 08:00 PM
      activity: leaving the local pub and walking back home
      duration: 60
    - time: 09:00 PM
      activity: preparing dinner and cooking a healthy meal
      duration: 60
    - time: 10:00 PM
      activity: having dinner and relaxing
      duration: 60
    - time: 11:00 PM
      activity: sleeping
      duration: 60
```"""

user = """You will act as {{name}}.
Your identity is: 
{{identity}}

Here is today's plan in broad-strokes:
{%- for daily_plan_item in daily_plan %}
{{loop.index}}.) {{daily_plan_item.activity}} at {{daily_plan_item.time}}
{%- endfor %}

How does {{name}}'s complete hourly schedule look for today? You must follow the schedule format above. Do not forget any time slots (Starting from 12:00 AM to 11:00 PM)."""

ai = """Here is {{name}}'s complete hourly schedule for today:

```yaml
daily_schedule: {% for hour, activity, duration in prior_schedule %}
    - time: {{hour}}
      activity: {{activity}}
      duration: {{duration}}
{%- endfor %}
    - time: {{next_hour}}
      activity: """

class HourlyBreakdown(BaseModel):

    identity: str
    wake_up_hour: str
    name: str
    hourly_organized_activities: List[Dict[str, str]]

    async def run(self):
        chat_template = ChatPromptTemplate(messages=[
            SystemMessagePromptTemplate.from_template(
                system, template_format="jinja2"),
            HumanMessagePromptTemplate.from_template(user_shot_1, template_format="jinja2"),
            AIMessagePromptTemplate.from_template(ai_shot_1, template_format="jinja2"),
            HumanMessagePromptTemplate.from_template(user, template_format="jinja2"),
            AIMessagePromptTemplate.from_template(ai, template_format="jinja2")])
        _hourly_breakdown_chain = LLMChain(prompt=chat_template, llm=llm, llm_kwargs={
            "max_tokens": 1000,
            "top_p": 0.96,
            "temperature": 0.4}, verbose=True)

        i = 0
        while True:
            opening = self._build_hourly_schedule_opening()
            next_hour = hours[hours.index(opening[-1][0]) + 1]

            completion = await _hourly_breakdown_chain.ainvoke(input={"identity": self.identity,
                                                                      "name": self.name,
                                                                      "hours": hours,
                                                                      "daily_plan": self.hourly_organized_activities,
                                                                      "prior_schedule": opening,
                                                                      "next_hour": next_hour})
            pattern = r'```yaml(.*)```'
            match = re.search(pattern, completion["text"], re.DOTALL)
            if match:
                try:
                    output = yaml.safe_load(match.group(1))
                except Exception as error:
                    try:
                        output = await JsonExpert(wrong_json=match.group(1),
                                                       error_message=str(error)).run()
                    except:
                        continue

                try:
                    schedule = output["daily_schedule"]
                    return self._fix_schedule(schedule)
                except:
                    pass

            i += 1
            print(
                f"Retry {i}: Failed to generate hourly schedule for {self.name}. Trying again...")

    def _build_hourly_schedule_opening(self) -> str:
        try: 
          wake_up_index = hours.index(self.wake_up_hour.zfill(8))
        except:
          wake_up_hour = self.wake_up_hour.zfill(8)[0:2] + ":00 AM"
          wake_up_index = hours.index(wake_up_hour)

        prior_schedule = []
        for hour in hours[:wake_up_index]:
            prior_schedule.append((hour, "sleeping", 60))

        prior_schedule.append(
            (hours[wake_up_index], "waking up and finishing morning routine", 60))
        return prior_schedule

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

                full_schedule.append(
                    {"time": hour, "activity": full_schedule[-1]["activity"]})

            full_schedule.append(schedule_entry)

        return schedule


async def __tests():
    t = [
        HourlyBreakdown(
            identity="Emily Johnson is a 28 year old graphic designer. She is creative and enjoys exploring new art forms. She lives with her two cats and enjoys gardening. Emily is a yoga enthusiast and likes to cook healthy meals. She is a morning person and enjoys waking up early. She is a social person and enjoys meeting new people.",
            wake_up_hour="07:00 AM",
            name="Emily Johnson",
            hourly_organized_activities=[
                {"time": "08:00 AM", "activity": "breakfast"},
                {"time": "09:00 AM", "activity": "work"},
                {"time": "12:00 PM", "activity": "lunch"},
                {"time": "01:00 PM", "activity": "work"},
                {"time": "05:00 PM", "activity": "finish work and head to the town square for socializing and meeting new people"},
                {"time": "07:00 PM",
                 "activity": "visit a local pub for an evening beer"},
                {"time": "09:00 PM", "activity": "head back home"},
                {"time": "10:00 PM", "activity": "go to sleep"}]).run(),
        HourlyBreakdown(
            identity="John Doe is a 35 year old software developer. He is passionate about technology and loves coding. He lives in a quiet suburb and enjoys the peace it offers. John is an avid reader and spends his evenings reading tech articles. He prefers a structured day and enjoys the solitude of working from home.",
            wake_up_hour="06:00 AM",
            name="John Doe",
            hourly_organized_activities=[
                {"time": "07:00 AM", "activity": "morning run"},
                {"time": "08:00 AM", "activity": "breakfast and reading tech news"},
                {"time": "09:00 AM", "activity": "start work"},
                {"time": "12:00 PM", "activity": "lunch break and a short walk"},
                {"time": "01:00 PM", "activity": "continue work"},
                {"time": "06:00 PM", "activity": "end work and relax with a book"},
                {"time": "08:00 PM", "activity": "dinner"},
                {"time": "09:00 PM", "activity": "leisure time or hobby projects"},
                {"time": "11:00 PM", "activity": "go to sleep"}
            ]
        ).run(),
        HourlyBreakdown(
            identity="Sarah Lee is a 30 year old artist. She is deeply passionate about painting and spends most of her day in her studio. She loves nature and often takes long walks to find inspiration. Sarah is a night owl and finds herself most creative during the late hours.",
            wake_up_hour="09:00 AM",
            name="Sarah Lee",
            hourly_organized_activities=[
                {"time": "10:00 AM", "activity": "breakfast and morning meditation"},
                {"time": "11:00 AM", "activity": "studio time for painting"},
                {"time": "02:00 PM", "activity": "lunch and a walk in the park"},
                {"time": "03:00 PM", "activity": "return to studio work"},
                {"time": "07:00 PM", "activity": "dinner and socialize with friends"},
                {"time": "09:00 PM", "activity": "evening studio session"},
                {"time": "01:00 AM", "activity": "relaxation and bedtime routine"},
                {"time": "02:00 AM", "activity": "go to sleep"}
            ]
        ).run()
    ]

    return await asyncio.gather(*t)

if __name__ == "__main__":
    from pprint import pprint
    t = asyncio.run(__tests())
    pprint(t)
