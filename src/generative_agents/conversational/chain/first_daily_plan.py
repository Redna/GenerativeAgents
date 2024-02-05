import asyncio
import yaml
import re
from typing import List

from pydantic import BaseModel
from generative_agents import global_state
from generative_agents.conversational.llm import llm

from langchain.chains import LLMChain
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, ChatPromptTemplate

system = """You will act as the agent {agent_name}. Given the following information, you will generate a plan for the day.
Note: In this villiage neither cars, nor bikes exist. The only way to get around is by walking.

Output format: Output a valid yaml of the following format:
```yaml
my_plan_for_today:
  - time: <start time in 12-hour clock format>
    activity: <activity>
  - time: <star time in 12-hour clock format>
    activity: <activity>
  - time: <start time in 12-hour clock format>
    activity: <activity>
  # ...
```
"""

user_shot_1 = """Your identity is:
John Smith is a 30 year old Software Engineer. He is open minded and likes to meet new people.

In general, He is a single and lives near the city center. He likes to take long walks in the park. In his evening he likes to go to the pub and have a beer.

Today is Monday 02.03.2024.
What is John Smith's plan for today in broad-strokes?"""

ai_shot_1 = """```yaml
my_plan_for_today:
- time: 08:00 AM
  activity: Start work from home as a Software Engineer
- time: 12:30 PM
  activity: Lunch break and take a walk in the park near his residence
- time: 05:30 PM
  activity: Finish work, get ready, and head to the town square for socializing and meeting new people
- time: 07:00 PM
  activity: Visit a local pub for an evening beer
- time: 10:00 PM
  activity: go to sleep
```"""

user = """Your identity is:
{{agent_identity}}

In general, {{agent_lifestyle}}

Today is {{current_day}}.
What is {{agent_name}}'s plan for today in broad-strokes?"""

ai = """```yaml
my_plan_for_today:
- time: """

class FirstDailyPlan(BaseModel):
    agent_name: str
    agent_identity: str
    agent_lifestyle: str
    current_day: str
    wake_up_hour: str

    async def run(self):

        chat_template = ChatPromptTemplate(messages=[
            SystemMessagePromptTemplate.from_template(
                system, template_format="jinja2"),
            HumanMessagePromptTemplate.from_template(
                user_shot_1, template_format="jinja2"),
            AIMessagePromptTemplate.from_template(
                ai_shot_1, template_format="jinja2"),
            HumanMessagePromptTemplate.from_template(user, template_format="jinja2"),
            AIMessagePromptTemplate.from_template(ai, template_format="jinja2")]
        )

        _first_daily_plan_chain = LLMChain(prompt=chat_template, llm=llm, llm_kwargs={"max_tokens": 400,
                                                                                      "top_p": 0.95,
                                                                                      "temperature": 0.4}, verbose=True)

        for i in range(5):
            completion = await _first_daily_plan_chain.ainvoke(input={"agent_name": self.agent_name,
                                                                      "agent_identity": self.agent_identity,
                                                                      "agent_lifestyle": self.agent_lifestyle,
                                                                      "current_day": self.current_day,
                                                                      "wake_up_hour": self.wake_up_hour})
            pattern = r'```yaml(.*)```'
            match = re.search(pattern, completion["text"], re.DOTALL)
            if match: 
                try:
                    json_object = yaml.safe_load(match.group(1))
                    return json_object["my_plan_for_today"]
                except:
                    pass
            else:
                print("No yaml pattern found")
                continue

        print("Unable to generate the next utterance")
        return [{"activity": "Enjoying the day", "time": "11:00 AM"}]


async def __tests():
    t = [
        FirstDailyPlan(
            agent_name="Emily Johnson",
            agent_identity="Emily Johnson is a 28 year old graphic designer. She is creative and enjoys exploring new art forms.",
            agent_lifestyle="She lives with her two cats and enjoys gardening. Emily is a yoga enthusiast and likes to cook healthy meals.",
            current_day="Wednesday 04.03.2024",
            wake_up_hour="6:30 AM"
        ).run(),
        FirstDailyPlan(
            agent_name="Michael Brown",
            agent_identity="Michael Brown is a 45 year old teacher. He is passionate about education and loves reading.",
            agent_lifestyle="He is married with three children and lives in a suburban area. He enjoys jogging and playing guitar in his free time.",
            current_day="Friday 06.03.2024",
            wake_up_hour="5:45 AM"
        ).run(),
        FirstDailyPlan(
            agent_name="Sara Kim",
            agent_identity="Sara Kim is a 35 year old entrepreneur. She is ambitious and enjoys networking.",
            agent_lifestyle="She is a fitness enthusiast and follows a strict workout regimen. Sara loves to travel and experience different cultures.",
            current_day="Sunday 08.03.2024",
            wake_up_hour="8:00 AM"
        ).run(),
        FirstDailyPlan(
            agent_name="Alex Martinez",
            agent_identity="Alex Martinez is a 22 year old college student studying biology. He is curious and enjoys outdoor activities.",
            agent_lifestyle="He lives in a dorm and often participates in campus events. Alex likes hiking and playing soccer with friends.",
            current_day="Tuesday 10.03.2024",
            wake_up_hour="7:30 AM"
        ).run()
    ]

    return await asyncio.gather(*t)

if __name__ == "__main__":
    from pprint import pprint
    t = asyncio.run(__tests())
    pprint(t)
