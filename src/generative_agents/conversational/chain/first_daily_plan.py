import asyncio
import json
import re
from typing import List

from pydantic import BaseModel
from generative_agents import global_state
from generative_agents.conversational.llm import llm

from langchain.chains import LLMChain
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, ChatPromptTemplate

from generative_agents.conversational.output_parser.fuzzy_parser import FuzzyOutputParser, PatternWithDefault

system = """You will act as the agent {agent_name}. Given the following information, you will generate a plan for the day.
Note: In this villiage neither cars, nor bikes exist. The only way to get around is by walking.

Output format: Output a valid yaml of the following format:
```yaml
my_plan_for_today:
  - time: <time>
    activity: <activity>
  - time: <time>
    activity: <activity>
  - time: <time>
    activity: <activity>
  # ...
```
"""

user = """Your identity is:
{{agent_identity}}

In general, {{agent_lifestyle}}

Today is {{current_day}}.
What is {{agent_name}}'s plan for today in broad-strokes?"""

chat_template = ChatPromptTemplate(messages=[
    SystemMessagePromptTemplate.from_template(system, template_format="jinja2"),
    HumanMessagePromptTemplate.from_template(user, template_format="jinja2")],
)

_first_daily_plan_chain = LLMChain(prompt=chat_template, llm=llm, llm_kwargs={"max_tokens": 400,
                                                                    "top_p": 0.95,
                                                                   "temperature": 0.4}, verbose=global_state.verbose)

class FirstDailyPlan(BaseModel):
    agent_name: str
    agent_identity: str
    agent_lifestyle: str
    current_day: str
    wake_up_hour: str

    async def run(self):
        for i in range(5):   
            completion = await _first_daily_plan_chain.ainvoke(input={"agent_name": self.agent_name,
                                                                        "agent_identity": self.agent_identity,
                                                                        "agent_lifestyle": self.agent_lifestyle,
                                                                        "current_day": self.current_day,
                                                                        "wake_up_hour": self.wake_up_hour})
            

            pattern = r'(?:```yaml)?\n*(\.*?})\n(?:```)?'
            match = re.search(pattern, completion["text"], re.DOTALL)
            if match:
                try:
                    json_object = json.loads(match.group(1))
                    return json_object["my plan for today"]
                except:
                    pass
        
        print("Unable to generate the next utterance")
        return "I don't know what to say", True


async def __tests():
    t = [
        FirstDailyPlan(agent_name="John Smith",
                       agent_identity="John Smith is a 30 year old Software Engineer. He is open minded and likes to meet new people.",
                       agent_lifestyle="He is a single and lives near the city center. He likes to take long walks in the park. In his evening he likes to go to the pub and have a beer.",
                       current_day="Monday 02.03.2024",
                        wake_up_hour="7:00 AM").run(),
    ]
    
    return await asyncio.gather(*t)

if __name__ == "__main__":
    from pprint import pprint
    t = asyncio.run(__tests())
    pprint(t)