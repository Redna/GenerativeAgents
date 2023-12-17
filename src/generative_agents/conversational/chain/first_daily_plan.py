import json
import re
from typing import List

from pydantic import BaseModel
from generative_agents import global_state
from generative_agents.conversational.llm import llm
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from generative_agents.conversational.output_parser.fuzzy_parser import FuzzyOutputParser, PatternWithDefault

_template = """<|system|>You follow the tasks given by the user as close as possible. You will only generate 1 valid JSON object as mentioned below.
You will act as the agent {agent_name}.

Your identity is: 
{agent_identity}

Note: In this villiage neither cars, nor bikes exist. The only way to get around is by walking.
<|user|>
Output format: Output a valid json of the following format:
```
{{
    "my plan for today": [
        {{
            "time": "<time>",
            "activity": "<activity>"
        }},
        {{
            "time": "<time>",
            "activity": "<activity>"
        }},
        {{
            "time": "<time>",
            "activity": "<activity>"
        }}
    ]
}}
```
---
In general, {agent_lifestyle}

Today is {current_day}. 
Task: Create a valid JSON object of {agent_name}'s plan for today in broad-strokes:

<|assistant|>
```
{{
    "my plan for today": [
        {{
            "time": "{wake_up_hour}",
            "activity": "wake up and complete the morning routine"
        }},
        {{
            "time": \""""

_prompt = PromptTemplate(input_variables=["agent_name",
                                          "agent_identity",
                                          "agent_lifestyle",
                                          "current_day",
                                          "wake_up_hour"],
                            template=_template)

_first_daily_plan_chain = LLMChain(prompt=_prompt, llm=llm, llm_kwargs={"max_new_tokens": 400,
                                                                   "do_sample": True,
                                                                   "top_p": 0.95,
                                                                   "top_k": 50,
                                                                   "temperature": 0.4}, verbose=global_state.verbose)


class FirstDailyPlan(BaseModel):
    agent_name: str
    agent_identity: str
    agent_lifestyle: str
    current_day: str
    wake_up_hour: str

    async def run(self):
        for i in range(5):   
            _first_daily_plan_chain.llm_kwargs["cache_key"] = f"5first_daily_plan_{self.agent_name}_{global_state.tick}_{i}"

            completion = await _first_daily_plan_chain.arun(agent_name=self.agent_name,
                                            agent_identity=self.agent_identity,
                                            agent_lifestyle=self.agent_lifestyle,
                                            current_day=self.current_day,
                                            wake_up_hour=self.wake_up_hour)

            pattern = r'<\|assistant\|\>\n*```\n*(\{.*?\})\n```'
            match = re.search(pattern, completion, re.DOTALL)
            if match:
                try:
                    json_object = json.loads(match.group(1))
                    return json_object["my plan for today"]
                except:
                    pass
        
        print("Unable to generate the next utterance")
        return "I don't know what to say", True
        