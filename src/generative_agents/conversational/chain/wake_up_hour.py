import json
import re
from langchain import LLMChain, PromptTemplate
from pydantic import BaseModel
from generative_agents import global_state

from generative_agents.conversational.llm import llm
from generative_agents.conversational.output_parser.fuzzy_parser import FuzzyOutputParser, PatternWithDefault

_template = """<|system|>You follow the tasks given by the user as close as possible. You will only generate 1 JSON object as mentioned below.
You will act as {agent_name}.

Your identity is: 
{agent_identity}
<|user|>
Output format: Output a valid json of the following format:
```
{{
    "wake up hour": "<time in 12-hour clock format>"
}}
```
--- 
{agent_lifestyle}
When does {agent_name} wake up today?:
<|assistant|>
```
{{
    "wake up hour": \""""

_prompt = PromptTemplate(input_variables=["agent_name", 
                                          "agent_lifestyle",
                                          "agent_identity"],
                         template=_template)

_wake_up_hour_chain = LLMChain(prompt=_prompt, llm=llm, llm_kwargs={"max_new_tokens": 15,
                                                                   "do_sample": True,
                                                                   "top_p": 0.95,
                                                                   "top_k": 60,
                                                                   "temperature": 0.4})


class WakeUpHour(BaseModel):
    agent_name: str
    agent_identity: str
    agent_lifestyle: str

    async def run(self):
        for i in range(5):   
            _wake_up_hour_chain.llm_kwargs["cache_key"] = f"1wake_up_hour_{self.agent_name}_{global_state.tick}_{i}"

            completion = await _wake_up_hour_chain.arun(agent_name=self.agent_name,
                                            agent_identity=self.agent_identity,
                                            agent_lifestyle="In genral, " + self.agent_lifestyle if self.agent_lifestyle else "")

            pattern = r'<\|assistant\|\>\n*```\n*(\{.*?\})\n```'
            match = re.search(pattern, completion, re.DOTALL)
            if match:
                try:
                    json_object = json.loads(match.group(1))
                    wake_up_hour = json_object["wake up hour"].strip()
                    wake_up_hour = wake_up_hour if "am" in wake_up_hour.lower() or "pm" in wake_up_hour.lower() else wake_up_hour + " AM"
                    return wake_up_hour
                except:
                    pass
        
        print("Unable to generate the wake up hour")
        return "6:00 AM"