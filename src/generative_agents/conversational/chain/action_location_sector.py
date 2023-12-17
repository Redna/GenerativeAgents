import json
import random
import re
from pydantic import BaseModel
from generative_agents import global_state
from generative_agents.conversational.llm import llm
from langchain import LLMChain, PromptTemplate


## fewshot
_template = """<|system|>You follow the tasks given by the user as close as possible. You will only generate 1 JSON object as mentioned below.
You will act as the agent {agent_name}.

<|user|>
Choose an appropriate area from the area options for a task at hand. Stay in the current area if the activity can be done there. Only go out if the activity needs to take place in another place.
Also if an activity is not in the options, try to identify the closest area that can be used for the activity.

Output format: Output a valid json of the following format:
```
{{
    "reasoning": "<reasoning for yes or no and the next area selection in one brief sentence>"
    "activity in current area": "<yes or no>",
    "area": "<next area MUST be one of the List enclosed by [] >"
}}
```

{agent_name} lives in [{agent_home}] that has {agent_home_arenas}.
{agent_name} is currently in [{agent_current_sector}] that has {agent_current_sector_arenas}.
Area options: [{available_sectors_nearby}].

For {curr_action_description}, where should {agent_name} go? And can {agent_name} do the activity in the current area?
<|assistant|>
{{
    "reasoning": \""""

class ActionSectorLocations(BaseModel):
    agent_name: str
    agent_home: str
    agent_home_arenas: str
    agent_current_sector: str
    agent_current_sector_arenas: str
    available_sectors_nearby: str
    curr_action_description: str

    async def run(self):
        _prompt = PromptTemplate(input_variables=["agent_name",
                                                  "agent_home",
                                                  "agent_home_arenas",
                                                  "agent_current_sector",
                                                  "agent_current_sector_arenas",
                                                  "available_sectors_nearby",
                                                  "curr_action_description"],
                                 template=_template)
        
        _action_sector_locations_chain = LLMChain(prompt=_prompt, llm=llm, llm_kwargs={
            "max_new_tokens": 200,
            "do_sample": True,
            "top_p": 0.95,
            "top_k": 60,
            "temperature": 0.4}, verbose=global_state.verbose)

        possible_sectors = [sector.strip() for sector in self.available_sectors_nearby.split(",") if sector.strip()]
        possible_sectors.append(self.agent_current_sector)
        possible_sectors.append(self.agent_home)

        for i in range(5):
            _action_sector_locations_chain.llm_kwargs["cache_key"] = f"6action_sector_locations_{self.agent_name}_{global_state.tick}_{i}"
            completion = await _action_sector_locations_chain.arun(agent_name=self.agent_name,
                                                                agent_home=self.agent_home,
                                                                agent_home_arenas=self.agent_home_arenas,
                                                                agent_current_sector=self.agent_current_sector,
                                                                agent_current_sector_arenas=self.agent_current_sector_arenas,
                                                                available_sectors_nearby=self.available_sectors_nearby,
                                                                curr_action_description=self.curr_action_description)
            
            pattern = r'<\|assistant\|\>\n*(\{.*?\})'
            match = re.search(pattern, completion, re.DOTALL)
            if match:
                try:
                    json_object = json.loads(match.group(1))
                    
                    sector = json_object["area"]
                    if sector in possible_sectors:
                        return sector
                    
                    if sector in self.agent_home_arenas:
                        return self.agent_home

                    if sector in self.agent_current_sector_arenas:
                        return self.agent_current_sector
                except:
                    pass
        in_current_area = json_object["activity in current area"]
        if in_current_area == "yes":
            sector = self.agent_current_sector
        else:
            sector = random.choice(possible_sectors)
        
        print(f"Unable to identify next location. Can be done in Sector: {in_current_area}. Selecting randomly: {sector}")
        return sector
