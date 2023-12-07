import random
import re
from pydantic import BaseModel
from generative_agents import global_state
from generative_agents.conversational.llm import llm
from langchain import LLMChain, PromptTemplate


## fewshot

_template = """
Choose an appropriate area from the area options for a task at hand.

Sam Kim lives in [Sam Kim's house] that has Sam Kim's room, bathroom, kitchen.
Sam Kim is currently in [Sam Kim's house] that has Sam Kim's room, bathroom, kitchen. 
Area options: [Sam Kim's house, The Rose and Crown Pub, Hobbs Cafe, Oak Hill College, Johnson Park, Harvey Oak Supply Store, The Willows Market and Pharmacy].
* Stay in the current area if the activity can be done there. Only go out if the activity needs to take place in another place.
* Must be one of the |Area options], verbatim.
For taking a walk, Sam Kim should go to the following area: [Johnson Park]
---
Jane Anderson lives in [Oak Hill College Student Dormatory] that has Jane Anderson's room.
Jane Anderson is currently in [Oak Hill College] that has a classroom, library
Area options: [Oak Hill College Student Dormatory, The Rose and Crown Pub, Hobbs Cafe, Oak Hill College, Johnson Park, Harvey Oak Supply Store, The Willows Market and Pharmacy]. 
* Stay in the current area if the activity can be done there. Only go out if the activity needs to take place in another place.
* Must be one of the |Area options], verbatim.
For eating dinner, Jane Anderson should go to the following area: [Hobbs Cafe]
--- 
{agent_name} lives in [{agent_home}] that has {agent_home_arenas}.
{agent_name} is currently in [{agent_current_sector}] that has {agent_current_sector_arenas}.
Area options: [{available_sectors_nearby}].
* Stay in the current area if the activity can be done there. Only go out if the activity needs to take place in another place.
* Must be one of the [Area options], verbatim.
For {curr_action_description}, {agent_name} should go to the following area: ["""


_fallback_template = """
{agent_name} is currently in [{agent_current_sector}] that has {agent_current_sector_arenas}.

{agent_name} is planning the following activity: {curr_action_description}

Answer with YES if the activity can be done in the current area. Otherwise, answer with NO.
Answer:"""

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
            "max_new_tokens": 15,
            "do_sample": True,
            "top_p": 0.95,
            "top_k": 60,
            "temperature": 0.4}, verbose=global_state.verbose)

        possible_sectors = [sector.strip() for sector in self.available_sectors_nearby.split(",") if sector.strip()]
        possible_sectors.append(self.agent_current_sector)
        possible_sectors.append(self.agent_home)

        for i in range(5):
            _action_sector_locations_chain.llm_kwargs["cache_key"] = f"action_sector_locations_{self.agent_name}_{global_state.tick}_{i}"
            completion = await _action_sector_locations_chain.arun(agent_name=self.agent_name,
                                                                agent_home=self.agent_home,
                                                                agent_home_arenas=self.agent_home_arenas,
                                                                agent_current_sector=self.agent_current_sector,
                                                                agent_current_sector_arenas=self.agent_current_sector_arenas,
                                                                available_sectors_nearby=self.available_sectors_nearby,
                                                                curr_action_description=self.curr_action_description)
            
            pattern = rf"{self.agent_name} should go to the following area: \[(.*)\]"
            try:
                sector = re.findall(pattern, completion)[-1]
            except:
                continue
            
            if sector in possible_sectors:
                return sector
            
            if sector in self.agent_home_arenas:
                return self.agent_home
            
            if sector in self.agent_current_sector_arenas:
                return self.agent_current_sector


        _fallback_prompt = PromptTemplate(input_variables=["agent_name",
                                    "agent_current_sector",
                                    "agent_current_sector_arenas",
                                    "curr_action_description"],
                    template=_fallback_template)
        
        _fallback_chain = LLMChain(prompt=_fallback_prompt, llm=llm, llm_kwargs={
            "max_new_tokens": 3,
            "do_sample": True,
            "top_p": 0.95,
            "top_k": 60,
            "temperature": 0.1}, verbose=global_state.verbose)

        available_sectors_nearby = [sector.strip() for sector in self.available_sectors_nearby.split(",") if sector.strip()]
        
        for i in range(5):
            _action_sector_locations_chain.llm_kwargs["cache_key"] = f"action_sector_locations_fallback_{self.agent_name}_{global_state.tick}_{i}"
            completion = await _fallback_chain.arun(agent_name=self.agent_name,
                                                    agent_current_sector=self.agent_current_sector,
                                                    agent_current_sector_arenas=self.agent_current_sector_arenas,
                                                    curr_action_description=self.curr_action_description)
            
            pattern = rf"Answer: \[(.*)\]"
            try:
                answer = re.findall(pattern, completion)[-1]
                if "yes" in answer.lower():
                    print(f"Activity can be done in the current area: {self.agent_current_sector}")
                    return self.agent_current_sector
                if "no" in answer.lower():
                    print("fallback: Activity cannot be done in the current area. Selecting randomly.")
                    return random.choice(available_sectors_nearby)
            except:
                continue

        sector = random.choice(possible_sectors)
        print(f"Unable to identify next location. Selecting randomly: {sector}")
        return sector
