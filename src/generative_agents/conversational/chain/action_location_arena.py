import json
import random
import re
from pydantic import BaseModel
from generative_agents import global_state
from generative_agents.conversational.llm import llm
from langchain import LLMChain, PromptTemplate
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, ChatPromptTemplate

system = """Your task is to identify the next area for {name}. You need to output valid JSON.
Output format: 
{{
    "reason": "<for the next area selection>",
    "area": "<name of the area>"
}}"""

user_shot_1 = """Jane Anderson is in "kitchen" in "Jane Anderson's house".
Jane Anderson is going to "Jane Anderson's house" that has the following areas: [kitchen,  bedroom, bathroom]
Stay in the current area if the activity can be done there. Never go into other people's rooms unless necessary.
For cooking, to which area should Jane Anderson in "Jane Anderson's house"?"""

ai_shot_1 = """
{{
    "reason": "For cooking Jane Anderson should go to the kitchen.",
    "area": "kitchen"
}}
"""

user_shot_2 = """Tom Watson is in "common room" in "Tom Watson's apartment".
Tom Watson is going to "Hobbs Cafe" that has the following areas: [cafe]
Stay in the current area if the activity can be done there. Never go into other people's rooms unless necessary.
For getting coffee, to which area should Tom Watson go in "Hobbs Cafe"?"""

ai_shot_2 = """
{{
    "reason": "For getting coffee Tom Watson should go to the cafe.",
    "area": "cafe"
}}
"""

user_shot_3 = """{name} is in "{current_area}" in "{current_sector}".
{name} is going to "{sector}" that has the following areas: [{sector_arenas}]
Stay in the current area if the activity can be done there. Never go into other people's rooms unless necessary.
For {action_description}, to which area should {name} go in "{sector}"?"""

chat_template = ChatPromptTemplate(messages=[
        SystemMessagePromptTemplate.from_template(system),
        HumanMessagePromptTemplate.from_template(user_shot_1),
        AIMessagePromptTemplate.from_template(ai_shot_1),
        HumanMessagePromptTemplate.from_template(user_shot_2),
        AIMessagePromptTemplate.from_template(ai_shot_2),
        HumanMessagePromptTemplate.from_template(user_shot_3)])


class ActionArenaLocations(BaseModel):
    name: str
    current_area: str
    current_sector: str
    sector: str
    sector_arenas: str
    action_description: str

    async def run(self):
        _action_arena_locations_chain = LLMChain(prompt=chat_template, llm=llm, llm_kwargs={
            "max_tokens": 80,
            "top_p": 0.90,
            "temperature": 0.4,
        }, verbose=global_state.verbose)

        possible_arenas = [arena.strip() for arena in self.sector_arenas.split(",") if arena.strip()]
        possible_arenas.append(self.current_area)

        for i in range(5):
            completion = await _action_arena_locations_chain.ainvoke(input={"name": self.name,
                                                         "action_description": self.action_description})
            try:
                json_object = json.loads(completion["text"])
                arena = json_object["area"]
                if arena in possible_arenas:
                    return arena
            except Exception as e:
                pass
        
        arena = random.choice(possible_arenas)
        print("Unable to identify next location. Selecting randomly: ", arena)
        return arena

async def __tests():
    t = [ActionArenaLocations(name="John Doe", current_area="kitchen", current_sector="John Doe's house", sector="John Doe's house", sector_arenas="kitchen, bedroom, bathroom", action_description="Taking a shower").run(),
         ActionArenaLocations(name="John Doe", current_area="common room", current_sector="John Doe's apartment", sector="Jimmy Hay's", sector_arenas="kitchen, bedroom, bathroom", action_description="Getting coffee").run(),
         ActionArenaLocations(name="John Doe", current_area="common room", current_sector="John Doe's apartment", sector="John Doe's apartment", sector_arenas="kitchen, bedroom, bathroom", action_description="Putting on trousers").run(),
         ActionArenaLocations(name="John Doe", current_area="common room", current_sector="John Doe's apartment", sector="Hobbs Cafe", sector_arenas="kitchen, bedroom, bathroom", action_description="Putting on trousers").run(),]

    return await asyncio.gather(*t)

if __name__ == "__main__":
    import asyncio
    t = asyncio.run(__tests())
    print(t)