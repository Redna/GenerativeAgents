import json
import re
from pydantic import BaseModel
from generative_agents import global_state
from generative_agents.conversational.llm import llm
from langchain import LLMChain, PromptTemplate
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, ChatPromptTemplate

system = """You follow the task given by the user as close as possible. You will only generate 1 valid JSON object as mentioned below.
Choose an appropriate area from the area options for a given activity. Stay in the current area if the activity can be done there. Only go out if the activity needs to take place in another place.
Also if the activity cannot be done in the available options, try to identify the closest area that can be used for the activity.

Output format: Output a valid JSON of the following format:
{{
    "reasoning": "<reasoning for yes or no and the next area selection in one brief sentence>",
    "activity in current area": "<yes or no>",
    "area": "<next MUST be exactly one area enclosed by []>"
}}"""

user_shot_1 = """John Doe lives in [John Doe's apartment] that has bedroom, kitchen, living room, bathroom.
John Doe is currently in [John Doe\'s apartment] that has bathroom, living room, kitchen.
The following areas are nearby: [Supermarket, Library, Lyn's family room].

Task: For "taking a warm shower", where should John Doe go? And can John Doe do the activity in the current area?"""

ai_shot_1 = """{{
    "reasoning": "John Doe can take a warm shower at [John Doe\'s apartment] as it has a bathroom.",
    "activity in current area": "Yes",
    "area": "John Doe\'s apartment"
}}"""

user_shot_2 = """John Doe lives in [John Doe's apartment] that has bedroom, kitchen, living room, bathroom.
John Doe is currently in [Hobb's Cafe] that has cafe, restroom.
The following areas are nearby: [Supermarket, Library, Lyn's family room].

Task: For "getting a new book", where should John Doe go? And can John Doe do the activity in the current area?"""

ai_shot_2 = """{{
    "reasoning": "John Doe can get a new book at [Library]. A library is a place where books are kept.",
    "activity in current area": "No",
    "area": "Library"
}}"""

user = """{agent_name} lives in [{agent_home}] that has {agent_home_arenas}.
{agent_name} is currently in [{agent_current_sector}] that has {agent_current_sector_arenas}.
The following areas are nearby: [{available_sectors_nearby}].

Task: For "{curr_action_description}", where should {agent_name} go? And can {agent_name} do the activity in the current area?"""


class ActionSectorLocations(BaseModel):
    agent_name: str
    agent_home: str
    agent_home_arenas: str
    agent_current_sector: str
    agent_current_sector_arenas: str
    available_sectors_nearby: str
    curr_action_description: str

    async def run(self):
        chat_template = ChatPromptTemplate(messages=[
            SystemMessagePromptTemplate.from_template(system),
            HumanMessagePromptTemplate.from_template(user_shot_1),
            AIMessagePromptTemplate.from_template(ai_shot_1),
            HumanMessagePromptTemplate.from_template(user_shot_2),
            AIMessagePromptTemplate.from_template(ai_shot_2),
            HumanMessagePromptTemplate.from_template(user)])

        _action_sector_locations_chain = LLMChain(prompt=chat_template, llm=llm, llm_kwargs={
            "max_tokens": 200,
            "top_p": 0.90,
            "temperature": 0.7}, verbose=global_state.verbose)

        possible_sectors = [sector.strip(
        ) for sector in self.available_sectors_nearby.split(",") if sector.strip()]
        possible_sectors.append(self.agent_current_sector)
        possible_sectors.append(self.agent_home)

        for i in range(5):

            completion = await _action_sector_locations_chain.ainvoke(input={"agent_name": self.agent_name,
                                                                             "agent_home": self.agent_home,
                                                                             "agent_home_arenas": self.agent_home_arenas,
                                                                             "agent_current_sector": self.agent_current_sector,
                                                                             "agent_current_sector_arenas": self.agent_current_sector_arenas,
                                                                             "available_sectors_nearby": self.available_sectors_nearby,
                                                                             "curr_action_description": self.curr_action_description})

            try:
                json_object = json.loads(completion["text"])

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
            sector = "<random>"

        print(
            f"Unable to identify next location. Can be done in Sector: {in_current_area}. Selecting randomly: {sector}")
        return sector


async def __tests():
    global_state.verbose = True
    t = [ActionSectorLocations(agent_name="Jimmy Foe", agent_home="Jimmy Foe's apartment", agent_home_arenas="living room, bathroom", agent_current_sector="Hobbs Cafe", agent_current_sector_arenas="cafe, restroom", available_sectors_nearby="Supermarket, Library, Lyn's family room", curr_action_description="drinking a cafe").run(),
         ActionSectorLocations(agent_name="John Doe", agent_home="John Doe's apartment", agent_home_arenas="bedroom, kitchen, living room, bathroom", agent_current_sector="Jimmies Pharmacy",
                               agent_current_sector_arenas="counter", available_sectors_nearby="Supermarket, Library, Lyn's family room", curr_action_description="Visiting John Lyn").run(),
         ActionSectorLocations(agent_name="John Doe", agent_home="John Doe's apartment", agent_home_arenas="bedroom, kitchen, living room, bathroom", agent_current_sector="John Doe's apartment", agent_current_sector_arenas="bedroom, kitchen, living room, bathroom", available_sectors_nearby="Supermarket, Library, Lyn's family room", curr_action_description="Buying a can of beans").run(),]

    return await asyncio.gather(*t)

if __name__ == "__main__":
    import asyncio
    t = asyncio.run(__tests())
    print(t)
