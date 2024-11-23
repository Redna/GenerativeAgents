from enum import Enum
from typing import Annotated, Type, TypedDict
from pydantic import BaseModel, Field

from langchain_groq.chat_models import ChatGroq
from langchain_core.messages import HumanMessage

llm = ChatGroq(model="llama3-8b-8192",
               name="action_sector_location")

template = """Choose an appropriate area from the area options for a given activity. Stay in the current area if the activity can be done there. Only go out if the activity needs to take place in another place.

{agent_name} lives in [{agent_home}] that has {agent_home_arenas}.
{agent_name} is currently in [{agent_current_sector}] that has {agent_current_sector_arenas}.
The following areas are nearby: [{available_sectors_nearby}].

For "{curr_action_description}", where should {agent_name} go?"""


def model_from_enum(dynamic_enum: Enum) -> Type[BaseModel]:
    class ActionSectorLocation(TypedDict):
        """ Choose an appropriate area from the area options for a given activity. """

        reasoning: Annotated[str, ..., "Concise reasoning for for the next area."]
        next_area: dynamic_enum

    return ActionSectorLocation


def action_sector_locations(agent_name: str, agent_home: str, agent_home_arenas: str, agent_current_sector: str, agent_current_sector_arenas: str, available_sectors_nearby: str, curr_action_description: str) -> str:
    possible_sectors = ",".join([agent_home, agent_current_sector, available_sectors_nearby]).replace(", ", ",")
    areas = Enum("Areas", {sector: sector for sector in possible_sectors.split(",") if sector})
    model = model_from_enum(areas)

    structured_llm = llm.with_structured_output(model)

    content = template.format(agent_name=agent_name, agent_home=agent_home, agent_home_arenas=agent_home_arenas,
                                 agent_current_sector=agent_current_sector, agent_current_sector_arenas=agent_current_sector_arenas,
                                 available_sectors_nearby=available_sectors_nearby, curr_action_description=curr_action_description)

    action_sector_location = structured_llm.invoke([HumanMessage(content=content)])
    return action_sector_location["next_area"]


if __name__ == "__main__":
    print(action_sector_locations(agent_name="Jimmy Foe",
                                  agent_home="Jimmy Foe's apartment",
                                  agent_home_arenas="living room, bathroom",
                                  agent_current_sector="Hobbs Cafe",
                                  agent_current_sector_arenas="cafe, restroom",
                                  available_sectors_nearby="Supermarket, Library, Lyn's family room",
                                  curr_action_description="drinking a cafe"))
    print(action_sector_locations(agent_name="John Doe",
                                    agent_home="John Doe's apartment",
                                    agent_home_arenas="bedroom, kitchen, living room, bathroom",
                                    agent_current_sector="Jimmies Pharmacy",
                                    agent_current_sector_arenas="counter",
                                    available_sectors_nearby="Supermarket, Library, Lyn's family room",
                                    curr_action_description="Visiting John Lyn"))