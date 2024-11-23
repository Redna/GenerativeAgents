from enum import Enum
from typing import Type
from pydantic import BaseModel, Field

from langchain_groq.chat_models import ChatGroq
from langchain_core.messages import HumanMessage

llm = ChatGroq(model="llama3-8b-8192",
               name="action_arena_location")

template = """Your task is to identify the next area for a character. It has to be one area of the provided list

{name} is in the area "{current_area}" in "{current_sector}".
{name} is going to "{sector}" that has the following areas: [{sector_arenas}]
Stay in the current area if the activity can be done there. Never go into other people's rooms unless necessary.
For {action_description}, to which area should {name} go in "{sector}"?
"""

def model_from_enum(dynamic_enum: Enum) -> Type[BaseModel]:
    class ActionArenaLocation(BaseModel):
        """ Choose an appropriate area from the area options for a given activity. """
        reasoning: str = Field(
            description="Reasoning for yes or no and the next area selection in one brief sentence.")
        next_area: dynamic_enum = Field(
            description="The next area where the character should go.")
    return ActionArenaLocation


def action_area_locations(name: str, current_area: str, current_sector: str, sector: str, sector_arenas: str, action_description: str) -> str:
    areas = Enum("Areas", {arena: arena for arena in sector_arenas.split(", ")})
    model = model_from_enum(areas)

    structured_llm = llm.with_structured_output(model)

    content = template.format(name=name, current_area=current_area, current_sector=current_sector, sector=sector, sector_arenas=sector_arenas, action_description=action_description)

    action_arena_location = structured_llm.invoke([HumanMessage(content=content)])
    structured_llm.with_retry(stop_after_attempt=3).invoke([HumanMessage(content=content)])
    return action_arena_location.next_area.value


if __name__ == "__main__":
    print(action_area_locations(name="John Doe",
                                current_area="common room",
                                current_sector="John Doe's apartment",
                                sector="Hobbs Cafe",
                                sector_arenas="kitchen, bedroom, bathroom",
                                action_description="Putting on trousers"))
    print(action_area_locations(name="John Doe",
                                current_area="common room",
                                current_sector="John Doe's apartment",
                                sector="Hobbs Cafe",
                                sector_arenas="kitchen, bedroom, bathroom",
                                action_description="Getting coffee"))