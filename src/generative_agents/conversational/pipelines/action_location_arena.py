from enum import Enum
from typing import Type
from pydantic import BaseModel, Field

from generative_agents.conversational.pipelines.grammar_llm_pipeline import grammar_pipeline

template = """Your task is to identify the next area for a character. It has to be one area of the provided list. You need to output valid JSON.

{{name}} is in the area "{{current_area}}" in "{{current_sector}}".
{{name}} is going to "{{sector}}" that has the following areas: [{{sector_arenas}}]
Stay in the current area if the activity can be done there. Never go into other people's rooms unless necessary.
For {{action_description}}, to which area should {{name}} go in "{{sector}}"?
"""


def model_from_enum(dynamic_enum: Enum) -> Type[BaseModel]:
    class ActionArenaLocation(BaseModel):
        reasoning: str = Field(
            description="Reasoning for yes or no and the next area selection in one brief sentence.")
        next_area: dynamic_enum = Field(
            description="The next area the character should go to.")

    return ActionArenaLocation


def action_area_locations(name: str, current_area: str, current_sector: str, sector: str, sector_arenas: str, action_description: str) -> str:
    possible_areas = sector_arenas.split(", ")
    areas = Enum("Areas", possible_areas)
    model = model_from_enum(areas)

    action_arena_location = grammar_pipeline.run(model=model, prompt_template=template, template_variables={
        "name": name,
        "current_area": current_area,
        "current_sector": current_sector,
        "sector": sector,
        "sector_arenas": sector_arenas,
        "action_description": action_description
    })

    return action_arena_location.next_area


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