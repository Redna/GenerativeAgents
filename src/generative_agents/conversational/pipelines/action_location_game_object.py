from enum import Enum
from typing import Type
from pydantic import BaseModel, Field

from generative_agents.conversational.pipelines.grammar_llm_pipeline import grammar_pipeline

template = """Your task is to identify the next object for an action. You need to output valid JSON.
Current activity: {action_description}
Objects available: [{available_objects}]
Which object is the most relevant one, you MUST pick one?
"""


def model_from_enum(dynamic_enum: Enum) -> Type[BaseModel]:
    class ActionObjectLocation(BaseModel):
        next_object: dynamic_enum = Field(
            description="The next object the character should use.")
    return ActionObjectLocation

def action_location_game_object(action_description: str, available_objects: str) -> str:
    possible_objects = available_objects.split(", ")
    objects = Enum("Objects", possible_objects)
    model = model_from_enum(objects)

    action_object_location = grammar_pipeline.run(model=model, prompt_template=template, template_variables={
        "action_description": action_description,
        "available_objects": available_objects
    })

    return action_object_location.next_object

if __name__ == "__main__":
    print(action_location_game_object(action_description="napping",
                                      available_objects="bed, easel, closet, painting"))
    print(action_location_game_object(action_description="putting on a skirt",
                                        available_objects="easel, closet, sink, microwave"))
    print(action_location_game_object(action_description="getting milk",
                                        available_objects="stove, sink, fridge, counter"))