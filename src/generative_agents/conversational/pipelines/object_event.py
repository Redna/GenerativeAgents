from enum import Enum
from pydantic import BaseModel, Field

from generative_agents.conversational.pipelines.grammar_llm_pipeline import grammar_pipeline

template = """You are {{name}}. You are interacting with the environment. You need to determine the state of an object that is being used by someone.

What is {{object_name}}'s state when {{name}} is using it for "{{action_description}}"?
"""

class ObjectState(BaseModel):
    state: str = Field(
        description="The state of the object that is being used by someone.")

def describe_object_state(name: str, object_name: str, action_description: str) -> str:
    object_state = grammar_pipeline.run(model=ObjectState, prompt_template=template, template_variables={
        "name": name,
        "object_name": object_name,
        "action_description": action_description
    })

    return object_state.state

if __name__ == "__main__":
    print(describe_object_state(name="John Doe",
                                object_name="kitchen sink",
                                action_description="washing dishes"))
    print(describe_object_state(name="Alex Smith",
                                object_name="bicycle",
                                action_description="repairing the broken chain"))
    print(describe_object_state(name="Emma Johnson",
                                object_name="bookshelf",
                                action_description="assembling a new wooden bookshelf"))
    print(describe_object_state(name="Sophie Lee",
                                object_name="laptop",
                                action_description="cleaning the dust off the laptop's keyboard and screen"))