from enum import Enum
from pydantic import BaseModel, Field

from generative_agents.conversational.pipelines.grammar_llm_pipeline import grammar_pipeline

template = """You are {{name}}. You are interacting with the environment. You need to determine the state of an object that is being used by someone.

What is {{object_name}}'s state when {{name}} is using it for "{{action_description}}"?"""

class ObjectState(BaseModel):
    state: str = Field(description="The new state of the object when it has been used. This is always filled in.")

def describe_object_state(name: str, object_name: str, object_address: str, action_description: str) -> str:
    object_state = grammar_pipeline.run(model=ObjectState, prompt_template=template, template_variables={
        "name": name,
        "object_name": object_name,
        "action_description": action_description
    })

    return f"{object_name} is {object_state.state}""", (object_address, "is", object_state.state)

if __name__ == "__main__":            
    print(describe_object_state(name="John Doe",
                                object_name="kitchen sink",
                                object_address="John Doe's house:kitchen sink",
                                action_description="washing dishes"))
    print(describe_object_state(name="Alex Smith",
                                object_name="bicycle",
                                object_address="Alex's garage:bicycle",
                                action_description="repairing the broken chain"))
    print(describe_object_state(name="Emma Johnson",
                                object_name="bookshelf",
                                object_address="Emma's study room:bookshelf",
                                action_description="assembling a new wooden bookshelf"))
    print(describe_object_state(name="Sophie Lee",
                                object_name="laptop",
                                object_address="Sophie's office desk:laptop",
                                action_description="cleaning the dust off the laptop's keyboard and screen"))