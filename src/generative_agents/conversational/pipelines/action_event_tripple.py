from enum import Enum
from typing import Type
from pydantic import BaseModel, Field

from generative_agents.conversational.pipelines.grammar_llm_pipeline import grammar_pipeline

template = """Given a sentence identify the subject, predicate, and object from the sentence.

Sentence: {{name}} is {{action_description}}"""

class X(Enum):
    XYZ: str = "XYZ"

class Z(BaseModel):
    x: X
    a: str = 5


def _model_from_predefined_subject(enum: Enum) -> Type[BaseModel]:
    class ActionEvent(BaseModel):
        subject: enum
        predicate: str = Field(description="The action being performed")
        object: str = Field(description="The entity that the action is being performed on")
    
    return ActionEvent


def action_event_triple(name: str, action_description: str, address: str = None, ) -> str:
    model = _model_from_predefined_subject(enum=Enum("Subject", {name: name}))

    action_event = grammar_pipeline.run(model=model, prompt_template=template, template_variables={
        "name": name,
        "action_description": action_description
    })

    if address:
        action_event.subject = address
    
    return (action_event.subject, action_event.predicate, action_event.object)

if __name__ == "__main__":
    print(action_event_triple(name="John Doe", action_description="John Doe is taking a warm shower"))