from pydantic import BaseModel, Field
from typing import TypedDict
from typing import Annotated

from langchain_groq.chat_models import ChatGroq
from langchain_core.messages import HumanMessage

llm = ChatGroq(model="llama3-8b-8192", name="object_event")

template = """You are {name}. You are interacting with the environment. You need to determine the state of an object that is being used by someone.

Determine "{object_name}" state when {name} is using it for "{action_description}"?"""

class ObjectState(TypedDict):
    """
    State of an object after it has been used.
    """
    state: Annotated[str, ..., "The new state of the object when it has been used."]

def describe_object_state(name: str, object_name: str, object_address: str, action_description: str) -> str:
    structured_llm = llm.with_structured_output(ObjectState)

    content = template.format(name=name, object_name=object_name, action_description=action_description)
    object_state = structured_llm.with_retry(stop_after_attempt=3).invoke([HumanMessage(content=content)])

    return f"{object_name} is {object_state['state']}""", (object_address, "is", object_state['state'])

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