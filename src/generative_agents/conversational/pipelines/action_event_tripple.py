from enum import Enum
from typing import Annotated, Type, TypedDict
from pydantic import BaseModel

from langchain_groq.chat_models import ChatGroq
from langchain_core.messages import HumanMessage

llm = ChatGroq(model="llama3-8b-8192",
               name="action_event_triple")

template = """Given a sentence identify the subject, predicate, and object from the sentence.

Examples:
1. She eats an apple.
subject: "She"
predicate: "eats"
object: "an apple"

2. The cat is sleeping.
subject: "The cat"
predicate: "is sleeping"
object: ""

3. The children gave mother a present.
subject: "The children"
predicate: "gave"
object: "mother a present"

4. Martha baked a cake for Polly.
subject: "Martha"
predicate: "baked"
object: "a cake for Polly"

5. (Paul Rain is) Morning routine, including a quick breakfast and a walk around the village to clear my mind
subject: "Paul Rain"
predicate: "is doing"
object: "morning routine, including a quick breakfast and a walk around the village to clear my mind"

6. (John Doe is) have a leisurely breakfast and catch up on some reading
subject: "John Doe"
predicate: "have"
object: "a leisurely breakfast and catch up on some reading"

Sentence: ({name} is) {action_description}"""

def action_event_triple(name: str, action_description: str, address: str = None) -> str:
    class SubjectPredicateObject(TypedDict):
        """
        Contain the subject, predicate, and object from a sentence.
        """
        subject: Annotated[str, ... , "Performs the action in the sentence. It is usually a noun or pronoun."]
        predicate: Annotated[str, ..., "The verb and information about the subject."]
        object: Annotated[str, ..., "receives or is affected by the action of the subject."]

    structured_llm = llm.with_structured_output(SubjectPredicateObject)

    content = template.format(name=name, action_description=action_description)

    spo = structured_llm.invoke([HumanMessage(content=content)])


    if address:
        spo["subject"] = address

    return (spo["subject"], spo["predicate"], spo["object"])

if __name__ == "__main__":
    print(action_event_triple(name="John Doe", action_description="John Doe is taking a warm shower"))
    print(action_event_triple(name="Maggie", action_description="Maggie is playing with her toys"))
    print(action_event_triple(name="Joaquin", action_description="Joaquin and his friends are playing soccer"))
    print(action_event_triple(name="John Doe", action_description="He is runnning"))
    print(action_event_triple(name="John Doe", action_description="Sara is about to go to the park"))