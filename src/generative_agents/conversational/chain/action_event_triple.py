import json
import re
from pydantic import BaseModel
from generative_agents import global_state
from generative_agents.conversational.llm import llm
from langchain.chains import LLMChain
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, ChatPromptTemplate

system = """You follow the tasks given by the user as close as possible. You will only generate 1 JSON object as mentioned below.

Output format: Output a valid json of the following format:
{{
    "sentence": "<sentence to extract the subject, predicate, and object from>",
    "subject": "<subject usually appears before the verb. For example, in 'The cat sleeps,' 'the cat' is the subject.>",
    "predicate": "<predicate in a sentence is everything that describes what the subject is doing or being, including the verb and often additional information like objects or adverbs. For example, in 'Birds sings a song,' 'sing' is the predicate.>",
    "object": "<object in a sentence is the noun or pronoun that receives the action of the verb, as in 'She threw the ball,' where 'the ball' is the object.>"
}}"""

user_shot_1 = """Task: Given a sentence identify the subject, predicate, and object from the sentence.
Sentence: Dolores Murphy is eating breakfast."""
ai_shot_1 = """{{
    "subject": "Dolores Murphy",
    "predicate": "eat",
    "object": "breakfast"
}}"""

user_shot_2 = """Task: Given a sentence, identify the subject, predicate, and object from the sentence.
Sentence: Michael Bernstein is writing email on a computer"""
ai_shot_2 = """{{
    "subject": "Michael Bernstein",
    "predicate": "write",
    "object": "email"
}}"""

user_shot_3 = """Task: Given a sentence, identify the subject, predicate, and object from the sentence.
Sentence: {name} is {action_description}"""


class ActionEventTriple(BaseModel):
    name: str
    address: str = None
    action_description: str

    async def run(self):
        chat_template = ChatPromptTemplate(messages=[
            SystemMessagePromptTemplate.from_template(system),
            HumanMessagePromptTemplate.from_template(user_shot_1),
            AIMessagePromptTemplate.from_template(ai_shot_1),
            HumanMessagePromptTemplate.from_template(user_shot_2),
            AIMessagePromptTemplate.from_template(ai_shot_2),
            HumanMessagePromptTemplate.from_template(user_shot_3)])

        for i in range(5):
            _action_event_triple_chain = LLMChain(prompt=chat_template, llm=llm, llm_kwargs={
                "max_tokens": 45,
                "top_p": 0.95,
                "temperature": 0.4},
                verbose=global_state.verbose)

            completion = await _action_event_triple_chain.ainvoke(input={"name": self.name, "action_description": self.action_description})

            try:
                json_object = json.loads(completion["text"])

                subject = json_object["subject"] if not self.address else self.address

                object_ = json_object["object"] if type(
                    json_object["object"]) == str else json_object["object"][0]
                return (subject, json_object["predicate"], object_)
            except:
                pass

        print("Unable to generate action event triple.")
        return self.address or self.name, self.action_description, "idle"


if __name__ == "__main__":
    import asyncio
    t = asyncio.run(ActionEventTriple(
        name="John Doe", action_description="John Doe is taking a warm shower").run())
    print(t)
