import asyncio
import re
from pydantic import BaseModel
from generative_agents import global_state
from generative_agents.conversational.llm import llm

from langchain.chains import LLMChain
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, ChatPromptTemplate


system = """You are {agent}. You will write about the personality and observations of {agent} based on a given event and related events."""

user = """Context
{identity}
{agent} perceived the following event: {event_description}
He remembered the following related events: {events}
He thought the following about the event: {thoughts}

What is {agent}'s personality and {agent}'s observations?"""


class FocusedEventToContext(BaseModel):
    agent: str
    identity: str
    event_description: str
    events: str
    thoughts: str

    async def run(self):
        chat_template = ChatPromptTemplate(messages=[
            SystemMessagePromptTemplate.from_template(system),
            HumanMessagePromptTemplate.from_template(user)])

        _focused_event_to_context = LLMChain(prompt=chat_template, llm=llm, llm_kwargs={
            "max_tokens": 200,
            "top_p": 0.92,
            "temperature": 0.8},
            verbose=global_state.verbose)

        completion = await _focused_event_to_context.ainvoke(input={"agent": self.agent,
                                                                    "identity": self.identity,
                                                                    "event_description": self.event_description,
                                                                    "events": self.events,
                                                                    "thoughts": self.thoughts})
        return completion["text"]


async def __tests():
    t = [
        FocusedEventToContext(agent="John", identity="John is a 22 year old student, who is learning a lot. He likes discussing with his peers. He is a social person.",
                              event_description="John is going to the library to study for his exams.",
                              events="The library is full of students. John is studying for his exams. John is a social person. John is a student. John is learning a lot. John likes discussing with his peers.",
                              thoughts="John is a social person. The exam is important. He hopes to meet his friends at the library. He is excited to study.").run(),
    ]

    return await asyncio.gather(*t)

if __name__ == "__main__":
    from pprint import pprint
    t = asyncio.run(__tests())
    pprint(t)
