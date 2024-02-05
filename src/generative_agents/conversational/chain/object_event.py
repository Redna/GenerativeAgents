import asyncio
import json
import re
from pydantic import BaseModel
import yaml
from generative_agents import global_state
from generative_agents.conversational.llm import llm
from langchain.chains import LLMChain
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, ChatPromptTemplate


system = """You are {{name}}. You are interacting with the environment. You need to determine the state of an object that is being used by someone.
Output format: Output a valid yaml of the following format:
```yaml
reasoning: <reasoning for the new object state>
new_object_state: <state of an object, should be one single word>
```"""

user_shot_1 = """What is shower's state when John Doe is using it for "getting ready for work"?"""
ai_shot_1 = """
```yaml
reasoning: John Doe is getting ready to work and using the shower. Hence, the shower is on.
new_object_state: "on"
```"""

user = """What is {{object_name}}'s state when {{name}} is using it for "{{action_description}}"?"""


class ObjectActionDescription(BaseModel):
    name: str
    object_name: str
    object_address: str
    action_description: str

    async def run(self):
        chat_template = ChatPromptTemplate(messages=[
            SystemMessagePromptTemplate.from_template(
                system, template_format="jinja2"),
            HumanMessagePromptTemplate.from_template(user_shot_1),
            AIMessagePromptTemplate.from_template(ai_shot_1),
            HumanMessagePromptTemplate.from_template(user, template_format="jinja2")])

        for i in range(5):
            _action_object_chain = LLMChain(prompt=chat_template, llm=llm, llm_kwargs={
                "max_tokens": 90,
                "top_p": 0.95,
                "temperature": 0.4},
                verbose=global_state.verbose)

            completion = await _action_object_chain.ainvoke(input={"object_name": self.object_name,
                                                                   "name": self.name,
                                                                   "action_description": self.action_description})

            pattern = r'```yaml(.*)```'
            match = re.search(pattern, completion["text"], re.DOTALL)
            if match:
                try:
                    output = yaml.safe_load(match.group(1))
                    state = output["new_object_state"]
                    return f"{self.object_name} is {state}""", (self.object_address, "is", state)
                except Exception as error:
                    pass

        print("Unable to generate action event triple.")
        return f"{self.object_name} is idle""", (self.object_address, "is", "idle")


async def __tests():
    t = [ObjectActionDescription(name="John Doe",
                                 object_name="kitchen sink",
                                 object_address="John Doe's house:kitchen sink",
                                 action_description="washing dishes").run(),
         ObjectActionDescription(name="Alex Smith",
                                 object_name="bicycle",
                                 object_address="Alex's garage:bicycle",
                                 action_description="repairing the broken chain").run(),
         ObjectActionDescription(name="Emma Johnson",
                                 object_name="bookshelf",
                                 object_address="Emma's study room:bookshelf",
                                 action_description="assembling a new wooden bookshelf").run(),
         ObjectActionDescription(name="Sophie Lee",
                                 object_name="laptop",
                                 object_address="Sophie's office desk:laptop",
                                 action_description="cleaning the dust off the laptop's keyboard and screen").run()
         ]

    return await asyncio.gather(*t)

if __name__ == "__main__":
    from pprint import pprint
    t = asyncio.run(__tests())
    pprint(t)
