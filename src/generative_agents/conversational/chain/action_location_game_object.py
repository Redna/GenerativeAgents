import json
import random
import re
from pydantic import BaseModel
import yaml
from generative_agents import global_state
from generative_agents.conversational.llm import llm
from langchain import LLMChain, PromptTemplate
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, ChatPromptTemplate

system = """Your task is to identify the next object for an action. You need to output valid JSON.
Output format:
    ```yaml
    reason: <reason for the next object selection>
    object: <name of the object it MUST be one of the provided list>
    ```"""

user_shot_1 = """Current activity: sleep in bed
Objects available: [bed, easel, closet, painting]
Which object is the most relevant one, you MUST pick one"""

ai_shot_1 = """```yaml
reason: For sleeping in bed, the next object should be bed.
object: bed
```"""

user_shot_2 = """Current activity: painting
Objects available: [easel, closet, sink, microwave]
Which object is the most relevant one, you MUST pick one"""

ai_shot_2 = """```yaml
reason: For painting, the next object should be easel.
object: easel
```"""

user_shot_3 = """Current activity: cooking
Objects available: [stove, sink, fridge, counter]
Which object is the most relevant one, you MUST pick one"""

ai_shot_3 = """```yaml
reason: For cooking, the next object should be stove.
object: stove
```"""

user = """Current activity: {action_description}
Objects available: [{available_objects}]
Which object is the most relevant one, you MUST pick one?"""


class ActionLocationGameObject(BaseModel):
    action_description: str
    available_objects: str
    retry: str = 0

    async def run(self):
        chat_template = ChatPromptTemplate(messages=[
            SystemMessagePromptTemplate.from_template(system),
            HumanMessagePromptTemplate.from_template(user_shot_1),
            AIMessagePromptTemplate.from_template(ai_shot_1),
            HumanMessagePromptTemplate.from_template(user_shot_2),
            AIMessagePromptTemplate.from_template(ai_shot_2),
            HumanMessagePromptTemplate.from_template(user_shot_3),
            AIMessagePromptTemplate.from_template(ai_shot_3),
            HumanMessagePromptTemplate.from_template(user)])

        _action_location_game_object_chain = LLMChain(prompt=chat_template, llm=llm, llm_kwargs={
            "max_tokens": 140,
            "temperature": 0.3},
            verbose=global_state.verbose)

        possible_objects = [object_.strip(
        ) for object_ in self.available_objects.split(",") if object_.strip()]

        for i in range(5):

            completion = await _action_location_game_object_chain.ainvoke(input={"action_description": self.action_description, "available_objects": self.available_objects})

            pattern = r'```yaml(.*)```'
            match = re.search(pattern, completion["text"], re.DOTALL)
            if match:
                try:
                    object_ = yaml.safe_load(match.group(1))
                    if object_ in possible_objects:
                        return object_
                except Exception as e:
                    pass

        object_ = possible_objects[-1]  # random.choice(possible_objects)
        print(f"Unable to identify next object. Selecting randomly: {object_}")
        return object_


async def __tests():
    tests = [
        ActionLocationGameObject(action_description="napping",
                                 available_objects="bed, easel, closet, painting").run(),
        ActionLocationGameObject(action_description="putting on a skirt",
                                 available_objects="easel, closet, sink, microwave").run(),
        ActionLocationGameObject(action_description="getting milk",
                                 available_objects="stove, sink, fridge, counter").run(),
    ]

    return await asyncio.gather(*tests)

if __name__ == "__main__":
    import asyncio
    t = asyncio.run(__tests())
    print(t)
