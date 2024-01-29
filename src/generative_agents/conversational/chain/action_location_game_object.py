import json
import random
import re
from pydantic import BaseModel
from generative_agents import global_state
from generative_agents.conversational.llm import llm
from langchain import LLMChain, PromptTemplate
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, ChatPromptTemplate

system = """Your task is to identify the next object for an action. You need to output valid JSON.
Output format:
{{
    "reason": "<reason for the next object selection>",
    "object": "<name of the object>"
}}"""

user_shot_1 = """Current activity: sleep in bed
Objects available: [bed, easel, closet, painting]
Which object is the most relevant one?"""

ai_shot_1 = """{{
    "reason": "For sleeping in bed, the next object should be bed.",
    "object": "bed"
}}"""

user_shot_2 = """Current activity: painting
Objects available: [easel, closet, sink, microwave]
Which object is the most relevant one?"""

ai_shot_2 = """{{
    "reason": "For painting, the next object should be easel.",
    "object": "easel"
}}"""

user_shot_3 = """Current activity: cooking
Objects available: [stove, sink, fridge, counter]
Which object is the most relevant one?"""

ai_shot_3 = """{{
    "reason": "For cooking, the next object should be stove.",
    "object": "stove"
}}"""

user_shot_4 = """Current activity: {action_description}
Objects available: [{available_objects}]
Which object is the most relevant one?"""

chat_template = ChatPromptTemplate(messages=[
        SystemMessagePromptTemplate.from_template(system),
        HumanMessagePromptTemplate.from_template(user_shot_1),
        AIMessagePromptTemplate.from_template(ai_shot_1),
        HumanMessagePromptTemplate.from_template(user_shot_2),
        AIMessagePromptTemplate.from_template(ai_shot_2),
        HumanMessagePromptTemplate.from_template(user_shot_3),
        AIMessagePromptTemplate.from_template(ai_shot_3),
        HumanMessagePromptTemplate.from_template(user_shot_4)])


class ActionLocationGameObject(BaseModel):
    action_description: str
    available_objects: str
    retry: str = 0

    async def run(self): 

      _action_location_game_object_chain = LLMChain(prompt=chat_template, llm=llm, llm_kwargs={
                                                        "max_tokens": 50,
                                                        "temperature": 0.3},
                                                        verbose=global_state.verbose)
      
      possible_objects = [object_.strip() for object_ in self.available_objects.split(",") if object_.strip()]

      for i in range(5):

        completion = await _action_location_game_object_chain.ainvoke(input={"action_description": self.action_description, "available_objects": self.available_objects})
        
        try:
          json_object = json.loads(completion["text"])
          object_ = json_object["object"]
          if object_ in possible_objects:
            return object_
        except Exception as e:
          pass

      object_ = possible_objects[-1] # random.choice(possible_objects)
      print(f"Unable to identify next object. Selecting randomly: {object_}")
      return object_


async def __tests():
  tests = [
    ActionLocationGameObject(action_description="napping", available_objects="bed, easel, closet, painting").run(),
    ActionLocationGameObject(action_description="putting on a skirt", available_objects="easel, closet, sink, microwave").run(),
    ActionLocationGameObject(action_description="getting milk", available_objects="stove, sink, fridge, counter").run(),
  ]

  return await asyncio.gather(*tests)

if __name__ == "__main__":
    import asyncio
    t = asyncio.run(__tests())
    print(t)

