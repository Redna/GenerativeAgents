import asyncio
import json
from pydantic import BaseModel
from generative_agents import global_state
from generative_agents.conversational.llm import llm
from langchain import LLMChain, PromptTemplate
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, ChatPromptTemplate

system = """You follow the tasks given by the user as close as possible. You will only generate 1 JSON object as mentioned below.
Output format:
{{
    "emojis": "<valid emoji>"
}}
"""

user_shot_1 = """Task: Provide one or two emoji that best represents the following statement or emotion: Taking a shower"""

ai_shot_1 = """{{
    "emoji": "🚿"
}}"""

user_shot_2 = """Task: Provide one or two emoji that best represents the following statement or emotion: Going for lunch"""
ai_shot_2 = """{{
    "emoji": "🍔"
}}"""

user = """Task: Provide one or two emoji that best represents the following statement or emotion: {action_description}"""


class ActionPronunciatio(BaseModel):
    action_description: str

    async def run(self):

        chat_template = ChatPromptTemplate(messages=[
            SystemMessagePromptTemplate.from_template(system),
            HumanMessagePromptTemplate.from_template(user_shot_1),
            AIMessagePromptTemplate.from_template(ai_shot_1),
            HumanMessagePromptTemplate.from_template(user_shot_2),
            AIMessagePromptTemplate.from_template(ai_shot_2),
            HumanMessagePromptTemplate.from_template(user)])

        for i in range(5):
            _action_pronunciatio_chain = LLMChain(prompt=chat_template, llm=llm, llm_kwargs={
                "max_tokens": 50,
                "top_p": 0.90,
                "temperature": 0.4,
            }, verbose=global_state.verbose)

            completion = await _action_pronunciatio_chain.ainvoke(input={"action_description": self.action_description})

            try:
                json_object = json.loads(completion["text"])
                return json_object["emoji"]
            except Exception as e:
                pass

        return "🤷"


async def __tests():
    t = [ActionPronunciatio(action_description="Taking a shower").run(),
         ActionPronunciatio(action_description="Drinking").run(),
         ActionPronunciatio(action_description="Taking a bath").run(),
         ActionPronunciatio(action_description="Visiting a friend").run(),
         ActionPronunciatio(action_description="Walking around").run(),
         ActionPronunciatio(action_description="Going to the toilet").run(),]

    return await asyncio.gather(*t)

if __name__ == "__main__":
    t = asyncio.run(__tests())
    print(t)
