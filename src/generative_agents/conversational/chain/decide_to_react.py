import asyncio
import json
import re
from pydantic import BaseModel
from generative_agents import global_state
from generative_agents.conversational.llm import llm
from langchain.chains import LLMChain
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, ChatPromptTemplate

system = """You will act as {agent}. Based on a given scenario you decide between two options. You will only generate exactly 1 valid JSON object as mentioned below.

Output format: Output a valid json of the following format:
{{
    "reasonForOptionSelection": "[reasoning for selecting one option]",
    "answer": "[fill in <Option 1> or <Option 2>]"
}}

"""

user = """Context: {context}
Right now, it is {current_time}. 
{agent} is {agent_observation} when {agent} saw {agent_with} in the middle of {agent_with_observation}.",

Let's think step by step. Of the following three options, what should {agent} do?
- Option 1: Wait on {initial_action_description} until {agent_with} is done {agent_with_action}
- Option 2: Continue on to {initial_action_description} now"""


class DecideToReact(BaseModel):
    context: str
    current_time: str
    agent: str
    agent_with: str
    agent_with_action: str
    agent_observation: str
    agent_with_observation: str
    initial_action_description: str

    async def run(self):
        tasks = []
        chat_template = ChatPromptTemplate(messages=[
            SystemMessagePromptTemplate.from_template(system),
            HumanMessagePromptTemplate.from_template(user)])
        for i in range(3):
            _decide_to_react_chain = LLMChain(prompt=chat_template, llm=llm, llm_kwargs={
                "max_tokens": 400,
                "top_p": 0.90,
                "temperature": 0.8}, verbose=True)
            completion = await _decide_to_react_chain.ainvoke(input={"context": self.context,
                                                                     "current_time": self.current_time,
                                                                     "agent": self.agent,
                                                                     "agent_with": self.agent_with,
                                                                     "agent_with_action": self.agent_with_action,
                                                                     "agent_observation": self.agent_observation,
                                                                     "agent_with_observation": self.agent_with_observation,
                                                                     "initial_action_description": self.initial_action_description})

            pattern = r'{.*?\}'
            match = re.search(pattern, completion["text"], re.DOTALL)
            if match:
                try:
                    json_object = json.loads(match.group(0))
                    return 1 if "option 1" in json_object["answer"].lower() else 2
                except:
                    pass

        print("Unable to decide to react.")
        return 2


async def __tests():
    t = [
        DecideToReact(context="You are walking through the park going to Johns Pub.",
                      current_time="10:00 PM",
                      agent="William Strange",
                      agent_with="John Smith",
                      agent_with_action="Jumping up and down",
                      agent_observation="walking through the park",
                      agent_with_observation="Jumping up and down in the park",
                      initial_action_description="walking through the park going to Johns Pub").run(),
    ]

    return await asyncio.gather(*t)

if __name__ == "__main__":
    t = asyncio.run(__tests())
    print(t)
