import asyncio
import json
import re
from pydantic import BaseModel
from generative_agents import global_state
from generative_agents.conversational.llm import llm
from langchain.chains import LLMChain
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, ChatPromptTemplate

system = """You will act as {init_agent}. Based on a given scenario you whether to initiate a conversation or not. You will only generate exactly 1 valid JSON object as mentioned below.
Output format: Output a valid json of the following format:
{{
    "reasonForDecision": "[reason for initiating or not initiating a conversation]",
    "answer": "[yes or no]"
}}
"""

user = """Context: {context}

Right now, it is {current_time}. {init_agent} and {agent_with} {last_chat_summary}.
{init_agent} is currently {init_agent_observation}
{agent_with_observation}

Would {init_agent} initiate a conversation with {agent_with}?"""

user_shot_1 = """Context: You are in the supermarket. Buying some groceries. You see your friend, John, in the same aisle as you.

Right now, it is 5:00 PM. Jaiden Smith and John Doe last chatted a month ago about a movie..
Jaiden Smith is currently reading the label on a cereal box.
deep in thought, looking at different kinds of tea.

Would Jaiden Smith initiate a conversation with John Doe?"""

agent_shot_1 = """{{
    "reasonForDecision": "Jaiden Smith is deep in thought, looking at different kinds of tea and might not immediately notice John Doe. However, considering they are friends and haven\'t communicated for a month, it could be a good opportunity to reconnect. Also, it\'s early evening which could imply that people are usually more sociable around this time.",
    "answer": "yes"
}}"""

chat_template = ChatPromptTemplate(messages=[
        SystemMessagePromptTemplate.from_template(system),
        HumanMessagePromptTemplate.from_template(user_shot_1),
        AIMessagePromptTemplate.from_template(agent_shot_1),
        HumanMessagePromptTemplate.from_template(user)],
)

class DecideToTalk(BaseModel):
    context: str
    current_time: str
    init_agent: str
    agent_with: str
    last_chat_summary: str
    init_agent_observation: str
    agent_with_observation: str

    async def run(self):

        _decide_to_talk_chain = LLMChain(prompt=chat_template, llm=llm, llm_kwargs={
            "max_tokens": 350,
            "top_p": 0.95,
            "temperature": 0.8,}
            , verbose=True)


        tasks = []
        for i in range(3):
            completion = await _decide_to_talk_chain.ainvoke(input={"context": self.context,
                                                            "current_time": self.current_time,
                                                            "init_agent": self.init_agent,
                                                            "agent_with": self.agent_with,
                                                            "last_chat_summary": self.last_chat_summary,
                                                            "init_agent_observation": self.init_agent_observation,
                                                            "agent_with_observation": self.agent_with_observation})

            pattern = r'\{.*?\}'
            match = re.search(pattern, completion["text"], re.DOTALL)
            if match:
                try:
                    json_object = json.loads(match.group(0))
                    return "yes" in json_object["answer"].lower()
                except:
                    pass
                
        print("Unable to decide to talk.")
        return False


async def __tests():
    t = [
        DecideToTalk(context="You are in the Pharmacy. Buying some medicines. You see your friend, Jaiden Suave, in the same aisle as you.",
                    current_time="7:00 PM",
                    init_agent="Kaitlyn Smith",
                    agent_with="Jaiden Suave",
                    last_chat_summary="last chattet at 03:00pm about the good cafe at Hobbs Cafe.",
                    init_agent_observation="looking for some painkillers.",
                    agent_with_observation="buying pavements for her son").run(),
        DecideToTalk(
            context="You are in the supermarket. Buying some groceries. You see your friend, John, in the same aisle as you.",
            current_time="5:00 PM",
            init_agent="Jaiden Smith",
            agent_with="John Doe",
            last_chat_summary="last chatted at 4:30 PM about weekend plans.",
            init_agent_observation="comparing prices of pasta sauces.",
            agent_with_observation="seems in a hurry, picking up items quickly."
        ).run(),
        DecideToTalk(
            context="You are in the supermarket. Buying some groceries. You see your friend, John, in the same aisle as you.",
            current_time="5:00 PM",
            init_agent="Jaiden Smith",
            agent_with="John Doe",
            last_chat_summary="last chatted a month ago about a movie.",
            init_agent_observation="reading the label on a cereal box.",
            agent_with_observation="deep in thought, looking at different kinds of tea."
        ).run(),
        DecideToTalk(
            context="You are in the supermarket. Buying some groceries. You see John Doe, in the same aisle as you.",
            current_time="5:00 PM",
            init_agent="Jaiden Smith",
            agent_with="John Doe",
            last_chat_summary="never chatted before.",
            init_agent_observation="looking for a new brand of coffee.",
            agent_with_observation="talking on the phone, seems engaged in the conversation."
        ).run()
    ]

    return await asyncio.gather(*t)

if __name__ == "__main__":
    t = asyncio.run(__tests())
    print(t)