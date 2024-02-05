import asyncio
import json
import re
from langchain import LLMChain, PromptTemplate
from pydantic import BaseModel
import yaml
from generative_agents import global_state

from generative_agents.conversational.llm import llm

from langchain.chains import LLMChain
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, ChatPromptTemplate


system = """You are in a roleplay game and act as an agent. Your task is to estimate the wake up hour of an agent.

Output format: Output a valid yaml of the following format:
```yaml
rationale: <maximum one sentence reason for the wake up hour>
wake_up_hour: "<time in 12-hour clock format>"
```"""

user_shot_1 = """You are John Doe. Your identity is: 
John Doe is a 30 year old software developer. He enjoys coding and is a coffee enthusiast. John loves reading sci-fi novels and playing chess in his free time. He is an early bird and enjoys the quiet mornings.. 

John Doe loves the morning as he feels super relaxed..
    
When does John Doe wake up today?"""

ai_shot_1 = """```yaml
rationale: John Doe is an early bird who enjoys the quiet mornings. This suggests that he might wake up early to start his day peacefully and have enough time for his hobbies before work.
wake_up_hour: "6:00 AM"
```"""

user = """You are {{agent_name}}. Your identity is: 
{{agent_identity}}. 

{{agent_lifestyle}}.
    
When does {{agent_name}} wake up today?"""

chat_template = ChatPromptTemplate(messages=[
    SystemMessagePromptTemplate.from_template(
        system, template_format="jinja2"),
    HumanMessagePromptTemplate.from_template(user_shot_1),
    AIMessagePromptTemplate.from_template(ai_shot_1),
    HumanMessagePromptTemplate.from_template(user, template_format="jinja2")])


class WakeUpHour(BaseModel):
    agent_name: str
    agent_identity: str
    agent_lifestyle: str

    async def run(self):
        for i in range(5):
            _wake_up_hour_chain = LLMChain(prompt=chat_template, llm=llm, llm_kwargs={"max_tokens": 200,
                                                                                      "top_p": 0.95,
                                                                                      "temperature": 0.4}, verbose=True)

            completion = await _wake_up_hour_chain.ainvoke(input={"agent_name": self.agent_name,
                                                                  "agent_identity": self.agent_identity,
                                                                  "agent_lifestyle": self.agent_lifestyle})

            pattern = r'```yaml(.*)```'
            match = re.search(pattern, completion["text"], re.DOTALL)
            if match:
                try:
                    output = yaml.safe_load(match.group(1))
                    # verify 12-hour clock format
                    match = re.search(r"\d{1,2}:\d{2} [AP]M", output["wake_up_hour"])
                    if match:
                        return match.group()
                    else:
                        continue
                except:
                    pass

        print("Unable to generate the wake up hour")
        return "6:00 AM"


async def __tests():
    t = [
        WakeUpHour(agent_name="Emily Clark",
                   agent_identity="Emily Clark is a 25 year old freelance graphic designer. She adores creating vibrant and unique designs. Emily is an avid traveler, seeking inspiration from different cultures around the world. She prefers working late nights to capture the essence of her travels in her designs.",
                   agent_lifestyle="Emily finds her creativity peaks in the quiet of the night.").run(),
        WakeUpHour(agent_name="Michael Thompson",
                   agent_identity="Michael Thompson is a 40 year old professional photographer. He specializes in wildlife photography and spends months in remote locations capturing the beauty of nature. Michael is an adventure seeker and enjoys hiking and camping.",
                   agent_lifestyle="Michael is accustomed to waking up at dawn to catch the perfect light for his photographs.").run(),
        WakeUpHour(agent_name="Linda Wu",
                   agent_identity="Linda Wu is a 35 year old entrepreneur running her own start-up. She is dedicated to creating eco-friendly products. Linda is passionate about sustainability and environmental conservation. She practices yoga daily to stay focused and energized.",
                   agent_lifestyle="Linda is an early riser, finding the morning the best time to plan her day and meditate.").run(),
        WakeUpHour(agent_name="Sophie Patel",
                   agent_identity="Sophie Patel is a 32 year old chef who specializes in fusion cuisine, blending flavors from her Indian heritage with international cuisines. Sophie loves experimenting with new recipes and spices. She hosts cooking classes and shares her culinary adventures on her blog.",
                   agent_lifestyle="Sophie believes that the early morning is the best time to explore fresh produce at the market, setting her alarm before sunrise.").run(),
    ]

    return await asyncio.gather(*t)

if __name__ == "__main__":
    from pprint import pprint
    t = asyncio.run(__tests())
    pprint(t)
