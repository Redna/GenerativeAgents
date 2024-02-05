import asyncio
import re
from pydantic import BaseModel
import yaml
from generative_agents import global_state
from generative_agents.conversational.llm import llm

from langchain.chains import LLMChain
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, ChatPromptTemplate


system = """You are {{agent_name}}. You are thinking about the importance of an event.
Output format: 
```yaml
rationale: <one sentence rationale for the poignance rating>
poignance: <poignance rating>
```"""

user_shot_1 = """Here is a brief description of Ryan Ravi:
Ryan Ravy is a 25 year old. He is an artist and a very creative person. All his life he has been greatly influenced by his family and friends. He is a very kind and caring person.

On the scale of 1 to 10, where 1 is purely mundane (e.g., brushing teeth, making bed) and 10 is extremely poignant (e.g., a break up, college acceptance), rate the likely poignancy of the following Event for Ryan Ravi.

How would you rate the Event "Ryan Ravi is painting a picture."?"""

ai_shot_1 = """```yaml
rationale: Painting a picture is an activity that holds significant meaning for Ryan Ravi as it represents his passion and creative expression.
poignance: 6
```"""

user = """Here is a brief description of {{agent_name}}:
{{agent_identity}}

On the scale of 1 to 10, where 1 is purely mundane (e.g., brushing teeth, making bed) and \
10 is extremely poignant (e.g., a break up, college acceptance), rate the likely poignancy \
of the following {{type_}} for {{agent_name}}.

How would you rate the {{type_}} "{{description}}"?"""


class Poignance(BaseModel):
    agent_name: str
    agent_identity: str
    type_: str
    description: str

    async def run(self):
        chat_template = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(
                    system, template_format="jinja2"),
                HumanMessagePromptTemplate.from_template(user_shot_1),
                AIMessagePromptTemplate.from_template(ai_shot_1),
                HumanMessagePromptTemplate.from_template(user, template_format="jinja2")])

        poignance_chain = LLMChain(prompt=chat_template, llm=llm, llm_kwargs={"max_tokens": 200,
                                                                              "top_p": 0.95,
                                                                              "temperature": 0.1},
                                   verbose=True)
        completion = await poignance_chain.ainvoke(input={"agent_name": self.agent_name,
                                                      "agent_identity": self.agent_identity,
                                                      "type_": self.type_,
                                                      "description": self.description})

        pattern = r'```yaml(.*)```'
        match = re.search(pattern, completion["text"], re.DOTALL)
        if match:
            try:
                output = yaml.safe_load(match.group(1))
                return int(output["poignance"])
            except Exception as error:
                pass
        
        return 2


async def __tests():
    t = [
        Poignance(agent_name="Emily Tan",
                  agent_identity="Emily Tan is a renowned sculptor with over a decade of experience. Her works explore themes of nature and human connection. She has showcased her art in various galleries around the world.",
                  type_="Exhibition",
                  description="Emily Tan is preparing for her solo art exhibition at the local gallery.").run(),
        Poignance(agent_name="Marcus Lee",
                  agent_identity="Marcus Lee is a young, dynamic author known for his captivating novels that blend mystery with deep psychological insights. His storytelling has garnered a loyal following.",
                  type_="Book Launch",
                  description="Marcus Lee is hosting a book launch for his latest novel.").run(),
        Poignance(agent_name="Sara Ahmed",
                  agent_identity="Sara Ahmed is a dedicated social worker and community organizer. She has spent years working with underprivileged communities, focusing on education and healthcare initiatives.",
                  type_="Community Service",
                  description="Sara Ahmed is organizing a community service event.").run(),
        Poignance(agent_name="Leo Gonzalez",
                  agent_identity="Leo Gonzalez is an environmental activist known for his passionate advocacy for climate change action. He has been involved in numerous campaigns to promote sustainability.",
                  type_="Campaign",
                  description="Leo Gonzalez is leading a campaign to raise awareness about the importance of renewable energy sources.").run(),
        Poignance(agent_name="Nina Patel",
                  agent_identity="Nina Patel is a classical musician and violinist with an exceptional talent. She has performed in prestigious concert halls and collaborated with renowned orchestras.",
                  type_="Performance",
                  description="Nina Patel is rehearsing for her upcoming performance.").run()
    ]

    return await asyncio.gather(*t)

if __name__ == "__main__":
    from pprint import pprint
    t = asyncio.run(__tests())
    pprint(t)
