import asyncio
import json
import re
from pydantic import BaseModel
from generative_agents import global_state
from generative_agents.conversational.llm import llm
from langchain import LLMChain, PromptTemplate
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, ChatPromptTemplate

system = """You summarize a conversation in one sentence. You need to output valid JSON.

Output format:
{{
    "summary": "<summary of the conversation>"
}}"""

user_shot_1 = """Conversation:
---
Joe Walther: Hello, where are you going at this hour?
Johan Frido: Well, I am heading to the orchestra.
---
What is the summary of the conversation above?"""

ai_shot_1 = """{{
    "summary": "Johan Frido informs Joe Walther that they are going to the orchestra at an unusual hour."
}}"""
user = """Conversation:
---
{conversation}
---

What is the summary of the conversation above?"""


class ConversationSummary(BaseModel):
    conversation: str

    # TODO implement retryable decorator
    async def run(self):
        chat_template = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(system),
                HumanMessagePromptTemplate.from_template(user_shot_1),
                AIMessagePromptTemplate.from_template(ai_shot_1),
                HumanMessagePromptTemplate.from_template(user)])
        _conversation_summary_chain = LLMChain(prompt=chat_template, llm=llm, llm_kwargs={
            "max_tokens": 100,
            "top_p": 0.95,
            "temperature": 0.3}, verbose=global_state.verbose)

        completion = await _conversation_summary_chain.ainvoke(input={"conversation": self.conversation})

        try:
            json_object = json.loads(completion["text"])
            return json_object["summary"]
        except Exception as e:
            pass

        return "This is a conversation"


async def __tests():
    t = [ConversationSummary(conversation="Rudolf: Hello, how are you?\nJoanne: I am fine, thank you.").run(),
         ConversationSummary(conversation="""Joe Walther: Hello, did you hear about Jim's party?
Frodo Reimsi: No, tell me more. You mean Jimmy Fraser?
Joe Walther: Jim Knofi. He is giving a dinner party.
Frodo Reimsi: I did not know that.""").run()
         ]

    return await asyncio.gather(*t)

if __name__ == "__main__":
    t = asyncio.run(__tests())
    print(t)
