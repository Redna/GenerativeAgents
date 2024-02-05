import re
from pydantic import BaseModel
from generative_agents import global_state
from generative_agents.conversational.llm import llm
from langchain import LLMChain, PromptTemplate
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate

system = "Your task is write a concise description about {agent}'s personality, family situation and characteristics. You include ALL the details provided in the given context (you MUST include all the names of persons, ages,...)."
user = """
Context:
{identity}

Who is {agent}?
"""

class Identity(BaseModel):
    agent: str
    identity: str

    async def run(self):
        chat_template = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(system),
                HumanMessagePromptTemplate.from_template(user)])
        _identity_chain = LLMChain(prompt=chat_template, llm=llm, verbose=global_state.verbose)

        completion = await _identity_chain.ainvoke(input={"agent":self.agent, "identity":self.identity})
        return completion["text"]


if __name__ == "__main__":
    import asyncio
    asyncio.run(Identity(agent="agent", identity="identity").run())