

from llm import llm
from langchain import LLMChain, PromptTemplate

_template = """
{agent_identity}

In general, {agent_lifestyle}
{agent_name}'s wake up hour:
"""

_prompt = PromptTemplate(input_variables=["agent_name", 
                                          "agent_lifestyle",
                                          "agent_identity"],
                         template=_template)

wake_up_hour_chain = LLMChain(prompt=_prompt, llm=llm)
