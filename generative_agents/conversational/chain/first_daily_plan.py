from llm import llm
from langchain import LLMChain, PromptTemplate

_template = """
{agent_identity}

In general, {agent_lifestyle}
Today is {current_day}. Here is {agent_name}'s plan today in broad-strokes (with the time of the day. e.g., have a lunch at 12:00 pm, watch TV from 7 to 8 pm): 1) wake up and complete the morning routine at {wake_up_hour}, 2)
"""

_prompt = PromptTemplate(input_variables=["agent_name",
                                          "agent_identity",
                                          "agent_lifestyle",
                                          "current_day",
                                          "wake_up_hour"],
                            template=_template)

first_daily_plan_chain = LLMChain(prompt=_prompt, llm=llm)
