import re
from generative_agents.conversational.llm import llm
from langchain import LLMChain, PromptTemplate

from generative_agents.conversational.output_parser.fuzzy_parser import FuzzyOutputParser, PatternWithDefault

_template = """
{agent_identity}

In general, {agent_lifestyle}
Today is {current_day}. Here is {agent_name}'s plan today in broad-strokes (with the time of the day. e.g., have a lunch at 12:00 pm, watch TV from 7 to 8 pm): 1) wake up and complete the morning routine at {wake_up_hour}, 2)"""

_prompt = PromptTemplate(input_variables=["agent_name",
                                          "agent_identity",
                                          "agent_lifestyle",
                                          "current_day",
                                          "wake_up_hour"],
                            template=_template)

_outputs = {"first_daily_plan": PatternWithDefault(pattern=re.compile(r"(1\) wake up and complete.*)", re.IGNORECASE + re.MULTILINE), 
                                              parsing_function=lambda x: x,
                                              default="1) wake up and complete the morning routine at 6:00 AM 2) go to bed at 10:00 PM")}

_output_parser = FuzzyOutputParser(output_definitions=_outputs)

first_daily_plan_chain = LLMChain(prompt=_prompt, llm=llm, llm_kwargs={"max_length": 300,
                                                                   "do_sample": True,
                                                                   "top_p": 0.95,
                                                                   "top_k": 60,
                                                                   "temperature": 0.4}
                                                                   , output_parser=_output_parser, verbose=True)
