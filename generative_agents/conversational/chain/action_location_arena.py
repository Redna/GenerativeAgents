from llm import llm
from langchain import LLMChain, PromptTemplate

_template = """
Jane Anderson is in kitchen in Jane Anderson's house.
Jane Anderson is going to Jane Anderson's house that has the following areas: [kitchen,  bedroom, bathroom]
Stay in the current area if the activity can be done there. Never go into other people's rooms unless necessary.
For cooking, Jane Anderson should go to the following area in Jane Anderson's house:
Answer: [kitchen]
---
Tom Watson is in common room in Tom Watson's apartment.
Tom Watson is going to Hobbs Cafe that has the following areas: [cafe]
Stay in the current area if the activity can be done there. Never go into other people's rooms unless necessary.
For getting coffee, Tom Watson should go to the following area in Hobbs Cafe:
Answer: [cafe]
---
{name} is in {current_area} in {current_sector}.
{name} is going to {sector} that has the following areas: [{sector_arenas}]
Stay in the current area if the activity can be done there. Never go into other people's rooms unless necessary.
For {action_description}, {name} should go to the following area in {sector} (MUST pick one of [{sector_arenas}]): ["""


_prompt = PromptTemplate(input_variables=["name",
                                          "current_area",
                                          "current_sector",
                                          "sector",
                                          "sector_arenas",
                                          "action_description"],
                            template=_template)

action_arena_locations_chain = LLMChain(prompt=_prompt, llm=llm)
