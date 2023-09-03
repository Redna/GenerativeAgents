"""
generate_obj_event_v1.txt

Variables: 
!<INPUT 0>! -- Object name 
!<INPUT 1>! -- Persona name
!<INPUT 2>! -- Persona action event description 
!<INPUT 3>! -- Object name 
!<INPUT 4>! -- Object name 

<commentblockmarker>###</commentblockmarker>
Task: We want to understand the state of an object that is being used by someone. 

Let's think step by step. 
We want to know about !<INPUT 0>!'s state. 
Step 1. !<INPUT 1>! is at/using the !<INPUT 2>!.
Step 2. Describe the !<INPUT 3>!'s state: !<INPUT 4>! is"""

from llm import llm
from langchain import LLMChain, PromptTemplate

_template = """
Task: We want to understand the state of an object that is being used by someone.

Let's think step by step.
We want to know about {object_name}'s state.
Step 1. {persona_name} is at/using the {event_description}.
Step 2. Describe the {object_name}'s state: {object_name} is"""

_prompt = PromptTemplate(input_variables=["object_name",
                                          "persona_name",
                                          "event_description"],
                            template=_template)

object_event_chain = LLMChain(prompt=_prompt, llm=llm)
