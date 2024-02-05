import json
import re
from typing import Optional
from pydantic import BaseModel
import yaml
from generative_agents import global_state
from generative_agents.conversational.llm import llm
from langchain.chains import LLMChain
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, ChatPromptTemplate


system = """You are a yaml expert. There is a yaml object below that is not valid. You will need to fix any syntax errors in the yaml object."""

user = """WRONG YAML:
{{wrong_yaml}}
ERROR MESSAGE:
{{error_message}}"""

assistant = """Here is the valid yaml object:
```yaml
"""

chat_template = ChatPromptTemplate(messages=[
    SystemMessagePromptTemplate.from_template(system),
    HumanMessagePromptTemplate.from_template(user, template_format="jinja2"),
    AIMessagePromptTemplate.from_template(assistant)])

_yaml_expert_chain = LLMChain(prompt=chat_template, llm=llm, llm_kwargs={
                                                "max_tokens": 600, 
                                                "temperature": 0.6,
                                                "top_p": 0.9},
                                                verbose=global_state.verbose)

class JsonExpert(BaseModel):
    wrong_yaml: str
    error_message: str

    async def run(self):  
        for i in range(2):   
            completion = await _yaml_expert_chain.ainvoke(input={"wrong_yaml": self.wrong_yaml, "error_message": self.error_message})
            
            pattern = r'```yaml(.*)```'
            match = re.search(pattern, completion["text"], re.DOTALL)
            if match:
                try:
                    corrected_yaml = yaml.safe_load(match.group(1))
                    return corrected_yaml
                except:
                    pass
        
        raise Exception("Unable to parse YAML object")




