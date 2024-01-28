import json
import re
from typing import Optional
from pydantic import BaseModel
from generative_agents import global_state
from generative_agents.conversational.llm import llm
from langchain import LLMChain, PromptTemplate



json_template = """<|system|>You are a JSON expert. There is a JSON object below that is not valid. You will need to fix any syntax errors in the JSON object.
<|user|>
WRONG JSON:
{wrong_json}
ERROR MESSAGE:
{error_message}
<|assistant|>
Here is the valid JSON object:
"""

_prompt = PromptTemplate(input_variables=["wrong_json", "error_message"],
                               template=json_template)

_json_expert_chain = LLMChain(prompt=_prompt, llm=llm, llm_kwargs={
                                                "max_new_tokens": 600, 
                                                "do_sample": True,
                                                "num_beams": 1,
                                                "temperature": 0.6,
                                                "top_p": 0.9,
                                                "top_k": 50},
                                                verbose=global_state.verbose)

class JsonExpert(BaseModel):
    wrong_json: str
    error_message: str

    async def run(self):  
        for i in range(5):   
            _json_expert_chain.llm_kwargs["cache_key"] = f"4json_expert_{self.wrong_json}{global_state.tick}_{i}"

            completion = await _json_expert_chain.arun(wrong_json=self.wrong_json,
                                                       error_message=self.error_message)
            
            pattern = r'Here is the valid JSON object:\n*```json\n*(\{.*?\})\n*```'
            match = re.search(pattern, completion, re.DOTALL)
            if match:
                try:
                    json_object = json.loads(match.group(1))
                    return json_object
                except:
                    pass
        
        raise Exception("Unable to parse JSON object")




