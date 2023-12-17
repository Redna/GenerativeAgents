import json
import re
from pydantic import BaseModel
from generative_agents import global_state
from generative_agents.conversational.llm import llm
from langchain import LLMChain, PromptTemplate

_template = """<|system|>You follow the tasks given by the user as close as possible. You will only generate exactly 1 JSON object as mentioned below.
<|user|>

Output format: Output a valid json of the following format:
{{
    "reasoning": "<reasoning for the new object state>",
    "new object state": "<state of an object, should be one signle word>"
}}
---
Task: given context, determine the state of an object that is being used by someone.

Let's think step by step. What is shower's state when John Doe is using it for "getting ready for work"?
{{
    "Reasoning": "John Doe is getting ready to work and using the shower. Hence, the shower is on.",
    "new object state": "on"
}}

---
Task: given context, determine the state of an object that is being used by someone.

Let's think step by step. What is {object_name}'s state when {name} is using it for "{action_description}"?
<|assistant|>
{{
    "Reasoning": \""""


class ObjectActionDescription(BaseModel):
    name: str
    object_name: str
    action_description: str

    async def run(self):

        _prompt = PromptTemplate(input_variables=["object_name",
                                                  "name",
                                                  "action_description"],
                                 template=_template)

        for i in range(5):
            _action_object_chain = LLMChain(prompt=_prompt, llm=llm, llm_kwargs={
                                                              "max_new_tokens":50,
                                                              "do_sample": True,
                                                              "top_p": 0.95,
                                                              "top_k": 10,
                                                              "temperature": 0.4,
                                                              "cache_key": f"_3action_object_chain_{self.name}_{self.object_name}_{self.action_description}_{i}_{global_state.tick}"},
                                                              verbose=global_state.verbose)

            completion = await _action_object_chain.arun(name=self.name, object_name=self.object_name, action_description=self.action_description)
            
            pattern = r'<\|assistant\|\>\n*(\{.*?\})'
            match = re.search(pattern, completion, re.DOTALL)
            if match:
                try:
                    json_object = json.loads(match.group(1))
                    return f"{self.object_name} is {json_object['new object state']}""", (self.object_name, "is", json_object['new object state'])
                except:
                    pass 
        
        print("Unable to generate action event triple.")
        return f"{self.object_name} is idle""", (self.object_name, "is", "idle")
