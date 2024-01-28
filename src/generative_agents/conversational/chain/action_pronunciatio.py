import json
import math
import random
import re
from pydantic import BaseModel
from generative_agents import global_state
from generative_agents.conversational.llm import llm
from langchain import LLMChain, PromptTemplate


_template = """<|system|>You follow the tasks given by the user as close as possible. You will only generate 1 JSON object as mentioned below.
<|user|>
Output format: Output a valid json of the following format:
{{
    "emojis": "<valid emoji>"
}}

---

Task: Provide one or two emoji that best represents the following statement or emotion: {action_description}
<|assistant|>
{{
    "emojis": \""""


class ActionPronunciatio(BaseModel):
    action_description: str

    async def run(self):
        _prompt = PromptTemplate(input_variables=["action_description"],
                                 template=_template)

    
        for i in range(5):   
            _action_pronunciatio_chain = LLMChain(prompt=_prompt, llm=llm, llm_kwargs={
            "max_new_tokens": 50,
            "do_sample": True,
            "top_p": 0.90,
            "top_k": 50,
            "temperature": 0.6,
            "cache_key": f"7action_pronunciatio_{self.action_description}_{global_state.tick}"
            }, verbose=global_state.verbose)

            completion = await _action_pronunciatio_chain.arun(action_description=self.action_description)

            pattern = r'<\|assistant\|\>\n*(\{.*?\})'
            match = re.search(pattern, completion, re.DOTALL)
            if match:
                try:
                    json_object = json.loads(match.group(1))
                    return json_object["emojis"]
                except:
                    pass
            
        return "🤷"
    
