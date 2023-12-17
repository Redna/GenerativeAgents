import asyncio
import json
import re
from pydantic import BaseModel
from generative_agents import global_state
from generative_agents.conversational.llm import llm
from langchain import LLMChain, PromptTemplate

_template = """###Task: given context, determine whether the subject will initiate a conversation with another. Answere with "yes" or "no".
### Format:
Context: []
Question: []
Reasoning: []
Answer in "yes" or "no": []
---
Context: {context}
Right now, it is {current_time}. {init_agent} and {agent_with} {last_chat_summary}.
{init_agent} is currently {init_agent_observation}
{agent_with_observation}

Question: Would {init_agent} initiate a conversation with {agent_with}?

Reasoning: Let's think step by step.
"""

_template = """<|system|>You follow the tasks given by the user as close as possible. You will only generate exactly 1 JSON object as mentioned below.
<|user|>

Output format: Output a valid json of the following format:
{{
    "Reasoning": "[reasoning for choosing yes or no]",
    "Answer": "[fill in]" # yes or no
}}
---

Context: {context}
Right now, it is {current_time}. {init_agent} and {agent_with} {last_chat_summary}.
{init_agent} is currently {init_agent_observation}
{agent_with} is currently {agent_with_observation}

Task: given context, determine whether the subject will initiate a conversation with another. Answere with "yes" or "no".

Let's think step by step. Would {init_agent} initiate a conversation with {agent_with}?
<|assistant|>
{{
    "Reasoning": \""""

_prompt = PromptTemplate(input_variables=["context",
                                          "current_time",
                                          "init_agent", 
                                          "agent_with", 
                                          "last_chat_summary",  
                                          "init_agent_observation", 
                                          "agent_with_observation"],
                         template=_template)

class DecideToTalk(BaseModel):
    context: str
    current_time: str
    init_agent: str
    agent_with: str
    last_chat_summary: str
    init_agent_observation: str
    agent_with_observation: str

    async def run(self):

        _decide_to_talk_chain = LLMChain(prompt=_prompt, llm=llm, llm_kwargs={
            "max_new_tokens": 350,
            "do_sample": True,
            "top_p": 0.95,
            "top_k": 30,
            "temperature": 0.8,
            "repetition_penalty": 1.01,
            "cache_key": f"decide_to_talk{self.init_agent}_{self.agent_with}_{global_state.tick}"}, verbose=global_state.verbose)

        tasks = []
        for i in range(3):
            tasks += [_decide_to_talk_chain.arun(context=self.context,
                                                      current_time=self.current_time,
                                                      init_agent=self.init_agent,
                                                      agent_with=self.agent_with,
                                                      last_chat_summary=self.last_chat_summary,
                                                      init_agent_observation=self.init_agent_observation,
                                                      agent_with_observation=self.agent_with_observation)]

        completions = await asyncio.gather(*tasks)

        for completion in completions:
            pattern = r'<\|assistant\|\>\n*(\{.*?\})'
            match = re.search(pattern, completion, re.DOTALL)
            if match:
                try:
                    # remove comments if there are any
                    string = re.sub(r'#.*', '', match.group(1))
                    json_object = json.loads(string)
                    return "yes" in json_object["Answer"].lower()
                except:
                    pass
            
        print("Unable to decide to talk.")
        return False