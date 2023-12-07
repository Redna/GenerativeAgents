import asyncio
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
    "Reasoning": "[reasoning for choosing one option]",
    "Answer": "[fill in]" # Option 1 or Option 2
}}
---

Context: {context}
Right now, it is {current_time}. 
{agent} is {agent_observation} when {agent} saw {agent_with} in the middle of {agent_with_observation}.",

Task -- given context and two options that a subject can take, determine which option is the most acceptable.
Let's think step by step. Of the following three options, what should {agent} do?
- Option 1: Wait on {initial_action_description} until {agent_with} is done {agent_with_action}
- Option 2: Continue on to {initial_action_description} now]
<|assistant|>
{{
    "Reasoning": \""""

_prompt = PromptTemplate(input_variables=["context",
                                          "current_time",
                                          "agent", 
                                          "agent_with", 
                                          "agent_with_action",
                                          "agent_observation",
                                          "agent_with_observation",
                                          "initial_action_description",
                                          ],
                         template=_template)

_decide_to_react_chain = LLMChain(prompt=_prompt, llm=llm, llm_kwargs={
        "max_new_tokens": 400,
        "do_sample": True,
        "top_p": 0.95,
        "top_k": 30,
        "temperature": 0.8,
        "repetition_penalty": 1.01}, verbose=global_state.verbose)

class DecideToReact(BaseModel):
    context: str
    current_time: str
    agent: str
    agent_with: str
    agent_with_action: str
    agent_observation: str
    agent_with_observation: str
    initial_action_description: str


    async def run(self):

        tasks = []
        for i in range(3):
            _decide_to_react_chain.llm_kwargs["cache_key"] = f"2decide_to_react_{self.agent}_{self.agent_with}_{global_state.tick}_{i}"
            tasks += [_decide_to_react_chain.arun(context=self.context,
                                                      current_time=self.current_time,
                                                      agent=self.agent,
                                                      agent_with=self.agent_with,
                                                      agent_with_action=self.agent_with_action,
                                                      agent_observation=self.agent_observation,
                                                      agent_with_observation=self.agent_with_observation,
                                                      initial_action_description=self.initial_action_description)]

        completions = await asyncio.gather(*tasks)

        for completion in completions:
            pattern = r'<\|assistant\|\>\n*(\{.*?\})'
            match = re.search(pattern, completion, re.DOTALL)
            if match:
                try:
                    json_object = json.loads(match.group(1))
                    return 1 if "option 1" in json_object["Answer"].lower() else 2
                except:
                    pass
            
        print("Unable to decide to react.")
        return 2