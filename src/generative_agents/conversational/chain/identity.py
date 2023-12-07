import re
from pydantic import BaseModel
from generative_agents import global_state
from generative_agents.conversational.llm import llm
from langchain import LLMChain, PromptTemplate

_template = """<|system|> You write a concise description about {agent}'s personality, family situation and characteristics. You include ALL the details provided in the given context (you MUST include all the names of persons, ages,...). 
Do not make up any details.
<|user|>
Context:
{identity}

Who is {agent}?
<|assistant|>"""


_prompt = PromptTemplate(input_variables=["agent", "identity"],
                            template=_template)

class Identity(BaseModel):
    agent: str
    identity: str

    async def run(self):
        _identity_chain = LLMChain(prompt=_prompt, llm=llm, llm_kwargs={
            "max_new_tokens": 300,
            "do_sample": True,
            "top_p": 0.93,
            "top_k": 40,
            "temperature": 0.4,
            "cache_key": f"6identity_{self.agent}_{global_state.tick}"}, verbose=global_state.verbose)

        completion = await _identity_chain.arun(agent=self.agent, identity=self.identity)

        pattern = r"<|assistant|>[\n ]{0,3}(.*)"
        return re.findall(pattern, completion)[-1]
