from pydantic import BaseModel
from generative_agents import global_state
from generative_agents.conversational.llm import llm
from langchain import LLMChain, PromptTemplate

_template = """
Here is a brief description of {agent_name}:
{agent_identity}

On the scale of 1 to 10, where 1 is purely mundane (e.g., brushing teeth, making bed) and \
10 is extremely poignant (e.g., a break up, college acceptance), rate the likely poignancy \
of the following {type_} for {agent_name}.

{type_}: {description}
Rating:"""

_prompt = PromptTemplate(input_variables=["agent_name", 
                                          "agent_identity", 
                                          "description", 
                                          "type_"],
                         template=_template)

poignance_chain = LLMChain(prompt=_prompt, llm=llm, llm_kwargs={"max_new_tokens": 2,
                                                                   "do_sample": True,
                                                                   "top_p": 0.95,
                                                                   "top_k": 60,
                                                                   "temperature": 0.1},
                                                                   verbose=True)


class Poingnance(BaseModel):
    agent_name: str
    agent_identity: str
    type_: str
    description: str

    async def run(self):
        poignance_chain.llm_kwargs["cache_key"] = f"2_poignance_{self.agent_name}_{self.description}_{global_state.tick}"

        result = await poignance_chain.arun(agent_name=self.agent_name,
                                                    agent_identity=self.agent_identity,
                                                    type_=self.type_,
                                                    description=self.description)
        try:
            return int(result[-2:])
        except:
            return 1
    