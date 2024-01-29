"""
summarize_chat_relationship_v2.txt

Variables: 
!<INPUT 0>! -- Statements
!<INPUT 1>! -- curr persona name
!<INPUT 2>! -- target_persona.scratch.name

<commentblockmarker>###</commentblockmarker>
[Statements]
!<INPUT 0>!

Based on the statements above, summarize !<INPUT 1>! and !<INPUT 2>!'s relationship. What do they feel or know about each other?
"""

import re
from pydantic import BaseModel
from generative_agents import global_state
from generative_agents.conversational.llm import llm
from langchain import LLMChain, PromptTemplate

_template = """<|system|>Based on the statements below, summarize {agent} and {agent_with}'s relationship.
<|user|>
[Statements]
{statements}

What do they feel or know about each other?
<|assistant|>
"""

_prompt = PromptTemplate(input_variables=["statements",
                                            "agent",
                                            "agent_with"],
                             template=_template)

class ChatRelationshipSummarization(BaseModel):
    statements: str
    agent: str
    agent_with: str

    async def run(self):
        _summarize_chat_relationship_chain = LLMChain(prompt=_prompt, llm=llm, llm_kwargs={
            "max_tokens": 300,

            "top_p": 0.95,
            "temperature": 0.8},
            verbose=global_state.verbose)


        completion = await _summarize_chat_relationship_chain.arun(statements=self.statements,
                                                        agent=self.agent,
                                                        agent_with=self.agent_with)
        
        pattern = rf"<|assistant|>\n?(.*)"
        match = re.findall(pattern, completion)[-1]
        return match