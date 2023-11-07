from generative_agents.conversational.llm import llm
from langchain import LLMChain, PromptTemplate


_template = """
Conversation:
{conversation}

Summarize the conversation above in one sentence:
This is a conversation about"""

_prompt = PromptTemplate(template=_template,
                         input_variables=["conversation"])

conversation_summary_chain = LLMChain(prompt=_prompt, llm=llm)
