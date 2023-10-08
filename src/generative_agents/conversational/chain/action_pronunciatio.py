from generative_agents.conversational.llm import llm
from langchain import LLMChain, PromptTemplate

_template = """
Convert an action description to an emoji (important: use two or less emojis).

Action description: {action_description}
Emoji:"""

_prompt = PromptTemplate(input_variables=["action_description"],
                         template=_template)

action_pronunciatio_chain = LLMChain(prompt=_prompt, llm=llm)
