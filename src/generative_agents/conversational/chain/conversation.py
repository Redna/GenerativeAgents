from generative_agents.conversational.llm import llm
from langchain import LLMChain, PromptTemplate


_template = """
Context for the task:

PART 1.
{identity}

Here is the memory that is in {agent}'s head:
{memory}

PART 2.
Past Context:
{past_context}

Current Location: {location}

Current Context:
{agent} was {agent_action} when {agent} saw {agent_with} in the middle of {agent_with_action}.
{agent} is initiating a conversation with {agent_with}.

{agent} and {agent_with} are chatting. Here is their conversation so far:
{conversation}

---
Task: Given the above, what should {agent} say to {agent_with} next in the conversation? And did it end the conversation?

Output format: Output a json of the following format:
{{
"{agent}": "<{agent}'s utterance>",
"Did the conversation end with {agent}'s utterance?": "<json Boolean>"
}}
"""

_prompt = PromptTemplate(template=_template,
                         input_variables=["identity",
                                          "memory", 
                                          "past_context", 
                                          "location",
                                          "agent", 
                                          "agent_action", 
                                          "agent_with", 
                                          "agent_with_action", 
                                          "conversation"])

conversation_chain = LLMChain(prompt=_prompt, llm=llm)
