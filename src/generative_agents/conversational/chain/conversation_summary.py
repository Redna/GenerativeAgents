import re
from pydantic import BaseModel
from generative_agents import global_state
from generative_agents.conversational.llm import llm
from langchain import LLMChain, PromptTemplate


_template = """<|user|>You stick to the task given by the user and summarizes a conversation in one sentence.
<|user|>
Conversation:
{conversation}

Summarize the conversation above in one sentence:
<|assistant|>This is a conversation about"""

_prompt = PromptTemplate(template=_template,
                         input_variables=["conversation"])

_conversation_summary_chain = LLMChain(prompt=_prompt, llm=llm, llm_kwargs={
            "max_new_tokens": 100,
            "do_sample": True,
            "top_p": 0.95,
            "top_k": 30,
            "temperature": 0.3,
            "repetition_penalty": 1.01}, verbose=global_state.verbose)

class ConversationSummary(BaseModel):
    agent: str
    conversation: str

    #TODO implement retryable decorator
    async def run(self):
        for i in range(5):
            try:
                _conversation_summary_chain.llm_kwargs["cache_key"] = f"conversation_summary_{self.agent}_{global_state.tick}_{i}"
                completion = await _conversation_summary_chain.arun(conversation=self.conversation)
                pattern = r"<|assistant|>(.*)"
                summary = re.findall(pattern, completion)[-1]
                return summary
            except:
                pass
        print("Unable to generate a summary!")
        return "A conversation"