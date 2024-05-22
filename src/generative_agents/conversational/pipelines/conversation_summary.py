from enum import Enum
from typing import Type
from pydantic import BaseModel, Field

from generative_agents.conversational.pipelines.grammar_llm_pipeline import grammar_pipeline

template = """You summarize a conversation in one sentence.

Conversation:
---
{{conversation}}
---

What is the summary of the conversation above? Answer in valid json.
"""


class ConversationSummary(BaseModel):
    summary: str = Field(description="The summary of the conversation")

def conversation_summary(conversation: str) -> str:
    model = ConversationSummary

    summary = grammar_pipeline.run(model=model, prompt_template=template, template_variables={
        "conversation": conversation
    })

    return summary.summary

if __name__ == "__main__":
    print(conversation_summary(conversation="Rudolf: Hello, how are you?\nJoanne: I am fine, thank you."))
    print(conversation_summary(conversation="""Joe Walther: Hello, did you hear about Jim's party?
Frodo Reimsi: No, tell me more. You mean Jimmy Fraser?
Joe Walther: Jim Knofi. He is giving a dinner party.
Frodo Reimsi: I did not know that."""))
    