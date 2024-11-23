from pydantic import BaseModel, Field

from langchain_groq.chat_models import ChatGroq
from langchain_core.messages import HumanMessage

llm = ChatGroq(model="llama3-8b-8192",
               name="conversation_summary")

template = """Conversation:
---
{conversation}
---
You summarize a conversation in one sentence.
"""


class ConversationSummary(BaseModel):
    summary: str = Field(description="The summary of the conversation")

def conversation_summary(conversation: str) -> str:
    model = ConversationSummary

    structured_llm = llm.with_structured_output(model)
    content = template.format(conversation=conversation)
    summary = structured_llm.invoke([HumanMessage(content=content)])

    return summary.summary

if __name__ == "__main__":
    print(conversation_summary(conversation="Rudolf: Hello, how are you?\nJoanne: I am fine, thank you."))
    print(conversation_summary(conversation="""Joe Walther: Hello, did you hear about Jim's party?
Frodo Reimsi: No, tell me more. You mean Jimmy Fraser?
Joe Walther: Jim Knofi. He is giving a dinner party.
Frodo Reimsi: I did not know that."""))
