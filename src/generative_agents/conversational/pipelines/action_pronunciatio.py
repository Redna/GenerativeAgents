from pydantic import BaseModel, Field

from langchain_groq.chat_models import ChatGroq
from langchain_core.messages import HumanMessage

llm = ChatGroq(model="llama3-8b-8192",
               name="action_pronunciatio")

template = """Provide one or two emoji that best represents the following statement or emotion: {action_description}"""

class Emoji(BaseModel):
    emoji: str = Field(
        description="Maximum two emojis that best represents the following statement or emotion.")

def action_pronunciatio(action_description: str) -> str:
    structured_llm = llm.with_structured_output(Emoji)

    content = template.format(action_description=action_description)
    emoji = structured_llm.invoke([HumanMessage(content=content)])

    return emoji.emoji

if __name__ == "__main__":
    print(action_pronunciatio(action_description="Taking a shower"))
    print(action_pronunciatio(action_description="Drinking"))
    print(action_pronunciatio(action_description="Taking a bath"))
    print(action_pronunciatio(action_description="Visiting a friend"))
    print(action_pronunciatio(action_description="Walking around"))