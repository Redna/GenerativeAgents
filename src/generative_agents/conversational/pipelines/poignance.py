from functools import lru_cache
from pydantic import BaseModel, Field

from langchain_groq.chat_models import ChatGroq
from langchain_core.messages import HumanMessage

llm = ChatGroq(model="llama3-8b-8192", name="poignance")

template = """You are {agent_name}. You are rating the importance of an event.

Here is a brief description of {agent_name}:
{agent_identity}

How would you rate the {type_} "{description}"?
"""

class Poignance(BaseModel):
    rating: int = Field(ge=0, lt=11, description="Rating of the poignance of the event. (0-10)")

@lru_cache(maxsize=2048)
def rate_poignance(agent_name: str, agent_identity: str, type_: str, description: str) -> int:
    structured_llm = llm.with_structured_output(Poignance)

    content = template.format(agent_name=agent_name, agent_identity=agent_identity, type_=type_, description=description)
    poignance = structured_llm.invoke([HumanMessage(content=content)])

    return poignance.rating

def __tests():
    print(rate_poignance("Emily Tan",
                         "Emily Tan is a renowned sculptor with over a decade of experience. Her works explore themes of nature and human connection. She has showcased her art in various galleries around the world.",
                         "Exhibition",
                         "Emily Tan is preparing for her solo art exhibition at the local gallery."))
    print(rate_poignance("Marcus Lee",
                            "Marcus Lee is a young, dynamic author known for his captivating novels that blend mystery with deep psychological insights. His storytelling has garnered a loyal following.",
                            "Book Launch",
                            "Marcus Lee is hosting a book launch for his latest novel."))
    print(rate_poignance("Sara Ahmed",
                            "Sara Ahmed is a dedicated social worker and community organizer. She has spent years working with underprivileged communities, focusing on education and healthcare initiatives.",
                            "Community Service",
                            "Sara Ahmed is organizing a community service event."))
    print(rate_poignance("Leo Gonzalez",
                            "Leo Gonzalez is an environmental activist known for his passionate advocacy for climate change action. He has been involved in numerous campaigns to promote sustainability.",
                            "Campaign",
                            "Leo Gonzalez is leading a campaign to raise awareness about the importance of renewable energy sources."))

if __name__ == "__main__":
    __tests()
