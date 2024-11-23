from pydantic import BaseModel, Field
from langchain_groq.chat_models import ChatGroq
from langchain_core.messages import HumanMessage

llm = ChatGroq(model="llama3-8b-8192", name="chat_relationship")

template = """You are {agent} and judge about your relationships.
Statements:
{statements}

In summary, what do you feel or know about {agent} and {agent_with}'s relationship?
"""

class ChatRelationship(BaseModel):
    relationship_summary: str = Field(
        description="Contains a summary of the relationship between two agents.")

def summarize_chat_relationship(statements: str, agent: str, agent_with: str) -> str:
    structured_llm = llm.with_structured_output(ChatRelationship)

    content = template.format(statements=statements, agent=agent, agent_with=agent_with)
    chat_relationship = structured_llm.invoke([HumanMessage(content=content)])
    return chat_relationship.relationship_summary

if __name__ == "__main__":
        print(summarize_chat_relationship(statements="""Leo and Mia were discussing the organization of the annual community fair;
                                          Mia suggested involving local schools to increase participation;
                                          Leo thought it would be great to include a charity auction;
                                          They decided to meet next Saturday to finalize the event schedule;""",
                                          agent="Leo", agent_with="Mia"))
        print(summarize_chat_relationship(statements="""Ethan and Zoe were outlining the objectives for their joint research project;
                                            Zoe recommended applying for a grant to secure funding;
                                            Ethan offered to draft the research methodology section;
                                            They agreed to compile their findings for a conference presentation;""",
                                            agent="Ethan", agent_with="Zoe"))
        print(summarize_chat_relationship(statements="""Ava and Jack were strategizing for the upcoming basketball tournament;
                                            Jack suggested extra practice sessions to improve teamwork;
                                            Ava thought inviting a coach for a workshop would be beneficial;
                                            They planned to gather the team on Thursday to discuss logistics;""",
                                            agent="Ava", agent_with="Jack"))
