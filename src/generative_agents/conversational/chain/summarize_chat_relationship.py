import asyncio
import re
from pydantic import BaseModel
from generative_agents import global_state
from generative_agents.conversational.llm import llm
from langchain.chains import LLMChain
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, ChatPromptTemplate

system = "You act as a person and judge about your relationships"

user_shot_1 = """Statements:
Alex and Samantha discussed their upcoming project deadline;
Samantha mentioned she recently met with Derek to brainstorm ideas;
Alex and Samantha decided to work late at the office to finish their presentation;
They plan to present their project findings at the team meeting next Thursday;

In summary, what do you feel or know about Alex and Samantha's relationship?"""

ai_shot_1 = """Brian and Julia appear to have a close relationship. They are making plans for an extended trip together, which often requires trust, comfort, and familiarity between individuals. The fact that Julia is coordinating with Emma regarding pet care during their absence suggests that they might be living together or at least share responsibilities for their pets. Their joint decision to visit national parks indicates a shared interest in nature and travel. Finally, the mention of finalizing an itinerary by the end of the month shows that they are actively working on this project together. Overall, Brian and Julia seem to have a strong bond, possibly as friends or romantic partners."""

user_shot_2 = """Statements:
Brian and Julia are planning a road trip for the summer;
Julia is coordinating with Emma to ensure pet care during their absence;
Brian suggested stopping at several national parks along the way;
They aim to finalize their itinerary by the end of the month;

In summary, what do you feel or know about Brian and Julia's relationship?"""

ai_shot_2 = """Brian and Julia appear to have a close relationship. They are making plans for an extended trip together, which often requires trust, comfort, and familiarity between individuals. The fact that Julia is coordinating with Emma regarding pet care during their absence suggests that they might be living together or at least share responsibilities for their pets. Their joint decision to visit national parks indicates a shared interest in nature and travel. Finally, the mention of finalizing an itinerary by the end of the month shows that they are actively working on this project together. Overall, Brian and Julia seem to have a strong bond, possibly as friends or romantic partners."""

user_shot_3 = """Statements:
Nathan and Olivia were debating the best strategy for the new marketing campaign;
Olivia shared insights from her previous campaign that could be beneficial;
Nathan proposed scheduling a meeting with the creative team for more ideas;
They agreed to draft a proposal to present to the department head next week;

In summary, what do you feel or know about Nathan and Olivia's relationship?"""

ai_shot_3 = """Nathan and Olivia seem to have a professional relationship, likely as colleagues in the marketing department. They are collaborating on an ongoing project - developing a new marketing campaign. Their discussions involve sharing insights, brainstorming ideas with their team, and preparing proposals for higher management. This indicates that they work closely together and rely on each other's expertise to achieve their common goal."""
user = """Statements:
{{statements}}

In summary, what do you feel or know about {{agent}} and {{agent_with}}'s relationship?"""


class ChatRelationshipSummarization(BaseModel):
    statements: str
    agent: str
    agent_with: str

    async def run(self):
        chat_template = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(
                    system, template_format="jinja2"),
                HumanMessagePromptTemplate.from_template(user_shot_1),
                AIMessagePromptTemplate.from_template(ai_shot_1),
                HumanMessagePromptTemplate.from_template(user_shot_2),
                AIMessagePromptTemplate.from_template(ai_shot_2),
                HumanMessagePromptTemplate.from_template(user_shot_3),
                AIMessagePromptTemplate.from_template(ai_shot_3),
                HumanMessagePromptTemplate.from_template(user, template_format="jinja2")])

        _chat_relationship_summary = LLMChain(prompt=chat_template, llm=llm, llm_kwargs={"max_tokens": 250,
                                                                                         "top_p": 0.95,
                                                                                         "temperature": 0.1},
                                              verbose=True)

        completion = await _chat_relationship_summary.ainvoke(input={"agent": self.agent,
                                                                     "agent_with": self.agent_with,
                                                                     "statements": self.statements})
        return completion["text"]


async def __tests():
    t = [
        ChatRelationshipSummarization(statements="""Leo and Mia were discussing the organization of the annual community fair;
Mia suggested involving local schools to increase participation;
Leo thought it would be great to include a charity auction;
They decided to meet next Saturday to finalize the event schedule;""",
                                      agent="Leo", agent_with="Mia").run(),
        ChatRelationshipSummarization(statements="""Ethan and Zoe were outlining the objectives for their joint research project;
Zoe recommended applying for a grant to secure funding;
Ethan offered to draft the research methodology section;
They agreed to compile their findings for a conference presentation;""",
                                      agent="Ethan", agent_with="Zoe").run(),
        ChatRelationshipSummarization(statements="""Ava and Jack were strategizing for the upcoming basketball tournament;
Jack suggested extra practice sessions to improve teamwork;
Ava thought inviting a coach for a workshop would be beneficial;
They planned to gather the team on Thursday to discuss logistics;""",
                                      agent="Ava", agent_with="Jack").run(),
        ChatRelationshipSummarization(statements="""Sarah and Mark were brainstorming ideas for their new music album;
Mark shared a concept for a song inspired by their recent travels;
Sarah proposed using unconventional instruments to add uniqueness;
They agreed to start recording demos over the weekend;""",
                                      agent="Sarah", agent_with="Mark").run(),
    ]

    return await asyncio.gather(*t)

if __name__ == "__main__":
    from pprint import pprint
    t = asyncio.run(__tests())
    pprint(t)
