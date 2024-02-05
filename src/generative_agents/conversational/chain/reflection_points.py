import asyncio

from pydantic import BaseModel
from generative_agents.conversational.llm import llm
from langchain.chains import LLMChain
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, ChatPromptTemplate


system = "You are reflecting on the subjects in the statements. You need to determine the most salient high-level questions we can answer about the subjects in the statements."

user_shot_1 = """Carlos Sanches is about to participate on a Tennis match;
John Lee had a conversation with his friend Carlos Sanches about the new job offer he received;
John Lee is going to work at the new job offer he received. 
John Lee ate breakfast this morning.

Given only the information above, what are 3 most salient high-level questions we can answer about the subjects in the statements?"""

ai_shot_1 = """What is the relationship between John Lee and Carlos Sanches?
How does Carlos Sanches' participation in a tennis match relate to his interaction with John Lee?
What are the implications of John Lee's new job offer on his personal and professional life?"""


user_shot_2 = """The local community garden is being expanded;
Volunteers, including Susan and Tom, spend weekends working on the garden;
They plant new vegetables and flowers, and install a small pond;
The garden becomes a beautiful spot for relaxation and education in the neighborhood.

Given only the information above, what are 2 most salient high-level questions we can answer about the subjects in the statements?"""

ai_shot_2 = """What impact does the community garden's expansion have on the neighborhood?
How do the efforts of volunteers like Susan and Tom contribute to the development and sustainability of community projects?"""


user = """{{memory}}

Given only the information above, what are {{count}} most salient high-level questions we can answer about the subjects in the statements?"""

class ReflectionPoints(BaseModel):
    memory: str
    count: int

    async def run(self):
        chat_template = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(
                    system, template_format="jinja2"),
                HumanMessagePromptTemplate.from_template(user_shot_1),
                AIMessagePromptTemplate.from_template(ai_shot_1),
                HumanMessagePromptTemplate.from_template(user_shot_2),
                AIMessagePromptTemplate.from_template(ai_shot_2),
                HumanMessagePromptTemplate.from_template(user, template_format="jinja2")])

        poignance_chain = LLMChain(prompt=chat_template, llm=llm, llm_kwargs={"max_tokens": 250,
                                                                              "top_p": 0.95,
                                                                              "temperature": 0.1},
                                   verbose=True)
        completion = await poignance_chain.ainvoke(input={"memory": self.memory, "count": self.count})

        return completion["text"]


async def __tests():
    t = [ReflectionPoints(memory="""Alice Johnson is preparing for her final exams in university;
Mark has been helping Alice study by quizzing her on various subjects;
Alice appreciates Mark's support and feels more confident about her exams;
Mark and Alice plan to celebrate together after the exams are over.""", count=2).run(),
ReflectionPoints(memory="""Emma Thompson has started a new diet to improve her health;
She has been sharing her progress with her friend Sarah, who is a nutritionist;
Sarah offers advice and encourages Emma to stick with her goals;
Emma feels motivated and grateful for Sarah's help and expertise.""", count=4).run(),
ReflectionPoints(memory="""Robert Green is training for a marathon next month;
He runs early in the morning before work and tracks his progress;
His colleague, Linda, notices his dedication and asks for running tips;
Robert is happy to share his training routine and motivate Linda to start running.""", count=3).run(),
    ]

    return await asyncio.gather(*t)

if __name__ == "__main__":
    from pprint import pprint
    t = asyncio.run(__tests())
    pprint(t)
