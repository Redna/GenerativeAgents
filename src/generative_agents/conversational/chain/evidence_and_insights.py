import asyncio
import re
import yaml

from typing import List
from pydantic import BaseModel
from generative_agents.conversational.llm import llm
from langchain.chains import LLMChain
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, ChatPromptTemplate

system = """You are infering standalone insights from statments. You will only generate exactly 1 valid YAML object as mentioned below.
Output format:
    ```yaml
    insights: {% for i in range(number_of_insights) %}
        - <insight>: <comma separated reference to statements (e.g. 1,5,3)>
    ```
{%- endfor %}"""

user_shot_1 = """Input:
    1. John Rossi is a student
    2. John Rossi is learning a lot
    3. John Rossi likes discussing with his peers
What 2 high-level standalone insights can you infer from the above statements?"""

ai_shot_1 = """```yaml
insights:
    - John Rossi is educated: 1,3,2
    - John Rossi is a social person: 3
```"""

user = """
Input:
{%- for statement in statements %}
    {{ loop.index }}. {{statement | trim}}
{%- endfor %}
What {{number_of_insights}} high-level standalone insights can you infer from the above statements?"""


chat_template = ChatPromptTemplate(messages=[
    SystemMessagePromptTemplate.from_template(system, template_format="jinja2"),
    HumanMessagePromptTemplate.from_template(user_shot_1, template_format="jinja2"),
    AIMessagePromptTemplate.from_template(ai_shot_1, template_format="jinja2"),
    HumanMessagePromptTemplate.from_template(user, template_format="jinja2")],
)

_evidence_and_insights_chain = LLMChain(prompt=chat_template, llm=llm, llm_kwargs={
            "max_tokens": 350,
            "top_p": 0.95,
            "temperature": 0.8,}
            , verbose=True)

class EvidenceAndInsightsChain(BaseModel):
    statements: List[str]
    number_of_insights: int

    async def run(self):
        for _ in range(3):
            completion = await _evidence_and_insights_chain.ainvoke(input={"statements": self.statements,
                                                                            "number_of_insights": self.number_of_insights})

            pattern = r'```yaml(.*)```'
            match = re.search(pattern, completion["text"], re.DOTALL)
            if match:
                try:
                    insights_and_evidences = yaml.safe_load(match.group(1))
                    insights_and_evidences["statements"] = self.statements
                    return insights_and_evidences
                except:
                    pass

        return {"insights": []}

async def __tests():
    t = [
        EvidenceAndInsightsChain(
            statements=["John Rossi is a student", "John Rossi is learning a lot", "John Rossi likes discussing with his peers"],
            number_of_insights=2
        ).run(),
        EvidenceAndInsightsChain(statements=["Sara Johnson is an engineer", "Sara Johnson works on innovative projects", "Sara Johnson enjoys collaborating with her team"], number_of_insights=2).run(),
        EvidenceAndInsightsChain(statements=["David Smith is a chef", "David Smith specializes in Italian cuisine", "David Smith values using fresh ingredients"], number_of_insights=2).run(),
        EvidenceAndInsightsChain(statements=["Alex Martinez is a biologist", "Alex Martinez studies marine life"], number_of_insights=1).run(),
        EvidenceAndInsightsChain(statements=["Rachel Kim is a writer", "Rachel Kim writes fantasy novels", "Rachel Kim draws inspiration from history", "Rachel Kim has a dedicated readership"], number_of_insights=3).run(),
        EvidenceAndInsightsChain(statements=["Kevin Wong is a doctor", "Kevin Wong specializes in cardiology", "Kevin Wong conducts medical research", "Kevin Wong mentors medical students", "Kevin Wong advocates for healthcare innovation"], number_of_insights=4).run()
    ]

    return await asyncio.gather(*t)

if __name__ == "__main__":
    from pprint import pprint
    t = asyncio.run(__tests())
    pprint(t)