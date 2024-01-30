import asyncio
import re
import yaml

from typing import List
from pydantic import BaseModel
from generative_agents.conversational.llm import llm
from langchain.chains import LLMChain
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, ChatPromptTemplate

system = """You will act as {init_agent}. Based on a given scenario you whether to initiate a conversation or not. You will only generate exactly 1 valid JSON object as mentioned below.
Output format: Output a valid json of the following format:
Output format:
```yaml
insights: 
{% for i in range(number_of_insights) %} 
    - <insight>: <comma separated reference to statements (e.g. 1,5,3)>
{% endfor %}"""

user = """
Input:
{% for statement in statements %}
    {{ loop.index }}. {{statement}}
{% endfor %}
What {{number_of_insights}} high-level insights can you infer from the above statements?"""

chat_template = ChatPromptTemplate(messages=[
        SystemMessagePromptTemplate.from_template(system, template_format="jinja2"),
        HumanMessagePromptTemplate.from_template(user)],
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

        tasks = []
        for i in range(3):
            f = chat_template.format(statements=self.statements, 
                                     number_of_insights=self.number_of_insights)
            completion = await _evidence_and_insights_chain.ainvoke(input={"statements": self.statements,
                                                                    "number_of_insights": self.number_of_insights})

            pattern = r'\{.*?\}'
            match = re.search(pattern, completion["text"], re.DOTALL)
            if match:
                try:
                    json_object = yaml
                    return "yes" in json_object["answer"].lower()
                except:
                    pass
                
        print("Unable to decide to talk.")
        return False


async def __tests():
    t = [
        EvidenceAndInsightsChain(
            statements=["I am a student", "I am a teacher", "I am a doctor"],
            number_of_insights=2
        ).run()
    ]

    return await asyncio.gather(*t)

if __name__ == "__main__":
    t = asyncio.run(__tests())
    print(t)