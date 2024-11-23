from pydantic import create_model

from langchain_groq.chat_models import ChatGroq
from langchain_core.messages import HumanMessage

llm = ChatGroq(model="llama3-8b-8192", name="evidence_and_insights")

template = """You are infering standalone insights from statments.

Input:
{statements_list}
What {number_of_insights} high-level standalone insights can you infer from the above statements?"""

def evidence_and_insights(statements: list[str], number_of_insights: int) -> list[str]:
    insights = {f"Insight {i}": (str, ...) for i in range(1, number_of_insights + 1)}
    NumberOfInsights = create_model("ReflectionPoints", **insights)

    structured_llm = llm.with_structured_output(NumberOfInsights)

    statements_list = '\n'.join([f"{i+1}. {statement.strip()}" for i, statement in enumerate(statements)])
    content = template.format(statements_list=statements_list, number_of_insights=number_of_insights)

    reflection_points = structured_llm.invoke([HumanMessage(content=content)])

    return [insight for insight in reflection_points.values()]

if __name__ == "__main__":
    print(evidence_and_insights(statements=["John Rossi is a student", "John Rossi is learning a lot",
                    "John Rossi likes discussing with his peers"], number_of_insights=2))
    print(evidence_and_insights(statements=["Sara Johnson is an engineer", "Sara Johnson works on innovative projects",
                                "Sara Johnson enjoys collaborating with her team"], number_of_insights=2))
    print(evidence_and_insights(statements=["David Smith is a chef", "David Smith specializes in Italian cuisine",
                                "David Smith values using fresh ingredients"], number_of_insights=2))
    print(evidence_and_insights(statements=[
                                "Alex Martinez is a biologist", "Alex Martinez studies marine life"], number_of_insights=1))

